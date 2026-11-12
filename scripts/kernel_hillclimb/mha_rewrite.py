#!/usr/bin/env python3
"""Rewrite Lc0 big-net attention blocks into com.microsoft MultiHeadAttention.

Lab tooling for docs/bignet_fused_attention_2026-11.md (candidate #2 of
docs/kernel_hillclimb_roofline_2026-11.md). Each encoder's attention core, as
exported by lc0's ONNX converter,

    Q/K/V [B*64, D] -> Reshape [B,64,H,d] -> Transpose (Q,V: 0213; K: 0231)
    QK = MatMul(Q, K) * scale + smolgen_bias[B,H,64,64]
    P  = Softmax(QK, axis=3)
    O  = MatMul(P, V) -> Transpose 0213 -> Reshape [B*64, D]

becomes

    Q/K/V [B*64, D] -> Reshape [B,64,D]
    O' = MultiHeadAttention(q, k, v, bias='', key_padding_mask='', attention_bias=smolgen_bias,
                            num_heads=H, scale=scale)                     [B,64,D]
    O  = Reshape(O', [B*64, D])        (same tensor name as the old out/reshape)

The Q/K/V bias Adds stay outside MHA on purpose: ORT fuses MatMul+Add into Gemm
already, and passing MHA's `bias` input would add TransferBSDToBNSH dispatches.
In ORT 1.27's WebGPU EP this shape (head_size % 4 == 0, no packed QKV) runs
contrib_ops/webgpu/bert/flash_attention.cc: one CopyKVCache dispatch plus one
FlashAttention dispatch per layer, instead of 4 Transposes + 2 MatMuls + Mul +
Add + Softmax. Note that kernel accumulates QK, softmax and P.V in the input
type (f16 for these nets), so the gate is the WASM parity + tie-epsilon one.

Usage (repo onnx venv, isolated):
  .venv-onnx/bin/python -I scripts/kernel_hillclimb/mha_rewrite.py IN.onnx OUT.onnx [--layers 0-14] [--check]
--check runs the original and the rewrite on ORT CPU for a few random-plane
batches and prints the output max-abs (the CPU MHA kernel is f32 internally).
"""
import argparse
import json
import re
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PAT = re.compile(r"^/encoder(\d+)/mha/QK/softmax$")


def parse_layers(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", help="encoder indices to rewrite, e.g. 0-14 or 0,3 (default all)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    layers = parse_layers(args.layers)

    model = onnx.load(args.src)
    g = model.graph
    by_out = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def producer(name, op):
        n = by_out.get(name)
        if n is None or n.op_type != op:
            raise ValueError(f"expected {op} producing {name}, got {n.op_type if n else None}")
        return n

    removed = set()
    new_nodes = {}  # anchor node name -> nodes to insert in its place
    shape_inits = []
    rewritten = []
    for n in g.node:
        m = PAT.match(n.name)
        if not m:
            continue
        layer = int(m.group(1))
        if layers is not None and layer not in layers:
            continue
        p = f"/encoder{layer}/mha"
        softmax = n
        bias_add = producer(softmax.input[0], "Add")
        scale_mul = producer(bias_add.input[0], "Mul")
        smolgen = bias_add.input[1]
        qk = producer(scale_mul.input[0], "MatMul")
        scale = float(numpy_helper.to_array(inits[scale_mul.input[1]]).reshape(-1)[0])
        q_t = producer(qk.input[0], "Transpose")
        k_t = producer(qk.input[1], "Transpose")
        (av,) = consumers[softmax.output[0]]
        assert av.op_type == "MatMul" and av.input[0] == softmax.output[0]
        v_t = producer(av.input[1], "Transpose")
        (out_t,) = consumers[av.output[0]]
        (out_r,) = consumers[out_t.output[0]]
        assert out_t.op_type == "Transpose" and out_r.op_type == "Reshape"
        perms = {t.name: list(helper.get_attribute_value(t.attribute[0])) for t in (q_t, k_t, v_t, out_t)}
        assert perms[q_t.name] == [0, 2, 1, 3] and perms[v_t.name] == [0, 2, 1, 3]
        assert perms[k_t.name] == [0, 2, 3, 1] and perms[out_t.name] == [0, 2, 1, 3]
        q_r, k_r, v_r = (producer(t.input[0], "Reshape") for t in (q_t, k_t, v_t))
        head_shape = numpy_helper.to_array(inits[q_r.input[1]]).tolist()  # [-1, 64, H, d]
        _, tokens, heads, depth = head_shape
        d_model = heads * depth
        # Only the attention-internal tensors may disappear.
        for t in (q_r, k_r, v_r, q_t, k_t, v_t, qk, scale_mul, bias_add, softmax, av, out_t):
            for o in t.output:
                users = [c for c in consumers.get(o, []) if c.name not in {x.name for x in (q_t, k_t, v_t, qk, scale_mul, bias_add, softmax, av, out_t, out_r)}]
                assert not users, f"{o} has outside consumers {[u.name for u in users]}"
        removed.update(x.name for x in (q_r, k_r, v_r, q_t, k_t, v_t, qk, scale_mul, bias_add, softmax, av, out_t, out_r))
        bsd = f"/const{p}/mha_bsd_shape"
        shape_inits.append(numpy_helper.from_array(np.array([-1, tokens, d_model], dtype=np.int64), bsd))
        nodes = []
        names = []
        for which, r in (("Q", q_r), ("K", k_r), ("V", v_r)):
            out = f"{p}/{which}/bsd"
            nodes.append(helper.make_node("Reshape", [r.input[0], bsd], [out], name=f"{p}/{which}/bsd_reshape"))
            names.append(out)
        mha_out = f"{p}/mha_fused"
        nodes.append(
            helper.make_node(
                "MultiHeadAttention",
                [names[0], names[1], names[2], "", "", smolgen],
                [mha_out],
                name=f"{p}/MultiHeadAttention",
                domain="com.microsoft",
                num_heads=heads,
                scale=scale,
            )
        )
        nodes.append(helper.make_node("Reshape", [mha_out, out_r.input[1]], [out_r.output[0]], name=f"{p}/out/reshape"))
        new_nodes[out_r.name] = nodes
        rewritten.append({"layer": layer, "heads": heads, "depth": depth, "scale": scale, "smolgen": smolgen})

    if not rewritten:
        sys.exit("no attention blocks matched")
    # Rebuild the node list topologically: the MHA group goes where the old
    # output Reshape was (all its inputs are produced earlier).
    kept = []
    for n in g.node:
        if n.name in new_nodes:
            kept.extend(new_nodes[n.name])
        elif n.name not in removed:
            kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    g.initializer.extend(shape_inits)
    used = {i for n in g.node for i in n.input}
    for init in list(g.initializer):
        if init.name not in used:
            g.initializer.remove(init)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(json.dumps({"status": "MHA_REWRITE_DONE", "dst": args.dst, "blocks": len(rewritten), "nodes": len(g.node), "first": rewritten[0]}))
    if args.check:
        cpu_check(args.src, args.dst)


def cpu_check(src, dst):
    import onnxruntime as ort

    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    inp = a.get_inputs()[0]
    shape = [d if isinstance(d, int) else 8 for d in inp.shape]
    dtype = np.float16 if "float16" in inp.type else np.float32
    rng = np.random.default_rng(0)
    worst = {}
    for _ in range(2):
        x = (rng.random(shape) < 0.1).astype(dtype)
        ra = a.run(None, {inp.name: x})
        rb = b.run(None, {inp.name: x})
        for o, va, vb in zip(a.get_outputs(), ra, rb):
            worst[o.name] = max(worst.get(o.name, 0.0), float(np.max(np.abs(va.astype(np.float32) - vb.astype(np.float32)))))
    print(json.dumps({"status": "MHA_CPU_CHECK", "maxAbs": worst}))


if __name__ == "__main__":
    main()
