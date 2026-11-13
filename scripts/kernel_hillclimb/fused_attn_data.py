#!/usr/bin/env python3
"""Real attention-block activations + sub-models for lab/kernel-hillclimb/fused-attn.html.

For one Lc0 big net (docs/bignet_fused_attention_2026-11.md) this writes, under
OUT/<net>/:

  attn_orig.onnx   the encoder's attention core exactly as exported (the node slice):
                   inputs Q/K/V [B*64, D] (after the bias Add) and the smolgen bias
                   [B, H, 64, 64]; output the out/reshape tensor [B*64, D]
  attn_mha.onnx    the same core after scripts/kernel_hillclimb/mha_rewrite.py
                   (com.microsoft MultiHeadAttention, ORT WebGPU FlashAttention)
  L<i>.{q,k,v,bias}.f16   inputs captured from ORT CPU on the full net for 8 fixture
                   positions (fixtures/lc0/fen_only.json, planes from scripts/kernel_hillclimb/dump_planes.mjs)
  L<i>.out_cpu.f16 ORT CPU output of attn_orig for those inputs
  L<i>.ref.f32     float64 numpy reference of the same math from the f16 inputs
  manifest.json

Usage (repo onnx venv, isolated):
  .venv-onnx/bin/python -I scripts/kernel_hillclimb/fused_attn_data.py --net bt4 --model M.onnx \
      --mha-model M.mha.onnx --planes DIR --layers 0,7,14 --out artifacts/kernel-hillclimb/fused-attn
"""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def add_outputs(model, names):
    g = model.graph
    have = {o.name for o in g.output}
    for n in names:
        if n not in have:
            g.output.append(helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT16, None))
    return model


def extract(src_path, dst_path, inputs, output):
    """Nodes between `inputs` and `output` (onnx.utils.extract_model needs shape
    inference, which does not know com.microsoft MultiHeadAttention). IO is f16;
    shapes are left symbolic."""
    m = onnx.load(src_path)
    g = m.graph
    by_out = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    keep, stack, seen = [], [output], set()
    while stack:
        t = stack.pop()
        if t in seen or t in inputs or t in inits or t == "":
            continue
        seen.add(t)
        n = by_out[t]
        if n.name not in {k.name for k in keep}:
            keep.append(n)
            stack.extend(n.input)
    order = {n.name: i for i, n in enumerate(g.node)}
    keep.sort(key=lambda n: order[n.name])
    used = {i for n in keep for i in n.input}
    graph = helper.make_graph(
        keep,
        "attn_core",
        [helper.make_tensor_value_info(i, onnx.TensorProto.FLOAT16, None) for i in inputs],
        [helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT16, None)],
        [inits[i] for i in sorted(used) if i in inits],
    )
    sub = helper.make_model(graph, opset_imports=list(m.opset_import))
    sub.ir_version = m.ir_version
    onnx.save(sub, dst_path)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mha-model", required=True)
    ap.add_argument("--mha32-model", help="mha_rewrite.py --compute f32 output (adds the attn_mha32 sub-model)")
    ap.add_argument("--planes", required=True)
    ap.add_argument("--layers", default="0,7,14")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = os.path.join(args.out, args.net)
    os.makedirs(out_dir, exist_ok=True)

    planes = np.fromfile(os.path.join(args.planes, "planes.f32"), dtype=np.float32).reshape(-1, 112, 8, 8)
    model = onnx.load(args.model)
    inp = model.graph.input[0]
    batch = inp.type.tensor_type.shape.dim[0].dim_value or 8
    x = planes[:batch].astype(np.float16)

    names = {}
    for L in layers:
        p = f"/encoder{L}"
        names[L] = {
            "q": f"{p}/mha/Q/b",
            "k": f"{p}/mha/K/b",
            "v": f"{p}/mha/V/b",
            "bias": f"{p}/smolgen/out/reshape",
            "out": f"{p}/mha/out/reshape",
        }
    tapped = add_outputs(model, [n for d in names.values() for n in d.values()])
    tap_path = os.path.join(out_dir, "_tapped.onnx")
    onnx.save(tapped, tap_path)
    sess = ort.InferenceSession(tap_path, providers=["CPUExecutionProvider"])
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp.name: x})))
    os.remove(tap_path)

    # Sub-models: the attention core of layer layers[0] (all layers share the structure;
    # only the scale constant and shape constants are copied).
    L0 = layers[0]
    n0 = names[L0]
    orig_path = os.path.join(out_dir, "attn_orig.onnx")
    mha_path = os.path.join(out_dir, "attn_mha.onnx")
    sub_inputs = [n0["q"], n0["k"], n0["v"], n0["bias"]]
    extract(args.model, orig_path, sub_inputs, n0["out"])
    extract(args.mha_model, mha_path, sub_inputs, n0["out"])
    sub_models = {"orig": "attn_orig.onnx", "mha": "attn_mha.onnx"}
    if args.mha32_model:
        extract(args.mha32_model, os.path.join(out_dir, "attn_mha32.onnx"), sub_inputs, n0["out"])
        sub_models["mha32"] = "attn_mha32.onnx"
    sub_orig = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    sub_mha = ort.InferenceSession(mha_path, providers=["CPUExecutionProvider"])

    heads = outs[n0["bias"]].shape[1]
    d_model = outs[n0["q"]].shape[1]
    depth = d_model // heads
    scale = None
    for init in model.graph.initializer:
        if init.name == f"/encoder{L0}/mha/QK/scale/w":
            scale = float(onnx.numpy_helper.to_array(init).reshape(-1)[0])
    manifest = {
        "net": args.net,
        "model": os.path.basename(args.model),
        "batch": batch,
        "tokens": 64,
        "heads": heads,
        "depth": depth,
        "dModel": d_model,
        "scale": scale,
        "inputNames": {k: n0[k] for k in ("q", "k", "v", "bias")},
        "outputName": n0["out"],
        "subModels": sub_models,
        "layers": [],
    }
    for L in layers:
        n = names[L]
        q, k, v, bias = (outs[n[s]].astype(np.float16) for s in ("q", "k", "v", "bias"))
        feeds = {n0["q"]: q, n0["k"]: k, n0["v"]: v, n0["bias"]: bias}
        out_cpu = sub_orig.run(None, feeds)[0].astype(np.float16)
        out_mha_cpu = sub_mha.run(None, feeds)[0].astype(np.float16)
        # Exact reference in float64 from the f16 inputs (scale as the f16 constant).
        B = batch
        qh = q.astype(np.float64).reshape(B, 64, heads, depth).transpose(0, 2, 1, 3)
        kh = k.astype(np.float64).reshape(B, 64, heads, depth).transpose(0, 2, 3, 1)
        vh = v.astype(np.float64).reshape(B, 64, heads, depth).transpose(0, 2, 1, 3)
        s = (qh @ kh) * scale + bias.astype(np.float64)
        ref = (softmax(s) @ vh).transpose(0, 2, 1, 3).reshape(B * 64, d_model).astype(np.float32)
        full = outs[n["out"]].astype(np.float32)
        files = {}
        for key, arr, ext in (
            ("q", q, "f16"),
            ("k", k, "f16"),
            ("v", v, "f16"),
            ("bias", bias, "f16"),
            ("outCpu", out_cpu, "f16"),
            ("ref", ref, "f32"),
        ):
            fn = f"L{L}.{key}.{ext}"
            arr.tofile(os.path.join(out_dir, fn))
            files[key] = fn
        stats = {
            "cpuSubVsFullNet": float(np.abs(out_cpu.astype(np.float32) - full).max()),
            "cpuOrigVsRef": float(np.abs(out_cpu.astype(np.float32) - ref).max()),
            "cpuMhaVsRef": float(np.abs(out_mha_cpu.astype(np.float32) - ref).max()),
            "refAbsMax": float(np.abs(ref).max()),
            "scoreAbsMax": float(np.abs(s).max()),
            "biasAbsMax": float(np.abs(bias.astype(np.float32)).max()),
        }
        manifest["layers"].append({"layer": L, "files": files, "stats": stats})
        print(json.dumps({"layer": L, **stats}))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"status": "FUSED_ATTN_DATA_DONE", "out": out_dir, "layers": layers}))


if __name__ == "__main__":
    main()
