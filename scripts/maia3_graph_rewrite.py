#!/usr/bin/env python3
"""Graph rewrites of the shipped Maia3 ONNX (maia3_simplified.qdq8.onnx) for ORT WebGPU.

Each pass is an exact algebraic rewrite of the original graph: weights keep
their dtype and the int8 QDQ weights stay in the file (the browser dequant fold,
src/nn/onnxDequantFold.ts, still sees the same DequantizeLinear pattern), so the
download size is unchanged to within a few KB.

Passes (--passes, comma-separated, applied in this order):

  einsum   `Einsum bhi,oi->bho` against an initializer W[o,i] (the shared
           attention-bias projection, 8 nodes) -> MatMul(X, W^T) with W^T stored
           once as a new initializer. ORT WebGPU runs the Einsum at ~77 GFLOP/s
           on the Elo grid (b11); MatMul is its tiled GEMM.
  promo    The promotion-logit head: 256 scalar chains
           Gather(Gather(S,i),j) + Gather(Gather(P,k),l) -> Unsqueeze, feeding a
           257-input Concat. ORT WebGPU dispatches ~400 single-workgroup kernels
           for it (half the b1 invoke). Rewritten as
           Gather(S_flat, idx_s) + Gather(P_flat, idx_p) -> Concat(2 inputs):
           the same f16 additions on the same operands, so bit-identical.
  splitk   Gemm(X[B,K], W[N,K], bias, transB=1) with an initializer weight,
           K >= --splitk-min-k and N <= --splitk-max-n (the smolgen compress
           Gemm, 1x2048 -> 128, 4 workgroups at b1) -> split-K:
           Reshape(X,[B,S,1,K/S]) @ W^T[S,K/S,N] -> ReduceSum(axis 1) -> + bias.
           Changes the f16 accumulation order (not bit-identical); gated.
           --splitk-f32 accumulates the partials in f32 instead.
  static   Fix the batch dimension to --batch N on every graph input/output,
           then constant-fold everything that becomes constant (Shape/Gather/
           Unsqueeze/Concat shape arithmetic, ConstantOfShape, the Elo-embedding
           row gathers). DequantizeLinear is never folded (QDQ weights stay int8).

Run with the repo's onnx venv, isolated:
  <checkout>/.venv-onnx/bin/python -I scripts/maia3_graph_rewrite.py IN.onnx OUT.onnx \
      --passes einsum,promo [--batch 1] [--report OUT.report.json]
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator

PASSES = ("einsum", "promo", "splitk", "static")


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


class Graph:
    """Small mutable view: name -> initializer array, producer and consumer maps."""

    def __init__(self, model):
        self.model = model
        self.g = model.graph
        self.refresh()

    def refresh(self):
        self.inits = {t.name: t for t in self.g.initializer}
        self.producer = {o: n for n in self.g.node for o in n.output if o}
        self.consumers = defaultdict(list)
        for n in self.g.node:
            for i in n.input:
                if i:
                    self.consumers[i].append(n)

    def const(self, name):
        """Array value of an initializer or Constant node output, else None."""
        if name in self.inits:
            return numpy_helper.to_array(self.inits[name])
        n = self.producer.get(name)
        if n is not None and n.op_type == "Constant":
            v = attr(n, "value")
            if v is not None:
                return numpy_helper.to_array(v)
        return None

    def unique(self, base):
        taken = set(self.inits) | set(self.producer) | {i.name for i in self.g.input}
        name, k = base, 0
        while name in taken:
            k += 1
            name = f"{base}_{k}"
        return name

    def add_init(self, name, array):
        t = numpy_helper.from_array(np.array(array, order="C"), name)  # np.array keeps 0-d (ascontiguousarray does not)
        self.g.initializer.append(t)
        self.inits[name] = t
        return name

    def prune(self):
        """Drop nodes and initializers that no graph output depends on."""
        live = {o.name for o in self.g.output}
        keep = []
        for n in reversed(list(self.g.node)):
            if any(o in live for o in n.output):
                keep.append(n)
                live.update(i for i in n.input if i)
        keep.reverse()
        removed_nodes = len(self.g.node) - len(keep)
        del self.g.node[:]
        self.g.node.extend(keep)
        inits = [t for t in self.g.initializer if t.name in live]
        removed_inits = len(self.g.initializer) - len(inits)
        del self.g.initializer[:]
        self.g.initializer.extend(inits)
        self.refresh()
        return removed_nodes, removed_inits


def replace_node(graph, old, new_nodes):
    """Splice `new_nodes` in place of `old` (matched by its first output: protobuf
    repeated fields copy messages, so object identity does not survive edits)."""
    nodes = list(graph.g.node)
    idx = next(i for i, n in enumerate(nodes) if n.output[0] == old.output[0])
    nodes[idx : idx + 1] = new_nodes
    del graph.g.node[:]
    graph.g.node.extend(nodes)
    graph.refresh()


def keep_nodes(graph, predicate):
    nodes = [n for n in graph.g.node if predicate(n)]
    del graph.g.node[:]
    graph.g.node.extend(nodes)
    graph.refresh()


# --------------------------------------------------------------------- einsum
def pass_einsum(graph):
    rewritten = []
    transposed = {}
    for node in [n for n in graph.g.node if n.op_type == "Einsum"]:
        eq = attr(node, "equation").decode().replace(" ", "")
        if eq != "bhi,oi->bho":
            continue
        w = graph.const(node.input[1])
        if w is None or w.ndim != 2:
            raise SystemExit(f"einsum: {node.name} weight {node.input[1]} is not a 2-D constant")
        if node.input[1] not in transposed:
            transposed[node.input[1]] = graph.add_init(graph.unique(f"{node.input[1]}_T"), w.T.copy())
        mm = helper.make_node("MatMul", [node.input[0], transposed[node.input[1]]], list(node.output), name=f"{node.name}_as_MatMul")
        replace_node(graph, node, [mm])
        rewritten.append(node.name)
    removed = graph.prune()
    return {"rewritten": rewritten, "transposedWeights": list(transposed), "pruned": removed}


# ---------------------------------------------------------------------- promo
def scalar_gather_chain(graph, name):
    """Follow Gather(axis=1, scalar const index) chains: returns (base, [indices])."""
    idx = []
    while True:
        n = graph.producer.get(name)
        if n is None or n.op_type != "Gather":
            break
        k = graph.const(n.input[1])
        if k is None or k.ndim != 0 or attr(n, "axis", 0) != 1:
            break
        idx.append(int(k))
        name = n.input[0]
    return name, idx[::-1]


def static_dims(graph, name):
    inferred = shape_inference.infer_shapes(graph.model)
    for v in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        if v.name == name:
            return [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
    raise SystemExit(f"promo: no inferred shape for {name}")


def pass_promo(graph):
    cats = [n for n in graph.g.node if n.op_type == "Concat" and len(n.input) > 64]
    if len(cats) != 1:
        raise SystemExit(f"promo: expected one wide Concat, found {len(cats)}")
    cat = cats[0]
    axis = attr(cat, "axis")
    head, tail = cat.input[0], list(cat.input[1:])
    pairs = []
    for name in tail:
        u = graph.producer.get(name)
        if u is None or u.op_type != "Unsqueeze" or list(graph.const(u.input[1]).reshape(-1)) != [1]:
            raise SystemExit(f"promo: {name} is not Unsqueeze(axes=[1])")
        add = graph.producer.get(u.input[0])
        if add is None or add.op_type != "Add":
            raise SystemExit(f"promo: {u.input[0]} is not an Add")
        pairs.append([scalar_gather_chain(graph, add.input[0]), scalar_gather_chain(graph, add.input[1])])
    bases = {(a[0], b[0]) for a, b in pairs}
    if len(bases) != 1:
        raise SystemExit(f"promo: operands come from several bases {bases}")
    base_s, base_p = next(iter(bases))
    dims_s, dims_p = static_dims(graph, base_s)[1:], static_dims(graph, base_p)[1:]
    if None in dims_s or None in dims_p:
        raise SystemExit(f"promo: non-static trailing dims {dims_s} {dims_p}")

    def flat(indices, dims):
        if len(indices) != len(dims):
            raise SystemExit(f"promo: index chain {indices} does not cover dims {dims}")
        return int(np.ravel_multi_index([i % d for i, d in zip(indices, dims)], dims))

    idx_s = np.array([flat(a[1], dims_s) for a, _ in pairs], dtype=np.int64)
    idx_p = np.array([flat(b[1], dims_p) for _, b in pairs], dtype=np.int64)
    # The head input is Reshape(base_s, [-1, prod(dims_s)]); reuse it as S_flat.
    reshape = graph.producer.get(head)
    if reshape is None or reshape.op_type != "Reshape" or reshape.input[0] != base_s:
        raise SystemExit("promo: Concat head is not a Reshape of the square-pair logits")
    flat_shape = graph.add_init(graph.unique("promo_flat_shape"), np.array([0, -1], dtype=np.int64))
    i_s = graph.add_init(graph.unique("promo_idx_square_pair"), idx_s)
    i_p = graph.add_init(graph.unique("promo_idx_piece_bias"), idx_p)
    p_flat = graph.unique("/model/promo/piece_bias_flat")
    g_s, g_p, logits = graph.unique("/model/promo/square_pair"), graph.unique("/model/promo/piece_bias"), graph.unique("/model/promo/logits")
    new = [
        helper.make_node("Reshape", [base_p, flat_shape], [p_flat], name="/model/promo/Reshape"),
        helper.make_node("Gather", [head, i_s], [g_s], name="/model/promo/GatherSquarePair", axis=1),
        helper.make_node("Gather", [p_flat, i_p], [g_p], name="/model/promo/GatherPieceBias", axis=1),
        helper.make_node("Add", [g_s, g_p], [logits], name="/model/promo/Add"),
        helper.make_node("Concat", [head, logits], list(cat.output), name=cat.name, axis=axis),
    ]
    replace_node(graph, cat, new)
    removed = graph.prune()
    return {"promotionLogits": len(pairs), "squarePairBase": base_s, "pieceBiasBase": base_p, "pruned": removed}


# --------------------------------------------------------------------- splitk
def pass_splitk(graph, chunks, min_k, max_n, f32=False):
    rewritten = []
    for node in [n for n in graph.g.node if n.op_type == "Gemm"]:
        if attr(node, "transA", 0) or not attr(node, "transB", 0) or attr(node, "alpha", 1.0) != 1.0 or attr(node, "beta", 1.0) != 1.0:
            continue
        w = graph.const(node.input[1])
        bias = graph.const(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if w is None or w.ndim != 2:
            continue
        n_out, k = w.shape
        if k < min_k or n_out > max_n or k % chunks:
            continue
        if bias is not None and bias.shape != (n_out,):
            continue
        kc = k // chunks
        w_split = graph.add_init(graph.unique(f"{node.input[1]}_splitk{chunks}"), w.T.reshape(chunks, kc, n_out).copy())
        x_shape = graph.add_init(graph.unique(f"{node.name}_splitk_xshape"), np.array([0, chunks, 1, kc], dtype=np.int64))
        out_shape = graph.add_init(graph.unique(f"{node.name}_splitk_oshape"), np.array([0, n_out], dtype=np.int64))
        axes = graph.add_init(graph.unique(f"{node.name}_splitk_axes"), np.array([1], dtype=np.int64))
        base = node.name
        xr, part, red, flat = (graph.unique(f"{base}_splitk_{s}") for s in ("x", "partial", "sum", "flat"))
        new = [helper.make_node("Reshape", [node.input[0], x_shape], [xr], name=f"{base}_splitk_Reshape")]
        if f32 and w.dtype == np.float16:
            # f32 partials: the f16 weight stays in the file; ORT constant-folds its Cast at load.
            xr32, w32, part32, red32 = (graph.unique(f"{base}_splitk_{s}_f32") for s in ("x", "w", "partial", "sum"))
            new += [
                helper.make_node("Cast", [xr], [xr32], name=f"{base}_splitk_CastX", to=TensorProto.FLOAT),
                helper.make_node("Cast", [w_split], [w32], name=f"{base}_splitk_CastW", to=TensorProto.FLOAT),
                helper.make_node("MatMul", [xr32, w32], [part32], name=f"{base}_splitk_MatMul"),
                helper.make_node("ReduceSum", [part32, axes], [red32], name=f"{base}_splitk_ReduceSum", keepdims=0),
                helper.make_node("Cast", [red32], [red], name=f"{base}_splitk_CastSum", to=TensorProto.FLOAT16),
            ]
        else:
            new += [
                helper.make_node("MatMul", [xr, w_split], [part], name=f"{base}_splitk_MatMul"),
                helper.make_node("ReduceSum", [part, axes], [red], name=f"{base}_splitk_ReduceSum", keepdims=0),
            ]
        if bias is not None:
            new += [
                helper.make_node("Reshape", [red, out_shape], [flat], name=f"{base}_splitk_Flatten"),
                helper.make_node("Add", [flat, node.input[2]], list(node.output), name=f"{base}_splitk_Bias"),
            ]
        else:
            new.append(helper.make_node("Reshape", [red, out_shape], list(node.output), name=f"{base}_splitk_Flatten"))
        replace_node(graph, node, new)
        rewritten.append({"node": base, "k": int(k), "n": int(n_out), "chunks": chunks, "f32Partials": bool(f32 and w.dtype == np.float16)})
    removed = graph.prune()
    return {"rewritten": rewritten, "pruned": removed}


# --------------------------------------------------------------------- static
FOLD_MAX_ELEMENTS = 1 << 16
NEVER_FOLD = {"DequantizeLinear", "QuantizeLinear"}


def eval_node(node, values, opset):
    inputs = [i for i in node.input if i]
    vis = [helper.make_tensor_value_info(i, helper.np_dtype_to_tensor_dtype(values[i].dtype), values[i].shape) for i in inputs]
    outs = [helper.make_tensor_value_info(o, TensorProto.UNDEFINED, None) for o in node.output]
    model = helper.make_model(helper.make_graph([node], "fold", vis, outs), opset_imports=opset)
    model.ir_version = 10
    return ReferenceEvaluator(model).run(None, {i: values[i] for i in inputs})


def pass_static(graph, batch):
    g = graph.g
    for vi in list(g.input) + list(g.output):
        dims = vi.type.tensor_type.shape.dim
        if not dims or dims[0].HasField("dim_value"):
            continue
        dims[0].Clear()
        dims[0].dim_value = batch
    del g.value_info[:]
    # Constant nodes -> initializers so folding sees one kind of constant.
    for n in g.node:
        if n.op_type == "Constant":
            graph.add_init(n.output[0], numpy_helper.to_array(attr(n, "value")))
    keep_nodes(graph, lambda n: n.op_type != "Constant")
    folded = Counter()
    opset = list(graph.model.opset_import)
    for _ in range(64):
        inferred = shape_inference.infer_shapes(graph.model, strict_mode=True, data_prop=True)
        shapes = {}
        for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
            dims = v.type.tensor_type.shape.dim
            if v.type.tensor_type.HasField("shape") and all(d.HasField("dim_value") for d in dims):
                shapes[v.name] = [d.dim_value for d in dims]
        done = set()
        for node in g.node:
            if node.op_type in NEVER_FOLD:
                continue
            outs = None
            if node.op_type == "Shape" and node.input[0] in shapes:
                shape = np.array(shapes[node.input[0]], dtype=np.int64)
                start, end = attr(node, "start", 0), attr(node, "end", None)
                outs = [shape[start:end]]
            elif node.op_type == "Size" and node.input[0] in shapes:
                outs = [np.array(int(np.prod(shapes[node.input[0]])), dtype=np.int64)]
            elif all(i in graph.inits for i in node.input if i):
                out_numel = [int(np.prod(shapes[o])) if o in shapes else None for o in node.output]
                if any(n is None or n > FOLD_MAX_ELEMENTS for n in out_numel):
                    continue
                values = {i: numpy_helper.to_array(graph.inits[i]) for i in node.input if i}
                outs = eval_node(node, values, opset)
            if outs is None:
                continue
            for name, value in zip(node.output, outs):
                graph.add_init(name, np.asarray(value))
            done.add(node.output[0])
            folded[node.op_type] += 1
        keep_nodes(graph, lambda n: n.output[0] not in done)
        if not done:
            break
    removed = graph.prune()
    return {"batch": batch, "folded": dict(folded), "pruned": removed}


# ----------------------------------------------------------------------- main
def op_counts(model):
    return dict(Counter(n.op_type for n in model.graph.node).most_common())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--passes", default="einsum,promo", help=f"comma-separated subset of {','.join(PASSES)}")
    p.add_argument("--batch", type=int, help="static batch for the `static` pass")
    p.add_argument("--splitk-chunks", type=int, default=16)
    p.add_argument("--splitk-min-k", type=int, default=1024)
    p.add_argument("--splitk-max-n", type=int, default=256)
    p.add_argument("--splitk-f32", action="store_true", help="accumulate split-K partials in f32 (Cast in, Cast out)")
    p.add_argument("--report", help="write a JSON report (passes, op counts, sizes, sha256)")
    args = p.parse_args(argv)
    passes = [s for s in args.passes.split(",") if s]
    unknown = set(passes) - set(PASSES)
    if unknown:
        p.error(f"unknown passes {sorted(unknown)}")
    if "static" in passes and not args.batch:
        p.error("the static pass needs --batch")

    model = onnx.load(args.input)
    before = op_counts(model)
    graph = Graph(model)
    report = {"input": args.input, "passes": {}}
    for name in PASSES:  # fixed order regardless of how --passes lists them
        if name not in passes:
            continue
        if name == "einsum":
            report["passes"][name] = pass_einsum(graph)
        elif name == "promo":
            report["passes"][name] = pass_promo(graph)
        elif name == "splitk":
            report["passes"][name] = pass_splitk(graph, args.splitk_chunks, args.splitk_min_k, args.splitk_max_n, args.splitk_f32)
        elif name == "static":
            report["passes"][name] = pass_static(graph, args.batch)
    del model.graph.value_info[:]
    model.producer_name = (model.producer_name or "") + "+maia3_graph_rewrite"
    model.doc_string = f"maia3_graph_rewrite passes={','.join(p for p in PASSES if p in passes)}" + (f" batch={args.batch}" if "static" in passes else "")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, args.output)
    data = open(args.output, "rb").read()
    report.update(
        {
            "output": args.output,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "nodes": {"before": sum(before.values()), "after": len(model.graph.node)},
            "opsBefore": before,
            "opsAfter": op_counts(model),
        }
    )
    text = json.dumps(report, indent=2)
    if args.report:
        with open(args.report, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
