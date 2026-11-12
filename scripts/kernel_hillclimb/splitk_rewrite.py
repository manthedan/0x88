#!/usr/bin/env python3
"""Split-K rewrite of small-M weight MatMuls, to steer ORT WebGPU to wider grids.

ORT 1.27's WebGPU MatMul tiles the output (32 columns x 32 rows, or 32 x 8
when M <= 8) and puts the batch on grid z; its built-in split-K is enabled
only on Intel adapters. At Lc0 Small batch 1 (M = 64 token rows, or M = 1 for
the smolgen and head GEMVs) that leaves 4-64 workgroups per matmul. This pass
rewrites

    Y[M,N] = X[M,K] . W[K,N]

as a batched matmul over S slices of K plus a reduction:

    X  -> Reshape [M,S,K/S] -> Transpose [S,M,K/S]    (M == 1: Reshape [S,1,K/S])
    W  -> [S,K/S,N]   (same bytes: the int8 / f16 initializer is only re-dimensioned)
    P  =  MatMul(X', W')                  [S,M,N]   grid (N/32, M/32 or M/8, S)
    Y  =  ReduceSum(P, axes=[0])          [M,N]     (or MatMul(ones[1,S], P[S,N]) for M == 1)

so downstream nodes (the bias Add etc.) are untouched. For weight-only QDQ
models the DequantizeLinear stays in place with the per-column scale axis
moved from 1 to 2, so src/nn/onnxDequantFold.ts still folds it to an f16
initializer at load time and the download grows only by the new shape
constants. Transposed copies of a shared input (Q/K/V/smolgen read the same
LayerNorm output) are emitted once per (input, S).

S is the smallest divisor of K with K/S a multiple of 4 (ORT's vec4 path),
K/S >= --min-inner, and (N/32) * rowTiles * S >= --target-wg.

Run with the repo's onnx venv, isolated:
  .venv-onnx/bin/python -I scripts/kernel_hillclimb/splitk_rewrite.py IN.onnx OUT.onnx \
      [--target-wg 128] [--min-inner 32] [--max-s 64] [--include REGEX] [--exclude REGEX] \
      [--rows all|gemv|tokens] [--gemv-reduce reducesum|matmul] [--check]
"""
import argparse
import json
import re
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

# ORT 1.27 webgpu/math/matmul.cc ComputeMatMul: workgroup 8x8x1, 4 output
# columns per thread, 4 rows per thread unless M <= 8 (then 1).
ORT_TILE_N = 32


def ort_row_tile(m):
    return 8 if m <= 8 else 32


def ceil_div(a, b):
    return -(-a // b)


def choose_splits(m, k, n, target_wg, min_inner, max_s):
    base = ceil_div(n, ORT_TILE_N) * ceil_div(m, ort_row_tile(m))
    best = 1
    for s in range(2, max_s + 1):
        if k % s or (k // s) % 4 or k // s < min_inner:
            continue
        best = s
        if base * s >= target_wg:
            break
    return best, base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--target-wg', type=int, default=128, help='minimum workgroups per split matmul (default 128)')
    ap.add_argument('--min-inner', type=int, default=32, help='minimum K/S per slice (default 32, ORT tile depth)')
    ap.add_argument('--max-s', type=int, default=64)
    ap.add_argument('--include', default=None, help='only rewrite MatMuls whose name matches this regex')
    ap.add_argument('--exclude', default=None, help='skip MatMuls whose name matches this regex')
    ap.add_argument('--rows', choices=['all', 'gemv', 'tokens'], default='all', help='gemv: only M == 1; tokens: only M > 1')
    ap.add_argument('--gemv-reduce', choices=['reducesum', 'matmul'], default='reducesum',
                    help='M == 1 reduction: ReduceSum (ORT adds a transpose + shared reduce) or MatMul with ones[1,S] (ORT fuses the bias Add into it)')
    ap.add_argument('--reduce', choices=['reducesum', 'matmul'], default='reducesum',
                    help='M > 1 reduction: ReduceSum over S (naive, one dispatch) or MatMul ones[1,S] . P[S,M*N]')
    ap.add_argument('--assume-m', type=int, default=0,
                    help='rows to plan for when a 2-D input has a dynamic first dim (0: skip such MatMuls)')
    ap.add_argument('--max-base-wg', type=int, default=0,
                    help='only split MatMuls whose unsplit ORT grid has fewer workgroups than this (0: no limit)')
    ap.add_argument('--min-weight', type=int, default=4096, help='skip weights with fewer than K*N elements (default 4096)')
    ap.add_argument('--rank3', action='store_true', help='also rewrite MatMuls whose input is [1,...,1,M,K] (smolgen weight gen)')
    ap.add_argument('--check', action='store_true', help='run original and rewrite on ORT CPU and print max-abs per output')
    ap.add_argument('--summary', default=None, help='write a JSON summary of the rewritten nodes here')
    args = ap.parse_args()

    model = onnx.load(args.src)
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = [d.dim_value if d.HasField('dim_value') else None for d in v.type.tensor_type.shape.dim]
        shapes[v.name] = (dims, v.type.tensor_type.elem_type)
    for t in g.initializer:
        shapes[t.name] = (list(t.dims), t.data_type)
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    inc = re.compile(args.include) if args.include else None
    exc = re.compile(args.exclude) if args.exclude else None
    new_inits = []
    const_cache = {}
    split_input_cache = {}
    rewritten = []
    uid = [0]

    def const_i64(values):
        key = ('i64', tuple(values))
        if key not in const_cache:
            name = f'/splitk/const/shape_{"_".join(str(v) for v in values)}'
            new_inits.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))
            const_cache[key] = name
        return const_cache[key]

    def const_ones(s, elem_type):
        key = ('ones', s, elem_type)
        if key not in const_cache:
            dt = np.float16 if elem_type == TensorProto.FLOAT16 else np.float32
            name = f'/splitk/const/ones_{s}_{elem_type}'
            new_inits.append(numpy_helper.from_array(np.ones((1, s), dtype=dt), name))
            const_cache[key] = name
        return const_cache[key]

    # Pass 1: eligible MatMuls, grouped by the weight tensor they read (the smolgen
    # weight-gen DequantizeLinear feeds all ten encoders, so its slices are shared).
    groups = {}
    for node in g.node:
        if node.op_type != 'MatMul':
            continue
        if inc and not inc.search(node.name):
            continue
        if exc and exc.search(node.name):
            continue
        a_name, b_name = node.input[0], node.input[1]
        a_shape, a_type = shapes.get(a_name, (None, None))
        if a_shape and len(a_shape) == 2 and a_shape[0] is None and a_shape[1] is not None and args.assume_m:
            # Dynamic batch (LQO): plan with --assume-m rows, emit Reshape [-1,S,K/S] + ReduceSum.
            a_shape = [-1, a_shape[1]]
        if not a_shape or len(a_shape) < 2 or None in a_shape:
            continue
        # Leading unit dims (smolgen weight gen is [1,8,256]) fold into M; other batched inputs are skipped.
        if any(d != 1 for d in a_shape[:-2]) or (len(a_shape) > 2 and not args.rank3):
            continue
        m, k = a_shape[-2], a_shape[-1]
        if m == -1:
            m = args.assume_m
        if args.rows == 'gemv' and m != 1:
            continue
        if args.rows == 'tokens' and m == 1:
            continue
        groups.setdefault(b_name, []).append((node, a_name, a_shape, a_type, m, k))

    # Pass 2: per weight, check every consumer is eligible with one (M,K,N), pick S, re-dimension.
    plan = {}
    for b_name, uses in groups.items():
        dq = producer.get(b_name)
        if len(consumers.get(b_name, [])) != len(uses):
            continue
        if b_name in inits:
            w_init, kind = inits[b_name], 'init'
        elif dq is not None and dq.op_type == 'DequantizeLinear' and all(i in inits for i in dq.input if i):
            w_init, kind = inits[dq.input[0]], 'dq'
            if len(consumers.get(w_init.name, [])) != 1:
                continue
            scale = inits[dq.input[1]]
            axis = next((a.i for a in dq.attribute if a.name == 'axis'), 1)
            block = next((a.i for a in dq.attribute if a.name == 'block_size'), 0)
            per_tensor = int(np.prod(scale.dims)) == 1
            if block or not (per_tensor or axis in (1, -1)):
                continue
        else:
            continue
        if len(w_init.dims) != 2:
            continue
        kw, n = w_init.dims
        if kw * n < args.min_weight or len({(u[4], u[5]) for u in uses}) != 1 or uses[0][5] != kw:
            continue
        m, k = uses[0][4], uses[0][5]
        s, base_wg = choose_splits(m, k, n, args.target_wg, args.min_inner, args.max_s)
        if s == 1 or (args.max_base_wg and base_wg >= args.max_base_wg):
            continue
        ks = k // s
        # Weight: re-dimension in place (row-major [K,N] == [S,K/S,N]).
        del w_init.dims[:]
        w_init.dims.extend([s, ks, n])
        if kind == 'dq':
            for a in dq.attribute:
                if a.name == 'axis':
                    a.i = 2
            if not any(a.name == 'axis' for a in dq.attribute) and not per_tensor:
                dq.attribute.append(helper.make_attribute('axis', 2))
        for node, a_name, a_shape, a_type, _, _ in uses:
            plan[node.name] = (a_name, a_shape, a_type, m, k, n, s, base_wg, kind)

    # Pass 3: emit replacement nodes in place (topological order holds: a shared
    # Reshape/Transpose is emitted before its first consumer).
    replace = {}
    for node in g.node:
        if node.name not in plan:
            continue
        a_name, a_shape, a_type, m, k, n, s, base_wg, kind = plan[node.name]
        dynamic = a_shape[0] == -1
        b_name = node.input[1]
        ks = k // s
        uid[0] += 1
        tag = f'/splitk/{uid[0]}'
        nodes = []
        # Activation: [S,M,K/S], shared across consumers of the same input.
        key = (a_name, s)
        if key not in split_input_cache:
            out = f'{tag}/a_split'
            if m == 1 and not dynamic:
                nodes.append(helper.make_node('Reshape', [a_name, const_i64([s, 1, ks])], [out], name=f'{tag}/a_reshape'))
            else:
                mid = f'{tag}/a_mks'
                nodes.append(helper.make_node('Reshape', [a_name, const_i64([-1 if dynamic else m, s, ks])], [mid], name=f'{tag}/a_reshape'))
                nodes.append(helper.make_node('Transpose', [mid], [out], name=f'{tag}/a_transpose', perm=[1, 0, 2]))
            split_input_cache[key] = out
        a_split = split_input_cache[key]
        partial = f'{tag}/partial'
        nodes.append(helper.make_node('MatMul', [a_split, b_name], [partial], name=f'{node.name}/splitk{s}'))
        y = node.output[0]
        # Rank > 2 outputs are reduced to [M,N] and reshaped back to [1,..,M,N].
        y_mn = y if len(a_shape) == 2 else f'{tag}/y_mn'
        mode = 'reducesum' if dynamic else args.gemv_reduce if m == 1 else args.reduce
        if mode == 'matmul':
            # Y = ones[1,S] . P[S, M*N]. For M == 1 ORT fuses the following bias Add into a Gemm.
            flat = f'{tag}/partial_flat'
            nodes.append(helper.make_node('Reshape', [partial, const_i64([s, m * n])], [flat], name=f'{tag}/partial_reshape'))
            direct = m == 1 and len(a_shape) == 2  # [1,N] is already the output shape
            summed = y if direct else f'{tag}/sum_flat'
            nodes.append(helper.make_node('MatMul', [const_ones(s, a_type), flat], [summed], name=f'{tag}/reduce_matmul'))
            if not direct:
                nodes.append(helper.make_node('Reshape', [summed, const_i64(a_shape[:-1] + [n])], [y], name=f'{tag}/y_reshape'))
        else:
            nodes.append(helper.make_node('ReduceSum', [partial, const_i64([0])], [y_mn], name=f'{tag}/reduce', keepdims=0))
            if y_mn != y:
                nodes.append(helper.make_node('Reshape', [y_mn, const_i64(a_shape[:-1] + [n])], [y], name=f'{tag}/y_reshape'))
        replace[node.name] = nodes
        rewritten.append({
            'node': node.name, 'M': m, 'K': k, 'N': n, 'S': s, 'kPerSlice': ks,
            'workgroupsBefore': base_wg, 'workgroupsAfter': base_wg * s, 'weight': kind,
        })

    out_nodes = []
    for node in g.node:
        out_nodes.extend(replace.get(node.name, [node]))
    del g.node[:]
    g.node.extend(out_nodes)
    g.initializer.extend(new_inits)
    # Stale value_info (DQ outputs changed rank) would mislead ORT's shape checks.
    del g.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    summary = {
        'src': args.src, 'dst': args.dst,
        'options': {k: v for k, v in vars(args).items() if k not in ('src', 'dst', 'summary', 'check')},
        'rewritten': rewritten,
    }
    for r in rewritten:
        print(f"{r['node']:44s} M={r['M']:<3d} K={r['K']:<5d} N={r['N']:<5d} S={r['S']:<3d} wg {r['workgroupsBefore']}->{r['workgroupsAfter']}")
    print(f'{len(rewritten)} MatMuls rewritten -> {args.dst}', file=sys.stderr)
    if args.check:
        summary['cpuCheck'] = cpu_check(args.src, args.dst)
        print(json.dumps(summary['cpuCheck']), file=sys.stderr)
    if args.summary:
        with open(args.summary, 'w') as f:
            json.dump(summary, f, indent=1)


def cpu_check(src, dst):
    """Original vs rewrite on ORT CPU (start position + random planes); max-abs per output."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(src, so, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(dst, so, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    out = {}
    for meta in a.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
        dtype = np.float16 if 'float16' in meta.type else np.float32
        feeds = {meta.name: (rng.random(shape) < 0.1).astype(dtype)}
    ra = a.run(None, feeds)
    rb = b.run(None, feeds)
    for o, x, y in zip(a.get_outputs(), ra, rb):
        out[o.name] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
    return out


if __name__ == '__main__':
    main()
