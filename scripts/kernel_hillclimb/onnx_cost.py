#!/usr/bin/env python3
"""Per-node FLOP / byte cost of an ONNX model at a fixed batch, for the kernel
hill-climb roofline (scripts/kernel_hillclimb/roofline.mjs joins it with the
measured per-dispatch profile).

Run with the repo's onnx venv, isolated:
  .venv-onnx/bin/python -I scripts/kernel_hillclimb/onnx_cost.py MODEL.onnx --batch 8 [--out cost.json]

Conventions (documented so later candidates are judged on the same scale):
  * MatMul / Gemm: 2*M*N*K per output batch; Einsum: 2*prod(all index sizes);
    Conv: 2*out*Cin/g*kh*kw.
  * Elementwise ops: 1 FLOP per output element (transcendentals count 1);
    Softmax 5/elem, LayerNormalization 8/elem, reductions 1 per input elem.
  * Layout ops (Transpose, Reshape, Concat, Slice, Gather, ...) are 0 FLOP but
    their input+output bytes still count (they are real dispatches in ORT).
  * DequantizeLinear on an initializer is treated as folded (ORT dequant fold
    is default ON): 0 FLOP, its output becomes an f16 weight of the consumer.
  * bytes = compulsory traffic: every distinct input tensor once + outputs,
    in the tensor's element size. Weights are reported separately.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
}
LAYOUT = {'Transpose', 'Reshape', 'Concat', 'Slice', 'Gather', 'GatherElements', 'Split', 'Squeeze', 'Unsqueeze', 'Flatten', 'Expand', 'Cast', 'Identity', 'Tile', 'Pad', 'Shape', 'Constant', 'ConstantOfShape', 'Range', 'ScatterND', 'Where'}
PER_ELEM = {'Softmax': 5, 'LogSoftmax': 5, 'LayerNormalization': 8, 'SkipLayerNormalization': 9, 'SimplifiedLayerNormalization': 6, 'Gelu': 8, 'FastGelu': 8, 'QuickGelu': 4, 'Erf': 4, 'Mish': 6, 'Softplus': 3, 'Tanh': 2, 'Sigmoid': 3, 'Exp': 1, 'Log': 1, 'Sqrt': 1, 'Pow': 1}
REDUCE = {'ReduceMean', 'ReduceSum', 'ReduceMax', 'ReduceMin', 'ReduceL2', 'ReduceSumSquare', 'GlobalAveragePool'}


def fix_batch(model, batch):
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        for i, d in enumerate(dims):
            if d.dim_param or (i == 0 and d.dim_value in (0, -1)):
                d.Clear()
                d.dim_value = batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--out')
    args = ap.parse_args()

    model = onnx.load(args.model)
    fix_batch(model, args.batch)
    model = shape_inference.infer_shapes(model, data_prop=True)
    g = model.graph

    info = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        t = vi.type.tensor_type
        dims = [d.dim_value if d.dim_value > 0 else None for d in t.shape.dim]
        info[vi.name] = (dims, t.elem_type)
    inits = {}
    for init in g.initializer:
        inits[init.name] = (list(init.dims), init.data_type)
        info[init.name] = (list(init.dims), init.data_type)
    for node in g.node:
        if node.op_type == 'Constant':
            for a in node.attribute:
                if a.name == 'value':
                    info[node.output[0]] = (list(a.t.dims), a.t.data_type)
                    inits[node.output[0]] = (list(a.t.dims), a.t.data_type)

    # Folded DequantizeLinear: output is an f16 weight initializer.
    folded = set()
    for node in g.node:
        if node.op_type == 'DequantizeLinear' and node.input[0] in inits:
            dims, _ = inits[node.input[0]]
            out_type = info.get(node.output[0], (None, TensorProto.FLOAT16))[1] or TensorProto.FLOAT16
            # The fold emits f16 for f16 graphs; keep the inferred type when known.
            inits[node.output[0]] = (dims, out_type)
            info[node.output[0]] = (dims, out_type)
            folded.add(node.name or node.output[0])

    def numel(name):
        dims, _ = info.get(name, (None, None))
        if dims is None or any(d is None for d in dims):
            return None
        return int(np.prod(dims)) if dims else 1

    def nbytes(name):
        n = numel(name)
        if n is None:
            return None
        return n * DTYPE_BYTES.get(info[name][1], 4)

    nodes = []
    unknown = 0
    for idx, node in enumerate(g.node):
        op = node.op_type
        if op in ('Constant',) or (op == 'DequantizeLinear' and node.input[0] in inits):
            continue
        ins = [i for i in node.input if i]
        outs = [o for o in node.output if o]
        out_elems = sum(numel(o) or 0 for o in outs)
        flops = 0
        mnk = None
        if op in ('MatMul', 'FusedMatMul', 'MatMulInteger'):
            a, b = info.get(ins[0], (None, None))[0], info.get(ins[1], (None, None))[0]
            if a and b and None not in a and None not in b:
                k = a[-1]
                n = b[-1] if len(b) > 1 else 1
                m = a[-2] if len(a) > 1 else 1
                batch_elems = (numel(outs[0]) or 0) // max(1, m * n)
                flops = 2 * (numel(outs[0]) or 0) * k
                mnk = {'batch': batch_elems, 'M': m, 'N': n, 'K': k}
            else:
                unknown += 1
        elif op == 'Gemm':
            a = info.get(ins[0], (None, None))[0]
            if a and None not in a:
                trans_a = next((x.i for x in node.attribute if x.name == 'transA'), 0)
                k = a[0] if trans_a else a[-1]
                flops = 2 * (numel(outs[0]) or 0) * k + (numel(outs[0]) or 0) * (len(ins) > 2)
                o = info[outs[0]][0]
                mnk = {'batch': 1, 'M': o[0], 'N': o[1], 'K': k}
        elif op == 'Einsum':
            eq = next((x.s.decode() for x in node.attribute if x.name == 'equation'), '')
            lhs = eq.replace(' ', '').split('->')[0].split(',')
            sizes = {}
            for term, name in zip(lhs, ins):
                dims = info.get(name, (None, None))[0]
                if dims and len(dims) == len(term):
                    sizes.update({c: d for c, d in zip(term, dims) if d})
            if sizes and all(c in sizes for term in lhs for c in term):
                flops = 2 * int(np.prod(list(sizes.values())))
                mnk = {'einsum': eq}
            else:
                unknown += 1
        elif op in ('Conv', 'FusedConv'):
            w = info.get(ins[1], (None, None))[0]
            group = next((x.i for x in node.attribute if x.name == 'group'), 1)
            if w:
                flops = 2 * (numel(outs[0]) or 0) * int(np.prod(w[1:]))
        elif op in REDUCE:
            flops = numel(ins[0]) or 0
        elif op in LAYOUT:
            flops = 0
        else:
            flops = PER_ELEM.get(op, 1) * out_elems
        weight_bytes = sum(nbytes(i) or 0 for i in set(ins) if i in inits)
        act_in = sum(nbytes(i) or 0 for i in set(ins) if i not in inits)
        out_bytes = sum(nbytes(o) or 0 for o in outs)
        nodes.append({
            'index': idx,
            'op': op,
            'name': node.name,
            'flops': int(flops),
            'weightBytes': int(weight_bytes),
            'actInBytes': int(act_in),
            'outBytes': int(out_bytes),
            'bytes': int(weight_bytes + act_in + out_bytes),
            'mnk': mnk,
            'outShape': info.get(outs[0], (None, None))[0] if outs else None,
        })

    by_op = defaultdict(lambda: {'nodes': 0, 'flops': 0, 'bytes': 0, 'weightBytes': 0})
    for n in nodes:
        b = by_op[n['op']]
        b['nodes'] += 1
        b['flops'] += n['flops']
        b['bytes'] += n['bytes']
        b['weightBytes'] += n['weightBytes']
    total_w = sum(n['weightBytes'] for n in nodes)
    result = {
        'schema': 'kernel-hillclimb.onnx-cost.v1',
        'model': args.model,
        'batch': args.batch,
        'foldedDequantize': len(folded),
        'unknownShapeMatmuls': unknown,
        'totals': {
            'flops': sum(n['flops'] for n in nodes),
            'bytes': sum(n['bytes'] for n in nodes),
            'weightBytes': total_w,
            'nodes': len(nodes),
        },
        'byOp': dict(sorted(by_op.items(), key=lambda kv: -kv[1]['flops'])),
        'nodes': nodes,
    }
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')
    t = result['totals']
    print(json.dumps({'status': 'ONNX_COST', 'model': args.model.split('/')[-1], 'batch': args.batch, 'gflops': t['flops'] / 1e9, 'mbytes': t['bytes'] / 1e6, 'weightMB': t['weightBytes'] / 1e6, 'nodes': t['nodes'], 'unknown': unknown}))


if __name__ == '__main__':
    sys.exit(main())
