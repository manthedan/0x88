#!/usr/bin/env python3
"""Fixed-batch int32 export of the tiny squareformer ONNX for WebNN.

WebNN (ort-web webnn EP) requires static shapes and rejects int64 tensors
outright. This script onnxsim-folds the dynamic model at a fixed batch, then
converts data-path int64 to int32 while keeping the tensors that the ONNX
spec forces to int64 (Unsqueeze/Reshape/Slice/... axes and shape operands).
Conversion validated bit-exact vs the original on ORT CPU (2026-06-10).

Run from leelaweb with .venv-onnx:
  .venv-onnx/bin/python scripts/convert_tiny_onnx_fixed_i32.py --batch 1
  .venv-onnx/bin/python scripts/convert_tiny_onnx_fixed_i32.py --batch 16
"""
from __future__ import annotations
import argparse

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxsim import simplify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--in", dest="src", default="public/models/bt4_anneal_muon_best.onnx")
parser.add_argument("--out", dest="dst", default=None)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()
dst = args.dst or f"public/models/bt4_anneal_muon_best.batch{args.batch}.sim.i32.onnx"

model = onnx.load(args.src)
sim, ok = simplify(model, overwrite_input_shapes={
    "tokens": [args.batch, 64, 24],
    "attack_summary": [args.batch, 64, 28],
})
assert ok, "onnxsim check failed"
graph = sim.graph

# (op_type, input_index) slots the ONNX spec forces to int64.
REQ64 = {("Unsqueeze", 1), ("Squeeze", 1), ("Reshape", 1), ("Expand", 1),
         ("Slice", 1), ("Slice", 2), ("Slice", 3), ("Slice", 4),
         ("Tile", 1), ("Pad", 1), ("Pad", 2), ("ConstantOfShape", 0),
         ("ReduceSum", 1), ("ReduceMax", 1), ("ReduceMean", 1), ("TopK", 1), ("Split", 1)}
keep64 = set()
for node in graph.node:
    for idx, name in enumerate(node.input):
        if (node.op_type, idx) in REQ64:
            keep64.add(name)

converted = 0
for init in graph.initializer:
    if init.data_type == TensorProto.INT64 and init.name not in keep64:
        arr = numpy_helper.to_array(init)
        assert np.abs(arr).max() < 2**31, init.name
        init.CopyFrom(numpy_helper.from_array(arr.astype(np.int32), name=init.name))
        converted += 1
for vi in graph.input:
    if vi.type.tensor_type.elem_type == TensorProto.INT64 and vi.name not in keep64:
        vi.type.tensor_type.elem_type = TensorProto.INT32
del graph.value_info[:]  # optional hints; ORT re-infers the int32 paths
for node in graph.node:
    if node.op_type == "Constant" and node.output[0] not in keep64:
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == TensorProto.INT64:
                attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.int32)))
                converted += 1
onnx.save(sim, dst)
print(f"WROTE {dst} (batch={args.batch}, {converted} int64 tensors -> int32, kept {len(keep64)} spec-int64)")
