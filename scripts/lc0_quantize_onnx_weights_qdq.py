#!/usr/bin/env python3
"""Weight-only int8 QDQ quantization for the ORT lane.

Rewrites an f16 LC0 ONNX model so large MatMul weight initializers are stored
as per-output-channel int8 with f16 scales, dequantized in-graph via
DequantizeLinear. Compute stays f16 (the DQL output feeds the original
MatMul), so this is the ORT-lane twin of the TVMJS detached-int8 tensor-cache:
the win is download/storage, not kernels. Small or non-MatMul initializers
stay raw f16 (same threshold rationale as lc0_quantize_tensor_cache.py).

DequantizeLinear with f16 scales requires opset >= 19; the model's ai.onnx
opset is bumped to 21 (LC0 graphs use no ops with breaking changes in
17 -> 21).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--in", dest="src", required=True)
parser.add_argument("--out", dest="dst", required=True)
parser.add_argument("--min-elements", type=int, default=4096)
parser.add_argument("--report", default=None)
args = parser.parse_args()

model = onnx.load(args.src)
graph = model.graph
inits = {i.name: i for i in graph.initializer}

# Only MatMul B-side initializers: per-column (output channel) scales, axis=1.
targets: dict[str, list[onnx.NodeProto]] = {}
for node in graph.node:
    if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in inits:
        targets.setdefault(node.input[1], []).append(node)

report_rows = []
new_nodes = []
quant_bytes = 0
raw_bytes = 0
for name, init in list(inits.items()):
    arr = numpy_helper.to_array(init)
    raw_bytes += arr.nbytes
    consumers = targets.get(name)
    if not consumers or arr.dtype not in (np.float16, np.float32) or arr.ndim != 2 or arr.size <= args.min_elements:
        quant_bytes += arr.nbytes
        continue
    w = arr.astype(np.float32)
    amax = np.abs(w).max(axis=0)  # per output column of [K, N]
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(arr.dtype)
    q = np.clip(np.rint(w / scale.astype(np.float32)[None, :]), -127, 127).astype(np.int8)
    dequant = (q.astype(np.float32) * scale.astype(np.float32)[None, :]).astype(arr.dtype)
    err = dequant.astype(np.float32) - arr.astype(np.float32)
    denom = float(np.sqrt(np.mean(w * w))) or 1.0
    report_rows.append({
        "name": name, "shape": list(arr.shape),
        "maxAbsErr": float(np.abs(err).max()),
        "relRmsErr": float(np.sqrt(np.mean(err * err)) / denom),
    })
    q_init = numpy_helper.from_array(q, name=f"{name}_q8")
    s_init = numpy_helper.from_array(scale, name=f"{name}_q8_scale")
    graph.initializer.remove(init)
    graph.initializer.extend([q_init, s_init])
    dq_out = f"{name}_dequant"
    new_nodes.append(helper.make_node(
        "DequantizeLinear", [f"{name}_q8", f"{name}_q8_scale"], [dq_out],
        name=f"{name}_dql", axis=1,
    ))
    for node in consumers:
        node.input[1] = dq_out
    quant_bytes += q.nbytes + scale.nbytes

# DQL nodes must precede their consumers; prepending keeps topo order valid
# since their only inputs are initializers.
graph.node.extend([])
for node in reversed(new_nodes):
    graph.node.insert(0, node)

for opset in model.opset_import:
    if opset.domain in ("", "ai.onnx") and opset.version < 21:
        opset.version = 21
model.ir_version = max(model.ir_version, 10)

onnx.checker.check_model(model)
onnx.save(model, args.dst)
summary = {
    "schema": "lc0_browser.onnx_weights_qdq_int8.v1",
    "source": args.src,
    "quantizedTensors": len(report_rows),
    "f16Bytes": raw_bytes,
    "quantBytes": quant_bytes,
    "byteRatio": quant_bytes / raw_bytes if raw_bytes else None,
    "maxRelRmsErr": max((r["relRmsErr"] for r in report_rows), default=0.0),
    "outBytes": Path(args.dst).stat().st_size,
}
report = Path(args.report) if args.report else Path(args.dst).with_suffix(".qdq-report.json")
report.write_text(json.dumps({"summary": summary, "tensors": report_rows}, indent=1) + "\n")
print(json.dumps(summary, indent=1))
