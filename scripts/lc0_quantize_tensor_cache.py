#!/usr/bin/env python3
"""Quantize a detached f16 TVMJS tensor-cache to int8 weight storage.

Reads the tensor-cache directory dumped by the whole-ONNX probe with
--detach-params (tensor-cache.json + params_shard_*.bin, raw f16 records) and
writes a new directory in the same file layout where large 2D+ tensors are
stored as symmetric per-output-channel int8 (q = round(w / scale), scale =
amax(|w|, axis!=0) / 127, f32 scales appended after the int8 data in the same
shard). Small or 1-D tensors (biases, norms, embeddings under the threshold)
stay raw f16 — they are cheap and numerically sensitive.

This changes ONLY weight storage/transfer: the browser dequantizes back to
f16 before set_input, so GPU compute is unchanged. It is goal 1 (download
footprint) of the BT4 quantization exploration, not quantized kernels.

Record schema additions: mode ('int8-ch0' | 'raw'), scaleByteOffset,
scaleNbytes, channels. dtype stays the logical 'float16' so readers know the
dequantized type. quantization metadata + per-tensor error stats land in the
output tensor-cache.json and the --report sidecar.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--in", dest="src", required=True, help="detached f16 tensor-cache directory")
parser.add_argument("--out", dest="dst", required=True, help="output quantized tensor-cache directory")
parser.add_argument("--min-elements", type=int, default=4096, help="quantize only tensors larger than this")
parser.add_argument("--min-ndim", type=int, default=2, help="quantize only tensors with at least this many dims")
parser.add_argument("--shard-bytes", type=int, default=32 * 1024 * 1024, help="target shard size")
parser.add_argument("--report", default=None, help="quality report JSON path (default <out>/quantization-report.json)")
parser.add_argument("--max-rel-rms", type=float, default=None,
                    help="mixed precision: revert tensors whose int8 relative RMS error exceeds this back to raw f16")
parser.add_argument("--bits", type=int, default=8, choices=(4, 8),
                    help="8: per-output-channel int8 (default). 4: group-wise int4, two values per byte, offset-binary nibbles")
parser.add_argument("--group-size", type=int, default=64,
                    help="int4 group size along each row (scales are per channel x group)")
args = parser.parse_args()

src = Path(args.src)
dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)
cache = json.loads((src / "tensor-cache.json").read_text())

records_by_name = {}
for shard in cache["records"]:
    data = (src / shard["dataPath"]).read_bytes()
    for record in shard["records"]:
        if record.get("format") != "raw":
            raise SystemExit(f"unsupported record format {record.get('format')} for {record['name']}")
        if record["dtype"] != "float16":
            raise SystemExit(f"unsupported dtype {record['dtype']} for {record['name']}")
        start = record["byteOffset"]
        records_by_name[record["name"]] = (record, data[start:start + record["nbytes"]])

# Preserve param_N order — the browser feeds them positionally into set_input.
def param_index(name: str) -> int:
    return int(name.rsplit("_", 1)[1])

ordered = sorted(records_by_name.items(), key=lambda kv: param_index(kv[0]))

out_shards = []
current = bytearray()
current_records = []
report_rows = []
raw_bytes_total = 0
quant_bytes_total = 0

def flush_shard():
    global current, current_records
    if not current_records:
        return
    name = f"params_shard_{len(out_shards)}.bin"
    (dst / name).write_bytes(bytes(current))
    out_shards.append({
        "dataPath": name,
        "format": "raw-shard",
        "nbytes": len(current),
        "records": current_records,
        "md5sum": hashlib.md5(bytes(current)).hexdigest(),
    })
    current = bytearray()
    current_records = []

for name, (record, raw) in ordered:
    shape = record["shape"]
    elements = int(np.prod(shape))
    raw_bytes_total += len(raw)
    quantize = len(shape) >= args.min_ndim and elements > args.min_elements
    if quantize:
        w = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        flat = w.reshape(shape[0], -1)
        channels, cols = flat.shape
        if args.bits == 8:
            mode = "int8-ch0"
            amax = np.abs(flat).max(axis=1)
            scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
            dequant = (q.astype(np.float32) * scale[:, None]).astype(np.float16).astype(np.float32)
            qdata = q.tobytes()
        else:
            group = args.group_size
            mode = f"int4-g{group}"
            ngroups = -(-cols // group)
            padded = np.zeros((channels, ngroups * group), dtype=np.float32)
            padded[:, :cols] = flat
            grouped = padded.reshape(channels, ngroups, group)
            amax = np.abs(grouped).max(axis=2)
            scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
            q = np.clip(np.rint(grouped / scale[:, :, None]), -7, 7).astype(np.int8)
            dequant = (q.astype(np.float32) * scale[:, :, None]).reshape(channels, -1)[:, :cols].astype(np.float16).astype(np.float32)
            # Offset-binary nibbles packed two per byte, element-major over the
            # unpadded row-major order (decoder skips pad columns).
            nibbles = (q.reshape(channels, -1)[:, :cols].reshape(-1) + 8).astype(np.uint8)
            if nibbles.size % 2:
                nibbles = np.concatenate([nibbles, np.zeros(1, dtype=np.uint8)])
            qdata = (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes()
        err = dequant - flat.astype(np.float16).astype(np.float32)
        denom = float(np.sqrt(np.mean(flat * flat))) or 1.0
        rel_rms = float(np.sqrt(np.mean(err * err)) / denom)
        if args.max_rel_rms is not None and rel_rms > args.max_rel_rms:
            quantize = False
            report_rows.append({"name": name, "shape": shape, "mode": "raw",
                                "revertedRelRmsErr": rel_rms})
    if quantize:
        report_rows.append({
            "name": name, "shape": shape, "mode": mode,
            "maxAbsErr": float(np.abs(err).max()),
            "relRmsErr": rel_rms,
        })
        sdata = scale.tobytes()
        entry = {
            "name": name, "shape": shape, "dtype": "float16", "format": "raw",
            "mode": mode, "channels": channels, "columns": cols,
            "byteOffset": len(current), "nbytes": len(qdata),
            "scaleByteOffset": len(current) + len(qdata), "scaleNbytes": len(sdata),
        }
        if args.bits == 4:
            entry["groupSize"] = args.group_size
        current.extend(qdata)
        current.extend(sdata)
        quant_bytes_total += len(qdata) + len(sdata)
    else:
        if not (report_rows and report_rows[-1]["name"] == name):
            report_rows.append({"name": name, "shape": shape, "mode": "raw"})
        entry = {
            "name": name, "shape": shape, "dtype": "float16", "format": "raw",
            "mode": "raw", "byteOffset": len(current), "nbytes": len(raw),
        }
        current.extend(raw)
        quant_bytes_total += len(raw)
    current_records.append(entry)
    if len(current) >= args.shard_bytes:
        flush_shard()
flush_shard()

quantized_rows = [r for r in report_rows if r["mode"].startswith("int")]
summary = {
    "schema": "lc0_browser.tvmjs_tensor_cache_int8.v1",
    "source": str(src),
    "params": len(ordered),
    "quantizedTensors": len(quantized_rows),
    "rawTensors": len(report_rows) - len(quantized_rows),
    "f16Bytes": raw_bytes_total,
    "quantBytes": quant_bytes_total,
    "byteRatio": quant_bytes_total / raw_bytes_total if raw_bytes_total else None,
    "maxAbsErr": max((r["maxAbsErr"] for r in quantized_rows), default=0.0),
    "maxRelRmsErr": max((r["relRmsErr"] for r in quantized_rows), default=0.0),
}

out_cache = {
    "metadata": dict(cache.get("metadata") or {}, parameterStorage=("int8-ch0+raw-f16" if args.bits == 8 else f"int4-g{args.group_size}+raw-f16")),
    "quantization": summary,
    "records": out_shards,
}
(dst / "tensor-cache.json").write_text(json.dumps(out_cache) + "\n")
report_path = Path(args.report) if args.report else dst / "quantization-report.json"
report_path.write_text(json.dumps({"summary": summary, "tensors": report_rows}, indent=1) + "\n")
print(json.dumps(summary, indent=1))
