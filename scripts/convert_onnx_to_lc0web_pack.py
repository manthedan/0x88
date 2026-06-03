#!/usr/bin/env python3
"""Convert a fixed-shape LC0 ONNX model into a browser-native lc0web pack.

The pack is intentionally simple for the first custom-runtime experiments:

  model.lc0web.json
  weights.000.bin
  weights.001.bin
  ...

All ONNX initializers are written as raw little-endian tensor bytes into bounded
shards. The JSON manifest records tensor names, shapes, dtypes, byte ranges, and
per-tensor/per-shard hashes so a browser worker can fetch and verify shards
without holding or hashing one giant ONNX file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections import Counter
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

LC0WEB_VERSION = 1

TENSOR_DTYPE_NAMES = {
    TensorProto.FLOAT: "f32",
    TensorProto.UINT8: "u8",
    TensorProto.INT8: "i8",
    TensorProto.UINT16: "u16",
    TensorProto.INT16: "i16",
    TensorProto.INT32: "i32",
    TensorProto.INT64: "i64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "f16",
    TensorProto.DOUBLE: "f64",
    TensorProto.UINT32: "u32",
    TensorProto.UINT64: "u64",
    TensorProto.BFLOAT16: "bf16",
}

NUMPY_DTYPES = {
    TensorProto.FLOAT: np.dtype("<f4"),
    TensorProto.UINT8: np.dtype("u1"),
    TensorProto.INT8: np.dtype("i1"),
    TensorProto.UINT16: np.dtype("<u2"),
    TensorProto.INT16: np.dtype("<i2"),
    TensorProto.INT32: np.dtype("<i4"),
    TensorProto.INT64: np.dtype("<i8"),
    TensorProto.BOOL: np.dtype("?"),
    TensorProto.FLOAT16: np.dtype("<f2"),
    TensorProto.DOUBLE: np.dtype("<f8"),
    TensorProto.UINT32: np.dtype("<u4"),
    TensorProto.UINT64: np.dtype("<u8"),
    TensorProto.BFLOAT16: np.dtype("<u2"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
      for chunk in iter(lambda: f.read(1024 * 1024), b""):
          h.update(chunk)
    return h.hexdigest()


def tensor_raw_bytes(tensor: TensorProto) -> bytes:
    if tensor.data_type == TensorProto.STRING:
        raise ValueError(f"String tensor initializers are not supported in lc0web packs: {tensor.name}")
    if tensor.raw_data:
        return bytes(tensor.raw_data)
    array = numpy_helper.to_array(tensor)
    dtype = NUMPY_DTYPES.get(tensor.data_type)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array).tobytes(order="C")


def tensor_dtype_name(data_type: int) -> str:
    return TENSOR_DTYPE_NAMES.get(data_type, TensorProto.DataType.Name(data_type).lower())


def value_info_to_json(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    dims: list[int | str | None] = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append(None)
    return {
        "name": value.name,
        "dtype": tensor_dtype_name(tensor_type.elem_type),
        "onnxDtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dims,
    }


def attr_value_to_json(attr: AttributeProto) -> Any:
    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytesHex": value.hex()}
    if isinstance(value, TensorProto):
        raw = tensor_raw_bytes(value)
        return {
            "tensor": True,
            "name": value.name,
            "dtype": tensor_dtype_name(value.data_type),
            "onnxDtype": TensorProto.DataType.Name(value.data_type),
            "shape": list(value.dims),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }
    if isinstance(value, onnx.GraphProto):
        return {"graph": True, "name": value.name, "nodes": len(value.node)}
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, bytes):
                out.append(item.decode("utf-8", errors="replace"))
            elif isinstance(item, TensorProto):
                raw = tensor_raw_bytes(item)
                out.append({
                    "tensor": True,
                    "name": item.name,
                    "dtype": tensor_dtype_name(item.data_type),
                    "onnxDtype": TensorProto.DataType.Name(item.data_type),
                    "shape": list(item.dims),
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                })
            else:
                out.append(item)
        return out
    return value


def node_to_json(node: onnx.NodeProto) -> dict[str, Any]:
    attrs = {attr.name: attr_value_to_json(attr) for attr in node.attribute}
    return {
        "name": node.name,
        "opType": node.op_type,
        "domain": node.domain,
        "inputs": list(node.input),
        "outputs": list(node.output),
        **({"attributes": attrs} if attrs else {}),
    }


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return int(math.ceil(value / alignment) * alignment)


class ShardWriter:
    def __init__(self, out_dir: pathlib.Path, shard_bytes: int, alignment: int):
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self.alignment = alignment
        self.index = -1
        self.file = None
        self.path: pathlib.Path | None = None
        self.offset = 0
        self.shards: list[dict[str, Any]] = []
        self._open_next()

    def _open_next(self) -> None:
        if self.file is not None:
            self.file.close()
            assert self.path is not None
            self.shards.append({
                "file": self.path.name,
                "bytes": self.path.stat().st_size,
                "sha256": sha256_file(self.path),
            })
        self.index += 1
        self.path = self.out_dir / f"weights.{self.index:03d}.bin"
        self.file = self.path.open("wb")
        self.offset = 0

    def write_tensor(self, data: bytes) -> tuple[str, int]:
        assert self.file is not None and self.path is not None
        aligned = align_up(self.offset, self.alignment)
        padding = aligned - self.offset
        # If the tensor would exceed the target shard size and the current shard
        # is non-empty, start a fresh shard. Large tensors may occupy a shard by
        # themselves and exceed the advisory size.
        if aligned > 0 and aligned + len(data) > self.shard_bytes:
            self._open_next()
            aligned = 0
            padding = 0
        if padding:
            self.file.write(b"\x00" * padding)
            self.offset += padding
        tensor_offset = self.offset
        self.file.write(data)
        self.offset += len(data)
        return self.path.name, tensor_offset

    def close(self) -> list[dict[str, Any]]:
        if self.file is not None:
            self.file.close()
            assert self.path is not None
            # Keep a zero-tensor model from producing an empty shard only if it
            # already has useful content; LC0 models always have initializers.
            self.shards.append({
                "file": self.path.name,
                "bytes": self.path.stat().st_size,
                "sha256": sha256_file(self.path),
            })
            self.file = None
        return self.shards


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source = pathlib.Path(args.onnx).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source), load_external_data=True)
    graph = model.graph
    source_sha = sha256_file(source)
    with tempfile.TemporaryDirectory(prefix="lc0web-pack-", dir=str(out_dir.parent)) as tmp_name:
        tmp_dir = pathlib.Path(tmp_name)
        shard_writer = ShardWriter(tmp_dir, args.shard_bytes, args.align_bytes)
        tensors: list[dict[str, Any]] = []
        total_tensor_bytes = 0
        dtype_counter: Counter[str] = Counter()
        for tensor in graph.initializer:
            raw = tensor_raw_bytes(tensor)
            shard_file, offset = shard_writer.write_tensor(raw)
            dtype = tensor_dtype_name(tensor.data_type)
            dtype_counter[dtype] += 1
            total_tensor_bytes += len(raw)
            tensors.append({
                "name": tensor.name,
                "dtype": dtype,
                "onnxDtype": TensorProto.DataType.Name(tensor.data_type),
                "shape": list(tensor.dims),
                "shard": shard_file,
                "byteOffset": offset,
                "byteLength": len(raw),
                "sha256": sha256_bytes(raw),
            })
        shards = shard_writer.close()

        manifest = {
            "format": "lc0web",
            "version": LC0WEB_VERSION,
            "model": {
                "name": args.name or source.stem,
                "family": "lc0",
                "sourceFormat": "onnx",
                "sourceFile": source.name,
                "sourceSha256": source_sha,
                "architecture": args.architecture,
                "recommendedRuntime": args.recommended_runtime,
                "layout": args.layout,
            },
            "graph": {
                "name": graph.name,
                "opsets": [{"domain": opset.domain, "version": opset.version} for opset in model.opset_import],
                "inputs": [value_info_to_json(value) for value in graph.input],
                "outputs": [value_info_to_json(value) for value in graph.output],
                "nodes": [node_to_json(node) for node in graph.node],
                "opHistogram": dict(sorted(Counter(node.op_type for node in graph.node).items())),
            },
            "weights": {
                "shardBytesTarget": args.shard_bytes,
                "alignmentBytes": args.align_bytes,
                "totalTensorBytes": total_tensor_bytes,
                "tensorCount": len(tensors),
                "dtypeHistogram": dict(sorted(dtype_counter.items())),
                "shards": shards,
                "tensors": tensors,
            },
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        (tmp_dir / "model.lc0web.json").write_bytes(manifest_bytes)
        manifest["packSha256"] = sha256_bytes(manifest_bytes)
        # Rewrite including packSha256 for readers. This hash is intentionally a
        # content identifier for the pre-packSha JSON, avoiding self-reference.
        (tmp_dir / "model.lc0web.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        for child in out_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in tmp_dir.iterdir():
            child.replace(out_dir / child.name)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX LC0 model to lc0web shard pack")
    parser.add_argument("onnx", help="input ONNX model")
    parser.add_argument("out", help="output pack directory")
    parser.add_argument("--name", default=None, help="model name stored in manifest")
    parser.add_argument("--architecture", default="transformer", help="LC0 architecture label")
    parser.add_argument("--recommended-runtime", default="custom-webgpu", choices=["ort-webgpu", "custom-webgpu", "hybrid"], help="runtime hint")
    parser.add_argument("--layout", default="raw-f16", help="tensor layout label")
    parser.add_argument("--shard-bytes", type=int, default=16 * 1024 * 1024, help="target max bytes per shard")
    parser.add_argument("--align-bytes", type=int, default=64, help="tensor offset alignment within shards")
    return parser.parse_args()


if __name__ == "__main__":
    manifest = convert(parse_args())
    print(json.dumps({
        "format": manifest["format"],
        "name": manifest["model"]["name"],
        "sourceSha256": manifest["model"]["sourceSha256"],
        "tensorCount": manifest["weights"]["tensorCount"],
        "totalTensorBytes": manifest["weights"]["totalTensorBytes"],
        "shards": manifest["weights"]["shards"],
    }, indent=2))
