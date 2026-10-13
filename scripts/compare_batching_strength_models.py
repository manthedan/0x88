#!/usr/bin/env python3
"""Dependency-free ONNX protobuf byte comparison; does not execute a model.

Usage: python3 scripts/compare_batching_strength_models.py [batch1.onnx batch8.onnx]
Checks serialized nodes and ALL initializers, including quantization parameters.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path


def varint(data, offset):
    value, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7


def fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        yield number, value


def inspect(file):
    data = Path(file).read_bytes()
    graph = next(value for number, value in fields(data) if number == 7)
    initializers, nodes = {}, []
    for number, value in fields(graph):
        if number == 5:
            name = next(v.decode() for n, v in fields(value) if n == 8)
            initializers[name] = value
        elif number == 1:
            nodes.append(value)
    return {'file': file, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}, initializers, nodes


base = 'public/models/lc0/t1-256x10-distilled-swa-2432500.'
files = sys.argv[1:] or [base + 'batch1.f16.qdq8.onnx', base + 'batch8.f16.qdq8.onnx']
if len(files) != 2:
    raise SystemExit('Expected two ONNX paths')
left, li, ln = inspect(files[0])
right, ri, rn = inspect(files[1])
different = sorted(name for name in set(li) | set(ri) if li.get(name) != ri.get(name))
shapes = {}
for name in different:
    if name == '/const/pos_encoding_shape':
        shapes[name] = [list(struct.unpack('<3q', next(v for n, v in fields(i[name]) if n == 9))) for i in (li, ri)]
print(json.dumps({'files': [left, right], 'initializers': len(li), 'sameInitializerNames': set(li) == set(ri),
                  'differentInitializers': different, 'shapeValues': shapes,
                  'nodes': len(ln), 'identicalSerializedNodes': ln == rn}, indent=2))
