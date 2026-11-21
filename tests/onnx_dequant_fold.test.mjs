import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import {
  countOnnxDequantizeLinear,
  encodeTensorProto,
  f16BitsToF32,
  f32ToF16Bits,
  foldOnnxDequantizeLinear,
  ONNX_FLOAT,
  ONNX_FLOAT16,
  ONNX_INT4,
  ONNX_INT8,
  ONNX_UINT4,
  ONNX_UINT8,
  ProtoWriter,
  parseProtoFields,
} from '../src/nn/onnxDequantFold.ts';
import * as ort from '../src/nn/ortRuntime.ts';
import { skipUnlessAssets } from './lib/required_assets.mjs';

// ---- tiny ONNX builder (ModelProto/GraphProto/NodeProto/ValueInfoProto field numbers)
function valueInfo(name, elemType, dims) {
  const shape = new ProtoWriter();
  for (const d of dims) shape.bytesField(1, new ProtoWriter().varintField(1, d).finish());
  const tensorType = new ProtoWriter().varintField(1, elemType).bytesField(2, shape.finish()).finish();
  return new ProtoWriter().stringField(1, name).bytesField(2, new ProtoWriter().bytesField(1, tensorType).finish()).finish();
}
function attrInt(name, value) {
  return new ProtoWriter().stringField(1, name).varintField(3, value).varintField(20, 2).finish();
}
function node(opType, inputs, outputs, attrs = []) {
  const w = new ProtoWriter();
  for (const i of inputs) w.stringField(1, i);
  for (const o of outputs) w.stringField(2, o);
  w.stringField(4, opType);
  for (const a of attrs) w.bytesField(5, a);
  return w.finish();
}
function model({ nodes, initializers, inputs, outputs }) {
  const g = new ProtoWriter();
  for (const n of nodes) g.bytesField(1, n);
  g.stringField(2, 'fold-test');
  for (const t of initializers) g.bytesField(5, t);
  for (const i of inputs) g.bytesField(11, i);
  for (const o of outputs) g.bytesField(12, o);
  const opset = new ProtoWriter().stringField(1, '').varintField(2, 21).finish();
  return new ProtoWriter().varintField(1, 10).stringField(2, 'fold-test').bytesField(7, g.finish()).bytesField(8, opset).finish();
}
const f32Bytes = (values) => new Uint8Array(Float32Array.from(values).buffer);
const f16Bytes = (values) => new Uint8Array(Uint16Array.from(values.map(f32ToF16Bits)).buffer);
const int8Bytes = (values) => new Uint8Array(Int8Array.from(values).buffer);
function int4Bytes(values) {
  const out = new Uint8Array(Math.ceil(values.length / 2));
  values.forEach((v, i) => {
    out[i >> 1] |= (v & 0xf) << ((i & 1) * 4);
  });
  return out;
}
function matmul(x, rows, cols, w, n) {
  const y = new Float32Array(rows * n);
  for (let r = 0; r < rows; r++)
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let k = 0; k < cols; k++) acc += x[r * cols + k] * w[k * n + j];
      y[r * n + j] = acc;
    }
  return y;
}
const wasmSession = (bytes) => ort.InferenceSession.create(bytes, ort.sessionOptions(['wasm']));

test('corrupt protobuf lengths reject without looping', () => {
  // -11 rewinds this 11-byte field to its tag. Isolate it so a regression
  // cannot hang the test runner's event loop and defeat its timeout.
  const result = spawnSync(
    process.execPath,
    [
      '--experimental-strip-types',
      '--input-type=module',
      '-e',
      `import assert from 'node:assert/strict';
       import { parseProtoFields } from ${JSON.stringify(new URL('../src/nn/onnxDequantFold.ts', import.meta.url).href)};
       assert.throws(() => parseProtoFields(Uint8Array.of(10, 245, 255, 255, 255, 255, 255, 255, 255, 255, 1)), /protobuf.*length/);`,
    ],
    { encoding: 'utf8', timeout: 2000 },
  );
  assert.equal(result.status, 0, `${result.error ?? ''}\n${result.stderr}`);
  assert.throws(() => parseProtoFields(Uint8Array.of(10, 4, 1)), /protobuf.*length/);
  assert.throws(() => parseProtoFields(Uint8Array.of(10, 1, 0), 0, 20), /protobuf.*length|truncated/);
  assert.throws(() => parseProtoFields(Uint8Array.of(10, 3, 0), 0, 20), /protobuf.*length/);
});

test('ORT diagnostics report the original model after rejecting its folded rewrite', async (t) => {
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'])],
    initializers: [encodeTensorProto('Wq', ONNX_INT8, [1], int8Bytes([1])), encodeTensorProto('Ws', ONNX_FLOAT, [], f32Bytes([0.5]))],
    inputs: [],
    outputs: [valueInfo('W', ONNX_FLOAT, [1])],
  });
  const attempts = [];
  const session = { release() {} };
  t.mock.method(ort.InferenceSession, 'create', async (input) => {
    attempts.push(input);
    if (attempts.length === 1) throw new Error('folded model rejected');
    return session;
  });
  ort.setOrtDequantFoldForCurrentThread(true);
  try {
    assert.equal(await ort.createOrtSession(bytes), session);
    assert.equal(attempts.length, 2);
    assert.notEqual(attempts[0], bytes);
    assert.equal(attempts[1], bytes);
    const summary = ort.lastOrtDequantFoldSummary();
    assert.equal(summary.foldedNodes, 0);
    assert.equal(summary.removedInitializers, 0);
    assert.equal(summary.bytesAfter, bytes.byteLength);
    assert.equal(summary.discardedAfter, 'folded model rejected');
    assert.equal((await ort.collectOrtRuntimeDiagnostics()).dequantFold.discardedAfter, 'folded model rejected');
    await ort.releaseOrtSession(session);
  } finally {
    ort.setOrtDequantFoldForCurrentThread(null);
  }
});

test('f16 conversion round-trips and rounds to nearest even', () => {
  for (const v of [0, 1, -1, 0.5, 65504, 6.1035e-5, 3.0517578e-5, 1e-8, -2.5, 1024.5]) {
    const back = f16BitsToF32(f32ToF16Bits(v));
    assert.ok(Math.abs(back - v) <= Math.max(Math.abs(v) * 2 ** -11, 2 ** -25), `${v} -> ${back}`);
  }
  assert.equal(f32ToF16Bits(1 + 2 ** -11), 0x3c00, 'tie rounds to even (down)');
  assert.equal(f32ToF16Bits(1 + 3 * 2 ** -11), 0x3c02, 'tie rounds to even (up)');
  assert.equal(f32ToF16Bits(1e6), 0x7c00, 'overflow -> inf');
  assert.equal(f16BitsToF32(0xfbff), -65504);
});

test('per-column int8 DequantizeLinear folds into an f32 initializer with identical outputs', async () => {
  const q = [1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12]; // [4, 3]
  const scale = [0.5, 0.25, 0.125];
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrInt('axis', 1)]), node('MatMul', ['X', 'W'], ['Y'])],
    initializers: [encodeTensorProto('Wq', ONNX_INT8, [4, 3], int8Bytes(q)), encodeTensorProto('Ws', ONNX_FLOAT, [3], f32Bytes(scale))],
    inputs: [valueInfo('X', ONNX_FLOAT, [2, 4])],
    outputs: [valueInfo('Y', ONNX_FLOAT, [2, 3])],
  });
  assert.equal(countOnnxDequantizeLinear(bytes), 1);
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.equal(folded.foldedNodes, 1);
  assert.equal(folded.removedInitializers, 2);
  assert.equal(countOnnxDequantizeLinear(folded.bytes), 0);
  assert.equal(parseProtoFields(folded.bytes).length, parseProtoFields(bytes).length, 'top-level fields preserved');
  const x = Float32Array.from([1, 2, 3, 4, -1, 0.5, 2, -3]);
  const expected = matmul(
    x,
    2,
    4,
    q.map((v, i) => v * scale[i % 3]),
    3,
  );
  for (const [label, session] of [
    ['original', await wasmSession(bytes)],
    ['folded', await wasmSession(folded.bytes)],
  ]) {
    const out = await session.run({ X: new ort.Tensor('float32', x, [2, 4]) });
    assert.deepEqual(Array.from(out.Y.data), Array.from(expected), label);
    await session.release();
  }
});

test('blocked int4 DequantizeLinear (axis 0, block 2) folds to the reference dequantization', async () => {
  const q = [7, -7, 3, 0, 1, -1, 2, 4, -8, 5, -3, 6]; // [4, 3] rows blocked in pairs
  const scale = [0.5, 1, 2, 0.125, 0.25, 0.75]; // [2, 3]
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrInt('axis', 0), attrInt('block_size', 2)]), node('MatMul', ['X', 'W'], ['Y'])],
    initializers: [encodeTensorProto('Wq', ONNX_INT4, [4, 3], int4Bytes(q)), encodeTensorProto('Ws', ONNX_FLOAT, [2, 3], f32Bytes(scale))],
    inputs: [valueInfo('X', ONNX_FLOAT, [1, 4])],
    outputs: [valueInfo('Y', ONNX_FLOAT, [1, 3])],
  });
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.equal(folded.foldedNodes, 1);
  const w = q.map((v, i) => v * scale[Math.floor(Math.floor(i / 3) / 2) * 3 + (i % 3)]);
  const x = Float32Array.from([1, -2, 0.5, 4]);
  const expected = matmul(x, 1, 4, w, 3);
  const session = await wasmSession(folded.bytes);
  const out = await session.run({ X: new ort.Tensor('float32', x, [1, 4]) });
  assert.deepEqual(Array.from(out.Y.data), Array.from(expected));
  await session.release();
  try {
    const original = await wasmSession(bytes);
    const ref = await original.run({ X: new ort.Tensor('float32', x, [1, 4]) });
    assert.deepEqual(Array.from(ref.Y.data), Array.from(expected), 'ORT agrees with the reference on the unfolded graph');
    await original.release();
  } catch (err) {
    console.log(`(ORT wasm cannot run blocked int4 DequantizeLinear here: ${err.message.split('\n')[0]})`);
  }
});

test('f16 scales produce an f16 initializer that ORT reads back correctly', async () => {
  const q = [100, -50, 25, 3, -7, 127, 0, 1, 64, -128, 12, -12];
  const scale = [0.01, 0.002, 0.03];
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrInt('axis', 1)]), node('MatMul', ['X', 'W'], ['Y'])],
    initializers: [encodeTensorProto('Wq', ONNX_INT8, [4, 3], int8Bytes(q)), encodeTensorProto('Ws', ONNX_FLOAT16, [3], f16Bytes(scale))],
    inputs: [valueInfo('X', ONNX_FLOAT16, [1, 4])],
    outputs: [valueInfo('Y', ONNX_FLOAT16, [1, 3])],
  });
  ort.setOrtDequantFoldForCurrentThread(true);
  const session = await ort.createOrtSession(bytes);
  ort.setOrtDequantFoldForCurrentThread(null);
  const summary = ort.lastOrtDequantFoldSummary();
  assert.equal(summary?.foldedNodes, 1, 'createOrtSession folds by default');
  const x = [1, -1, 0.5, 2];
  const out = await session.run({ X: new ort.Tensor('float16', Uint16Array.from(x.map(f32ToF16Bits)), [1, 4]) });
  const got = out.Y.data instanceof Uint16Array ? Array.from(out.Y.data).map(f16BitsToF32) : Array.from(out.Y.data); // ORT hands back Float16Array where the runtime has one
  const w = q.map((v, i) => f16BitsToF32(f32ToF16Bits(v * f16BitsToF32(f32ToF16Bits(scale[i % 3])))));
  const expected = matmul(Float32Array.from(x), 1, 4, w, 3);
  for (const [i, v] of got.entries()) assert.ok(Math.abs(v - expected[i]) <= Math.abs(expected[i]) * 4e-3 + 1e-3, `Y[${i}] ${v} vs ${expected[i]}`);
  await ort.releaseOrtSession(session);
});

// Two's-complement int64 varint (10 bytes for negatives), which ProtoWriter.varint refuses.
function signedVarint(value) {
  let v = BigInt.asUintN(64, BigInt(value));
  const out = [];
  while (v >= 128n) {
    out.push(Number(v % 128n) | 0x80);
    v /= 128n;
  }
  out.push(Number(v));
  return Uint8Array.from(out);
}
function attrSignedInt(name, value) {
  return new ProtoWriter().stringField(1, name).tag(3, 0).raw(signedVarint(value)).varintField(20, 2).finish();
}
function attrGraph(name, graphBytes) {
  return new ProtoWriter().stringField(1, name).bytesField(6, graphBytes).varintField(20, 5).finish();
}
function attrGraphs(name, graphs) {
  const w = new ProtoWriter().stringField(1, name);
  for (const g of graphs) w.bytesField(11, g);
  return w.varintField(20, 10).finish();
}
function int8TensorAsInt32Data(name, dims, values) {
  const packed = new ProtoWriter();
  for (const v of values) packed.raw(signedVarint(v));
  const w = new ProtoWriter();
  for (const d of dims) w.varintField(1, d);
  w.varintField(2, ONNX_INT8).bytesField(5, packed.finish()).stringField(8, name);
  return w.finish();
}

test('negative axis and negative int32_data values decode as signed integers', async () => {
  const q = [1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12]; // [4, 3]
  const scale = [0.5, 0.25, 0.125];
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrSignedInt('axis', -1)]), node('MatMul', ['X', 'W'], ['Y'])],
    initializers: [int8TensorAsInt32Data('Wq', [4, 3], q), encodeTensorProto('Ws', ONNX_FLOAT, [3], f32Bytes(scale))],
    inputs: [valueInfo('X', ONNX_FLOAT, [2, 4])],
    outputs: [valueInfo('Y', ONNX_FLOAT, [2, 3])],
  });
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.equal(folded.foldedNodes, 1);
  const x = Float32Array.from([1, 2, 3, 4, -1, 0.5, 2, -3]);
  const expected = matmul(
    x,
    2,
    4,
    q.map((v, i) => v * scale[i % 3]),
    3,
  );
  for (const [label, session] of [
    ['original', await wasmSession(bytes)],
    ['folded', await wasmSession(folded.bytes)],
  ]) {
    const out = await session.run({ X: new ort.Tensor('float32', x, [2, 4]) });
    assert.deepEqual(Array.from(out.Y.data), Array.from(expected), label);
    await session.release();
  }
});

test('initializers captured by a subgraph survive the fold', () => {
  const thenBranch = new ProtoWriter()
    .bytesField(1, node('Identity', ['Ws'], ['Z']))
    .stringField(2, 'then')
    .bytesField(12, valueInfo('Z', ONNX_FLOAT, [3]))
    .finish();
  const elseBranch = new ProtoWriter()
    .bytesField(1, node('Identity', ['Zero'], ['Z']))
    .stringField(2, 'else')
    .bytesField(12, valueInfo('Z', ONNX_FLOAT, [3]))
    .finish();
  const bytes = model({
    nodes: [
      node('DequantizeLinear', ['Wq', 'Ws', 'Zp'], ['W'], [attrInt('axis', 1)]),
      node('MatMul', ['X', 'W'], ['Y']),
      node('If', ['cond'], ['Z'], [attrGraph('then_branch', thenBranch), attrGraph('else_branch', elseBranch)]),
      // A repeated-graph (GRAPHS, field 11) attribute capturing 'Zp' the same way.
      node(
        'CustomBodies',
        [],
        ['Z2'],
        [
          attrGraphs('bodies', [
            new ProtoWriter()
              .bytesField(1, node('Identity', ['Zp'], ['Z2']))
              .stringField(2, 'body')
              .finish(),
          ]),
        ],
      ),
    ],
    initializers: [
      encodeTensorProto('Wq', ONNX_INT8, [4, 3], int8Bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
      encodeTensorProto('Ws', ONNX_FLOAT, [3], f32Bytes([0.5, 0.25, 0.125])),
      encodeTensorProto('Zp', ONNX_INT8, [3], int8Bytes([0, 0, 0])),
      encodeTensorProto('Zero', ONNX_FLOAT, [3], f32Bytes([0, 0, 0])),
    ],
    inputs: [valueInfo('X', ONNX_FLOAT, [2, 4]), valueInfo('cond', 9, [])],
    outputs: [valueInfo('Y', ONNX_FLOAT, [2, 3]), valueInfo('Z', ONNX_FLOAT, [3])],
  });
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.equal(folded.foldedNodes, 1);
  assert.equal(folded.removedInitializers, 1, 'only Wq is dropped; Ws and Zp are read by subgraph bodies');
  const graph = parseProtoFields(folded.bytes).find((f) => f.field === 7);
  const names = parseProtoFields(folded.bytes, graph.valueStart, graph.valueEnd)
    .filter((f) => f.field === 5)
    .map((f) => parseProtoFields(folded.bytes, f.valueStart, f.valueEnd).find((t) => t.field === 8))
    .map((t) => new TextDecoder().decode(folded.bytes.subarray(t.valueStart, t.valueEnd)));
  assert.deepEqual(names.sort(), ['W', 'Ws', 'Zero', 'Zp']);
});

test('output_dtype selects the folded tensor type; unsupported output types are skipped', () => {
  const build = (outputDtype) =>
    model({
      nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrInt('axis', 1), attrInt('output_dtype', outputDtype)]), node('MatMul', ['X', 'W'], ['Y'])],
      initializers: [encodeTensorProto('Wq', ONNX_INT8, [2, 2], int8Bytes([1, -2, 3, 4])), encodeTensorProto('Ws', ONNX_FLOAT16, [2], f16Bytes([0.5, 0.25]))],
      inputs: [valueInfo('X', ONNX_FLOAT, [1, 2])],
      outputs: [valueInfo('Y', ONNX_FLOAT, [1, 2])],
    });
  const foldedType = (bytes) => {
    const graph = parseProtoFields(bytes).find((f) => f.field === 7);
    for (const f of parseProtoFields(bytes, graph.valueStart, graph.valueEnd).filter((t) => t.field === 5)) {
      const fields = parseProtoFields(bytes, f.valueStart, f.valueEnd);
      const name = fields.find((t) => t.field === 8);
      if (new TextDecoder().decode(bytes.subarray(name.valueStart, name.valueEnd)) === 'W') return fields.find((t) => t.field === 2).varint;
    }
    return null;
  };
  const asFloat = foldOnnxDequantizeLinear(build(ONNX_FLOAT));
  assert.equal(asFloat.foldedNodes, 1);
  assert.equal(foldedType(asFloat.bytes), ONNX_FLOAT, 'output_dtype=FLOAT overrides the f16 scale type');
  const asBf16 = foldOnnxDequantizeLinear(build(16 /* BFLOAT16 */));
  assert.equal(asBf16.foldedNodes, 0);
  assert.equal(asBf16.skippedNodes, 1);
});

test('a DequantizeLinear fed by an overridable graph input is not folded', () => {
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrInt('axis', 1)]), node('MatMul', ['X', 'W'], ['Y'])],
    initializers: [
      encodeTensorProto('Wq', ONNX_INT8, [4, 3], int8Bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
      encodeTensorProto('Ws', ONNX_FLOAT, [3], f32Bytes([0.5, 0.25, 0.125])),
    ],
    inputs: [valueInfo('X', ONNX_FLOAT, [2, 4]), valueInfo('Ws', ONNX_FLOAT, [3])],
    outputs: [valueInfo('Y', ONNX_FLOAT, [2, 3])],
  });
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.equal(folded.foldedNodes, 0);
  assert.equal(folded.skippedNodes, 1);
  assert.equal(folded.bytes, bytes, 'model bytes returned unchanged');
});

function foldedRaw(bytes, name = 'W') {
  const graph = parseProtoFields(bytes).find((f) => f.field === 7);
  for (const field of parseProtoFields(bytes, graph.valueStart, graph.valueEnd).filter((f) => f.field === 5)) {
    const fields = parseProtoFields(bytes, field.valueStart, field.valueEnd);
    const named = fields.find((f) => f.field === 8);
    if (new TextDecoder().decode(bytes.subarray(named.valueStart, named.valueEnd)) === name) {
      const raw = fields.find((f) => f.field === 9);
      return bytes.subarray(raw.valueStart, raw.valueEnd);
    }
  }
  assert.fail(`missing ${name}`);
}

test('raw int4 folding allocates just the exact final buffer and never constructs a Float32Array intermediate', () => {
  const count = 100001;
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'])],
    initializers: [
      encodeTensorProto('Wq', ONNX_INT4, [count], new Uint8Array(Math.ceil(count / 2)).fill(0x7f)),
      encodeTensorProto('Ws', ONNX_FLOAT16, [], f16Bytes([0.125])),
    ],
    inputs: [],
    outputs: [valueInfo('W', ONNX_FLOAT16, [count])],
  });
  const NativeBytes = globalThis.Uint8Array,
    NativeFloats = globalThis.Float32Array;
  const allocations = [];
  let folded;
  globalThis.Uint8Array = class extends NativeBytes {
    constructor(...args) {
      super(...args);
      if (typeof args[0] === 'number' && args[0] >= 4096) allocations.push(args[0]);
    }
  };
  globalThis.Float32Array = class extends NativeFloats {
    constructor() {
      throw new Error('forbidden Float32Array intermediate');
    }
  };
  try {
    folded = foldOnnxDequantizeLinear(bytes);
  } finally {
    globalThis.Uint8Array = NativeBytes;
    globalThis.Float32Array = NativeFloats;
  }
  assert.equal(folded.foldedNodes, 1);
  assert.deepEqual(allocations, [folded.bytes.byteLength]);
  assert.equal(foldedRaw(folded.bytes).byteLength, count * 2);
});

test('all integer dtypes/layouts/output dtypes match the old f32-intermediate rounding, including the half-reader fallback', () => {
  const descriptor = Object.getOwnPropertyDescriptor(DataView.prototype, 'getFloat16');
  const encoder = Object.getOwnPropertyDescriptor(DataView.prototype, 'setFloat16');
  for (const fallback of [false, true]) {
    if (fallback && descriptor) Object.defineProperty(DataView.prototype, 'getFloat16', { ...descriptor, value: undefined });
    if (fallback && encoder) Object.defineProperty(DataView.prototype, 'setFloat16', { ...encoder, value: undefined });
    try {
      for (const dtype of [ONNX_INT8, ONNX_UINT8, ONNX_INT4, ONNX_UINT4])
        for (const scaleType of [ONNX_FLOAT, ONNX_FLOAT16])
          for (const outputType of [ONNX_FLOAT, ONNX_FLOAT16])
            for (const layout of ['scalar', 'axis', 'blocked']) {
              const signed = dtype === ONNX_INT8 || dtype === ONNX_INT4,
                packed = dtype === ONNX_INT4 || dtype === ONNX_UINT4;
              const q = Array.from({ length: 9 }, (_, i) => (packed ? ((i * 3) % 16) - (signed ? 8 : 0) : ((i * 33) % 256) - (signed ? 128 : 0)));
              const scaleDims = layout === 'scalar' ? [] : layout === 'axis' ? [3] : [2, 3];
              const count = layout === 'scalar' ? 1 : layout === 'axis' ? 3 : 6;
              const scales = [0.123456789, 1e-39, 1 + 2 ** -23, 65504.1, -0, Infinity].slice(0, count);
              const stored = scaleType === ONNX_FLOAT16 ? scales.map((v) => f16BitsToF32(f32ToF16Bits(v))) : Array.from(Float32Array.from(scales));
              const zero = Array.from({ length: count }, (_, i) => (i % 3) + (signed ? -1 : 1));
              const attrs = [attrInt('output_dtype', outputType), attrSignedInt('axis', layout === 'blocked' ? 0 : -1)];
              if (layout === 'blocked') attrs.push(attrInt('block_size', 2));
              const input = model({
                nodes: [node('DequantizeLinear', ['Wq', 'Ws', 'Zp'], ['W'], attrs)],
                initializers: [
                  encodeTensorProto('Wq', dtype, [3, 3], packed ? int4Bytes(q) : int8Bytes(q)),
                  encodeTensorProto('Ws', scaleType, scaleDims, scaleType === ONNX_FLOAT16 ? f16Bytes(scales) : f32Bytes(scales)),
                  encodeTensorProto('Zp', dtype, scaleDims, packed ? int4Bytes(zero) : int8Bytes(zero)),
                ],
                inputs: [],
                outputs: [valueInfo('W', outputType, [3, 3])],
              });
              const values = Float32Array.from(q, (v, i) => {
                const s = layout === 'scalar' ? 0 : layout === 'axis' ? i % 3 : Math.floor(Math.floor(i / 3) / 2) * 3 + (i % 3);
                return (v - zero[s]) * stored[s];
              });
              const expected =
                outputType === ONNX_FLOAT16 ? new Uint8Array(Uint16Array.from(values, f32ToF16Bits).buffer) : new Uint8Array(Float32Array.from(values).buffer);
              const folded = foldOnnxDequantizeLinear(input);
              assert.equal(folded.foldedNodes, 1);
              assert.deepEqual(foldedRaw(folded.bytes), expected, `${dtype}/${scaleType}/${outputType}/${layout}/fallback=${fallback}`);
            }
    } finally {
      if (descriptor) Object.defineProperty(DataView.prototype, 'getFloat16', descriptor);
      if (encoder) Object.defineProperty(DataView.prototype, 'setFloat16', encoder);
    }
  }
});

test('native/fallback half readers canonicalize signed NaNs exactly as the old converter', () => {
  const scales = new Uint8Array(Uint16Array.of(0xfc01, 0x7c01).buffer);
  const bytes = model({
    nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W'], [attrInt('axis', 1), attrInt('output_dtype', ONNX_FLOAT)])],
    initializers: [encodeTensorProto('Wq', ONNX_INT8, [1, 2], int8Bytes([1, -1])), encodeTensorProto('Ws', ONNX_FLOAT16, [2], scales)],
    inputs: [],
    outputs: [valueInfo('W', ONNX_FLOAT, [1, 2])],
  });
  const values = Float32Array.from([1 * f16BitsToF32(0xfc01), -1 * f16BitsToF32(0x7c01)]);
  assert.deepEqual(foldedRaw(foldOnnxDequantizeLinear(bytes).bytes), new Uint8Array(Float32Array.from(values).buffer));
});

test('shared quantized inputs preserve the old folded-before-original ordering and skipped-node policy', () => {
  const q = encodeTensorProto('Wq', ONNX_INT8, [2], int8Bytes([1, -2]));
  const scale = encodeTensorProto('Ws', ONNX_FLOAT16, [], f16Bytes([0.5]));
  const second = node('DequantizeLinear', ['Wq', 'Ws'], ['Other']);
  const outputs = [valueInfo('W', ONNX_FLOAT16, [2]), valueInfo('Other', ONNX_FLOAT16, [2])];
  const folded = foldOnnxDequantizeLinear(
    model({ nodes: [node('DequantizeLinear', ['Wq', 'Ws'], ['W']), second], initializers: [q, scale], inputs: [], outputs }),
  );
  const expected = model({ nodes: [second], initializers: [encodeTensorProto('W', ONNX_FLOAT16, [2], f16Bytes([0.5, -1])), q, scale], inputs: [], outputs });
  assert.equal(folded.foldedNodes, 1);
  assert.equal(folded.skippedNodes, 1);
  assert.equal(folded.removedInitializers, 0);
  assert.deepEqual(folded.bytes, expected);
});

test('typed-array dequant fast paths match reference across odd/even offsets, negative zero, overflow, and axis layouts', () => {
  const qVals = Array.from({ length: 256 }, (_, i) => i - 128);
  const scales = [-0.5, 0.0, 0.0001220703125, 1000.0];
  for (const outName of ['W', 'W1', 'W12', 'W123']) {
    // Per-column (axis=1, inner=1) f16 output with f16 scales
    const colBytes = model({
      nodes: [node('DequantizeLinear', ['Wq', 'Ws'], [outName], [attrInt('axis', 1)])],
      initializers: [encodeTensorProto('Wq', ONNX_INT8, [64, 4], int8Bytes(qVals)), encodeTensorProto('Ws', ONNX_FLOAT16, [4], f16Bytes(scales))],
      inputs: [],
      outputs: [valueInfo(outName, ONNX_FLOAT16, [64, 4])],
    });
    const decodedScales = scales.map((s) => f16BitsToF32(f32ToF16Bits(s)));
    const expectedCol = f16Bytes(qVals.map((v, i) => v * decodedScales[i % 4]));
    assert.deepEqual(foldedRaw(foldOnnxDequantizeLinear(colBytes).bytes, outName), expectedCol, `col f16 ${outName}`);

    // Per-column (axis=1, inner=1) f32 output
    const colF32Bytes = model({
      nodes: [node('DequantizeLinear', ['Wq', 'Ws'], [outName], [attrInt('axis', 1), attrInt('output_dtype', ONNX_FLOAT)])],
      initializers: [encodeTensorProto('Wq', ONNX_INT8, [64, 4], int8Bytes(qVals)), encodeTensorProto('Ws', ONNX_FLOAT, [4], f32Bytes(scales))],
      inputs: [],
      outputs: [valueInfo(outName, ONNX_FLOAT, [64, 4])],
    });
    const expectedColF32 = f32Bytes(qVals.map((v, i) => v * Math.fround(scales[i % 4])));
    assert.deepEqual(foldedRaw(foldOnnxDequantizeLinear(colF32Bytes).bytes, outName), expectedColF32, `col f32 ${outName}`);

    // Per-row (axis=0, inner=64) with zero point and f32->f16 rounding
    const zp = [1, -2, 0, 5];
    const rowScales = [1 / 3, -0.0003, 512.0, 1 + 2 ** -11];
    const rowBytes = model({
      nodes: [node('DequantizeLinear', ['Wq', 'Ws', 'Wz'], [outName], [attrInt('axis', 0), attrInt('output_dtype', ONNX_FLOAT16)])],
      initializers: [
        encodeTensorProto('Wq', ONNX_INT8, [4, 64], int8Bytes(qVals)),
        encodeTensorProto('Ws', ONNX_FLOAT, [4], f32Bytes(rowScales)),
        encodeTensorProto('Wz', ONNX_INT8, [4], int8Bytes(zp)),
      ],
      inputs: [],
      outputs: [valueInfo(outName, ONNX_FLOAT16, [4, 64])],
    });
    const expectedRow = f16Bytes(qVals.map((v, i) => (v - zp[i >> 6]) * Math.fround(rowScales[i >> 6])));
    assert.deepEqual(foldedRaw(foldOnnxDequantizeLinear(rowBytes).bytes, outName), expectedRow, `row f16 ${outName}`);
  }
});

const QDQ8 = process.env.LC0_QDQ8_MODEL ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';
test('shipped LC0 QDQ8 model folds completely and evaluates like the in-graph version', { skip: skipUnlessAssets(QDQ8) }, async () => {
  const bytes = readFileSync(QDQ8);
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.ok(folded.foldedNodes >= 90, `folded ${folded.foldedNodes}`);
  assert.equal(folded.skippedNodes, 0);
  assert.equal(countOnnxDequantizeLinear(folded.bytes), 0);
  assert.ok(folded.bytesAfter > folded.bytesBefore * 1.8, `f16 weights should roughly double the size (${folded.bytesBefore} -> ${folded.bytesAfter})`);
  ort.setOrtDequantFoldForCurrentThread(false);
  const inGraph = await Lc0OnnxEvaluator.create(bytes);
  ort.setOrtDequantFoldForCurrentThread(true);
  const atLoad = await Lc0OnnxEvaluator.create(bytes);
  ort.setOrtDequantFoldForCurrentThread(null);
  assert.equal(ort.lastOrtDequantFoldSummary()?.foldedNodes, folded.foldedNodes);
  for (const fen of ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4']) {
    const a = await inGraph.evaluate(fen);
    const b = await atLoad.evaluate(fen);
    assert.equal(b.bestMove, a.bestMove, fen);
    assert.ok(Math.abs(a.q - b.q) < 2e-3, `q ${a.q} vs ${b.q}`);
    const priors = new Map(a.legalPriors.map((p) => [p.uci, p.prior]));
    for (const p of b.legalPriors) assert.ok(Math.abs(priors.get(p.uci) - p.prior) < 2e-3, `${fen} ${p.uci}`);
  }
  await inGraph.dispose();
  await atLoad.dispose();
});
