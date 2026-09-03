import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { test } from 'node:test';
import * as ort from '../src/nn/ortRuntime.ts';
import {
  ONNX_FLOAT, ONNX_FLOAT16, ONNX_INT4, ONNX_INT8, ProtoWriter, countOnnxDequantizeLinear, encodeTensorProto,
  f16BitsToF32, f32ToF16Bits, foldOnnxDequantizeLinear, parseProtoFields,
} from '../src/nn/onnxDequantFold.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

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
  values.forEach((v, i) => { out[i >> 1] |= (v & 0xf) << ((i & 1) * 4); });
  return out;
}
function matmul(x, rows, cols, w, n) {
  const y = new Float32Array(rows * n);
  for (let r = 0; r < rows; r++) for (let j = 0; j < n; j++) { let acc = 0; for (let k = 0; k < cols; k++) acc += x[r * cols + k] * w[k * n + j]; y[r * n + j] = acc; }
  return y;
}
const wasmSession = (bytes) => ort.InferenceSession.create(bytes, ort.sessionOptions(['wasm']));

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
    inputs: [valueInfo('X', ONNX_FLOAT, [2, 4])], outputs: [valueInfo('Y', ONNX_FLOAT, [2, 3])],
  });
  assert.equal(countOnnxDequantizeLinear(bytes), 1);
  const folded = foldOnnxDequantizeLinear(bytes);
  assert.equal(folded.foldedNodes, 1);
  assert.equal(folded.removedInitializers, 2);
  assert.equal(countOnnxDequantizeLinear(folded.bytes), 0);
  assert.equal(parseProtoFields(folded.bytes).length, parseProtoFields(bytes).length, 'top-level fields preserved');
  const x = Float32Array.from([1, 2, 3, 4, -1, 0.5, 2, -3]);
  const expected = matmul(x, 2, 4, q.map((v, i) => v * scale[i % 3]), 3);
  for (const [label, session] of [['original', await wasmSession(bytes)], ['folded', await wasmSession(folded.bytes)]]) {
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
    inputs: [valueInfo('X', ONNX_FLOAT, [1, 4])], outputs: [valueInfo('Y', ONNX_FLOAT, [1, 3])],
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
    inputs: [valueInfo('X', ONNX_FLOAT16, [1, 4])], outputs: [valueInfo('Y', ONNX_FLOAT16, [1, 3])],
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
  got.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) <= Math.abs(expected[i]) * 4e-3 + 1e-3, `Y[${i}] ${v} vs ${expected[i]}`));
  await ort.releaseOrtSession(session);
});

const QDQ8 = process.env.LC0_QDQ8_MODEL ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';
const qdq8Present = existsSync(QDQ8) && statSync(QDQ8).size > 0;
test('shipped LC0 QDQ8 model folds completely and evaluates like the in-graph version', { skip: !qdq8Present && `missing ${QDQ8}` }, async () => {
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
