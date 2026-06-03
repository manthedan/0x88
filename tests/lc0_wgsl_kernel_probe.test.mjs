import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as ort from '../src/nn/ortRuntime.ts';
import { createTinyAttentionOutputOnnxForTest, createTinyEncoder0BlockOnnxForTest, createTinyEncoder0FfnOnnxForTest, createTinyMatmulAddOnnxForTest, f16BitsToF32, lc0WebEncoderBlockTensorNames } from '../src/lc0/wgslMatmulAddProbe.ts';

test('f16BitsToF32 decodes representative IEEE half values used by lc0web kernels', () => {
  assert.equal(f16BitsToF32(0x0000), 0);
  assert.equal(Object.is(f16BitsToF32(0x8000), -0), true);
  assert.equal(f16BitsToF32(0x3c00), 1);
  assert.equal(f16BitsToF32(0xc000), -2);
  assert.equal(f16BitsToF32(0x3800), 0.5);
  assert.equal(f16BitsToF32(0x7c00), Infinity);
  assert.equal(f16BitsToF32(0xfc00), -Infinity);
  assert.ok(Number.isNaN(f16BitsToF32(0x7e00)));
  assert.ok(Math.abs(f16BitsToF32(0x0001) - 5.960464477539063e-8) < 1e-15);
});

test('lc0web encoder block tensor names can target later encoder prefixes', () => {
  const names = lc0WebEncoderBlockTensorNames('/encoder3');
  assert.equal(names.qkv.qWeight, '/encoder3/mha/Q/w/w');
  assert.equal(names.outDenseWeight, '/encoder3/mha/out/dense/w/w');
  assert.equal(names.ffnDense2Bias, '/encoder3/ffn/dense2/b/w');
  assert.equal(names.ln2Scale, '/encoder3/ln2/w/scale');
  assert.equal(names.smolgen.smolgenWeight, '/const/smolgen_w');
});

test('tiny MatMul+Add ONNX bytes run through ORT WASM from Uint8Array', async () => {
  ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const weight = new Float32Array(256 * 256);
  const bias = new Float32Array(256);
  weight[0] = 2;
  weight[1 * 256 + 1] = -3;
  bias[0] = 0.5;
  bias[1] = 1.25;
  const model = createTinyMatmulAddOnnxForTest(weight, bias);
  const session = await ort.createOrtSession(model);
  const input = new Float32Array(256);
  input[0] = 4;
  input[1] = -2;
  const outputs = await session.run({ input: new ort.Tensor('float32', input, [1, 256]) });
  const output = outputs.output.data;
  assert.ok(output instanceof Float32Array);
  assert.equal(output.length, 256);
  assert.equal(output[0], 8.5);
  assert.equal(output[1], 7.25);
  assert.equal(output[2], 0);
});

test('tiny attention-output ONNX bytes run projection residual and layernorm through ORT WASM', async () => {
  ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const weight = new Float32Array(256 * 256);
  const bias = new Float32Array(256);
  const scale = new Float32Array(256).fill(1);
  const lnBias = new Float32Array(256);
  for (let i = 0; i < 256; i++) weight[i * 256 + i] = 1;
  const model = createTinyAttentionOutputOnnxForTest(weight, bias, 1, scale, lnBias);
  const session = await ort.createOrtSession(model);
  const attention = new Float32Array(64 * 256);
  const residual = new Float32Array(64 * 256);
  attention[0] = 1;
  attention[1] = -1;
  const outputs = await session.run({
    attention: new ort.Tensor('float32', attention, [64, 256]),
    residual: new ort.Tensor('float32', residual, [64, 256]),
  });
  const output = outputs.output.data;
  assert.ok(output instanceof Float32Array);
  assert.equal(output.length, 64 * 256);
  const expected = 1 / Math.sqrt(2 / 256 + 9.999999974752427e-7);
  assert.ok(Math.abs(output[0] - expected) < 1e-5);
  assert.ok(Math.abs(output[1] + expected) < 1e-5);
  assert.equal(output[2], 0);
  assert.equal(output[256], 0);
});

function layerNormRow(values) {
  const epsilon = 9.999999974752427e-7;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const denom = Math.sqrt(variance + epsilon);
  return values.map((value) => (value - mean) / denom);
}

test('tiny encoder0 FFN ONNX bytes run sqrrelu residual and layernorm through ORT WASM', async () => {
  ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const dense1Weight = new Float32Array(256 * 1024);
  const dense1Bias = new Float32Array(1024);
  const dense2Weight = new Float32Array(1024 * 256);
  const dense2Bias = new Float32Array(256);
  const scale = new Float32Array(256).fill(1);
  const lnBias = new Float32Array(256);
  dense1Weight[0 * 1024 + 0] = 1;
  dense1Weight[1 * 1024 + 1] = 1;
  dense2Weight[0 * 256 + 0] = 1;
  dense2Weight[1 * 256 + 1] = 1;
  const model = createTinyEncoder0FfnOnnxForTest(dense1Weight, dense1Bias, dense2Weight, dense2Bias, 1, scale, lnBias);
  const session = await ort.createOrtSession(model);
  const input = new Float32Array(64 * 256);
  input[0] = 2;
  input[1] = -3;
  const outputs = await session.run({ input: new ort.Tensor('float32', input, [64, 256]) });
  const output = outputs.output.data;
  assert.ok(output instanceof Float32Array);
  assert.equal(output.length, 64 * 256);
  const expected = layerNormRow([6, -3, ...Array(254).fill(0)]);
  assert.ok(Math.abs(output[0] - expected[0]) < 1e-5);
  assert.ok(Math.abs(output[1] - expected[1]) < 1e-5);
  assert.ok(Math.abs(output[2] - expected[2]) < 1e-5);
  assert.equal(output[256], 0);
});

test('tiny encoder0 block ONNX bytes run attention output plus FFN through ORT WASM', async () => {
  ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const outWeight = new Float32Array(256 * 256);
  const outBias = new Float32Array(256);
  const dense1Weight = new Float32Array(256 * 1024);
  const dense1Bias = new Float32Array(1024);
  const dense2Weight = new Float32Array(1024 * 256);
  const dense2Bias = new Float32Array(256);
  const scale = new Float32Array(256).fill(1);
  const bias = new Float32Array(256);
  for (let i = 0; i < 256; i++) outWeight[i * 256 + i] = 1;
  const model = createTinyEncoder0BlockOnnxForTest(outWeight, outBias, 1, scale, bias, dense1Weight, dense1Bias, dense2Weight, dense2Bias, 0, scale, bias);
  const session = await ort.createOrtSession(model);
  const attention = new Float32Array(64 * 256);
  const residual = new Float32Array(64 * 256);
  attention[0] = 1;
  attention[1] = -1;
  const outputs = await session.run({
    attention: new ort.Tensor('float32', attention, [64, 256]),
    residual: new ort.Tensor('float32', residual, [64, 256]),
  });
  const output = outputs.output.data;
  assert.ok(output instanceof Float32Array);
  assert.equal(output.length, 64 * 256);
  const ln1 = layerNormRow([1, -1, ...Array(254).fill(0)]);
  const expected = layerNormRow(ln1);
  assert.ok(Math.abs(output[0] - expected[0]) < 2e-5);
  assert.ok(Math.abs(output[1] - expected[1]) < 2e-5);
  assert.ok(Math.abs(output[2] - expected[2]) < 2e-5);
  assert.equal(output[256], 0);
});
