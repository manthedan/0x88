import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as ort from '../src/nn/ortRuntime.ts';
import { createTinyMatmulAddOnnxForTest, f16BitsToF32 } from '../src/lc0/wgslMatmulAddProbe.ts';

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
