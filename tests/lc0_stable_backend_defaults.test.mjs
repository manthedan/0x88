import test from 'node:test';
import assert from 'node:assert/strict';
import { Lc0WebHybridEvaluator } from '../src/lc0/wgslMatmulAddProbe.ts';

test('lc0web hybrid evaluator keeps stable ORT-head defaults protected', () => {
  const evaluator = new Lc0WebHybridEvaluator({
    packUrl: '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json',
    verifyShards: false,
  });
  assert.equal(evaluator.headBackend, 'ort');
  assert.equal(evaluator.inputBackend, 'js');
  assert.equal(evaluator.legalPriorsBackend, 'js');
  assert.equal(evaluator.encoderKernelVariant, 'hand');
  assert.equal(evaluator.wgslBatchMode, 'physical');
  assert.equal(evaluator.layers, 10);
});

test('lc0web hybrid evaluator requires explicit opt-in for experimental WGSL/WASM/GPU paths', () => {
  const evaluator = new Lc0WebHybridEvaluator({
    packUrl: '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json',
    verifyShards: false,
    headBackend: 'wgsl',
    inputBackend: 'wasm',
    legalPriorsBackend: 'gpu',
    encoderKernelVariant: 'tvm-packed-f16',
    wgslBatchMode: 'serial',
  });
  assert.equal(evaluator.headBackend, 'wgsl');
  assert.equal(evaluator.inputBackend, 'wasm');
  assert.equal(evaluator.legalPriorsBackend, 'gpu');
  assert.equal(evaluator.encoderKernelVariant, 'tvm-packed-f16');
  assert.equal(evaluator.wgslBatchMode, 'serial');
});

test('lc0web hybrid evaluator rejects GPU legal priors without WGSL heads before runtime init', () => {
  assert.throws(() => new Lc0WebHybridEvaluator({
    packUrl: '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json',
    verifyShards: false,
    legalPriorsBackend: 'gpu',
  }), /GPU legal-prior backend requires WGSL heads/);
});

test('lc0web hybrid evaluator allows explicit mixed TVM FFN opt-in', () => {
  const evaluator = new Lc0WebHybridEvaluator({
    packUrl: '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json',
    verifyShards: false,
    encoderKernelVariant: 'mixed-tvm-ffn',
  });
  assert.equal(evaluator.headBackend, 'ort');
  assert.equal(evaluator.inputBackend, 'js');
  assert.equal(evaluator.legalPriorsBackend, 'js');
  assert.equal(evaluator.encoderKernelVariant, 'mixed-tvm-ffn');
});

test('lc0web hybrid evaluator allows explicit mixed TVM FFN plus out-projection opt-in', () => {
  const evaluator = new Lc0WebHybridEvaluator({
    packUrl: '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json',
    verifyShards: false,
    encoderKernelVariant: 'mixed-tvm-ffn-outproj',
  });
  assert.equal(evaluator.headBackend, 'ort');
  assert.equal(evaluator.inputBackend, 'js');
  assert.equal(evaluator.legalPriorsBackend, 'js');
  assert.equal(evaluator.encoderKernelVariant, 'mixed-tvm-ffn-outproj');
});
