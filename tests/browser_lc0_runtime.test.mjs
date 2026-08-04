import assert from 'node:assert/strict';
import test from 'node:test';
import {
  boundedIntValue,
  initialLc0Runtime,
  LC0_WHOLE_MODEL_WEBGPU_RUNTIME,
  lc0EncoderLayers,
  lc0RuntimeLabel,
  lc0WholeModelPhysicalBatch,
  lc0WholeModelRuntimeRequested,
  lc0WholeModelTensorCache,
  normalizeLc0Runtime,
} from '../src/lc0/browserLc0Runtime.ts';

test('LC0 browser runtime aliases normalize consistently across surfaces', () => {
  assert.equal(normalizeLc0Runtime('lc0web'), 'hybrid-ort-heads');
  assert.equal(normalizeLc0Runtime('wgsl'), 'hybrid-wgsl-heads');
  assert.equal(normalizeLc0Runtime('tvmjs-webgpu'), LC0_WHOLE_MODEL_WEBGPU_RUNTIME);
  assert.equal(normalizeLc0Runtime('unknown'), 'onnx');
  assert.equal(lc0RuntimeLabel('hybrid-wgsl-heads'), 'WGSL encoder + WGSL heads');
});

test('LC0 browser runtime query settings share bounded parsing', () => {
  const params = new URLSearchParams('headBackend=wgsl&encoderLayers=99&wholeModelBatch=0&tensorCache=1');
  assert.equal(initialLc0Runtime(params), 'hybrid-wgsl-heads');
  assert.equal(lc0EncoderLayers(params), 32);
  assert.equal(lc0WholeModelPhysicalBatch(params), 1);
  assert.equal(lc0WholeModelTensorCache(params), true);
  assert.equal(boundedIntValue('bad', 8, 1, 64), 8);
});

test('whole-model runtime requests accept the runtime and feature flag aliases', () => {
  assert.equal(lc0WholeModelRuntimeRequested(new URLSearchParams('runtime=whole-onnx-webgpu')), true);
  assert.equal(lc0WholeModelRuntimeRequested(new URLSearchParams('enableTvmjs=1')), true);
  assert.equal(lc0WholeModelRuntimeRequested(new URLSearchParams()), false);
});
