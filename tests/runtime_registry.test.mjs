import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
  CENTIPAWN_TVMJS_WEBGPU_V2,
  parseBrowserRuntimeSelector,
  promotedTvmjsRuntimeForModel,
  shouldAttemptTvmjsRuntime,
} from '../src/nn/runtimeRegistry.ts';

test('Centipawn TVMJS runtime is promoted with ORT fallback metadata', () => {
  assert.equal(CENTIPAWN_TVMJS_WEBGPU_V2.status, 'promoted');
  assert.equal(CENTIPAWN_TVMJS_WEBGPU_V2.runtime, 'tvmjs-webgpu');
  assert.equal(CENTIPAWN_TVMJS_WEBGPU_V2.fallback.runtime, 'ort');
  assert.match(CENTIPAWN_TVMJS_WEBGPU_V2.artifact.manifestUrl, /v2-shape-k16\/manifest\.json$/);
  assert.equal(promotedTvmjsRuntimeForModel(BT4_SOAP_REM_C19000_FINAL_MODEL_ID), CENTIPAWN_TVMJS_WEBGPU_V2);
  assert.equal(shouldAttemptTvmjsRuntime(BT4_SOAP_REM_C19000_FINAL_MODEL_ID, 'auto'), true);
  assert.equal(shouldAttemptTvmjsRuntime(BT4_SOAP_REM_C19000_FINAL_MODEL_ID, 'ort'), false);
});

test('runtime selector accepts TVMJS aliases', () => {
  for (const value of ['tvmjs', 'tvmjs-webgpu', 'webgpu-tvmjs', 'compiled-webgpu']) {
    assert.equal(parseBrowserRuntimeSelector(value), 'tvmjs-webgpu');
  }
});
