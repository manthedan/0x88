import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
  CENTIPAWN_TVMJS_WEBGPU_V2,
  parseBrowserRuntimeSelector,
  promotedTvmjsRuntimeForModel,
  shouldAttemptTvmjsRuntime,
  shouldAttemptCustomRuntime,
  S4FIX_V_EMA_MODEL_ID,
  squareformerModelSettings,
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

test('s4fix is ORT-only with model-scoped temperature and native history', () => {
  assert.deepEqual(squareformerModelSettings(S4FIX_V_EMA_MODEL_ID), { policyTemperature: 2, nativeHistory: true });
  assert.deepEqual(squareformerModelSettings(BT4_SOAP_REM_C19000_FINAL_MODEL_ID), { policyTemperature: 1, nativeHistory: false });
  assert.deepEqual(squareformerModelSettings('unregistered'), { policyTemperature: 1, nativeHistory: false });
  assert.equal(promotedTvmjsRuntimeForModel(S4FIX_V_EMA_MODEL_ID), undefined);
  for (const selector of ['auto', 'ort', 'tvmjs-webgpu', 'custom-webgpu']) {
    assert.equal(shouldAttemptTvmjsRuntime(S4FIX_V_EMA_MODEL_ID, selector), false);
    assert.equal(shouldAttemptCustomRuntime(S4FIX_V_EMA_MODEL_ID, selector), false);
  }
});

test('Centipawn model ids follow served file names, so research overrides keep their compiled runtimes', async () => {
  const registry = await import('../src/nn/runtimeRegistry.ts');
  const { centipawnModelIdForUrls } = registry;
  assert.equal(centipawnModelIdForUrls('/models/s4fix_v_ema.onnx', '/models/s4fix_v_ema.meta.json'), registry.S4FIX_V_EMA_MODEL_ID);
  assert.equal(centipawnModelIdForUrls('/models/bt4_soap_rem_c19000_final.onnx', '/models/bt4_soap_rem_c19000_final.meta.json'), registry.BT4_SOAP_REM_C19000_FINAL_MODEL_ID);
  assert.equal(centipawnModelIdForUrls('/models/bt4_anneal_muon_best.onnx'), registry.BT4_ANNEAL_MUON_BEST_MODEL_ID);
  assert.equal(centipawnModelIdForUrls('/models/unregistered.onnx', '/models/unregistered.meta.json'), null);
  assert.ok(registry.promotedTvmjsRuntimeForModel(centipawnModelIdForUrls('/models/bt4_soap_rem_c19000_final.onnx')));
});
