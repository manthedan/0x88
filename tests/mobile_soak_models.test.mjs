import assert from 'node:assert/strict';
import test from 'node:test';
import { createSoakBigNetBinding } from '../src/lc0/mobileSoakBigNet.ts';
import { acquireBigNetSearcher, disposeAllBigNetSearchersNow, peekBigNetSearcher } from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from '../src/lc0/lc0SmallModelSelection.ts';

for (const key of ['lqo', 't3', 'bt4']) {
  test(`${key} diagnostic pool and model config cannot collide with the shipped net`, (t) => {
    t.after(disposeAllBigNetSearchersNow);
    const shipped = acquireBigNetSearcher(key);
    const normal = createSoakBigNetBinding(key);
    assert.equal(normal.acquire(), shipped);
    // Even explicitly selecting the shipping URL must get a distinct diagnostic pool namespace.
    const selected = createSoakBigNetBinding(key, BIG_NETS[key].modelUrl);
    const alternate = selected.acquire();
    assert.notEqual(alternate, shipped);
    assert.equal(alternate.config.modelUrl, BIG_NETS[key].modelUrl);
    assert.equal(alternate.config.diagnosticModel, true);
    assert.equal(shipped.config.diagnosticModel, undefined);
    assert.equal(selected.retire(alternate), true);
    const replacement = selected.acquire();
    assert.notEqual(replacement, alternate);
    assert.equal(selected.retire(alternate), false);
    alternate.dispose();
    assert.equal(peekBigNetSearcher(key), shipped);
    assert.equal(selected.acquire(), replacement);
    selected.retire(replacement);
    replacement.dispose();
  });
}

test('plain f16 LQO override retains production worker config and reports no registered split-K rewrite', () => {
  const binding = createSoakBigNetBinding('lqo', '/models/lc0/lqo_v2.f16.onnx');
  const searcher = binding.acquire();
  assert.equal(searcher.config.modelUrl, '/models/lc0/lqo_v2.f16.onnx');
  assert.equal(searcher.config.recommendedBatchSize, BIG_NETS.lqo.recommendedBatchSize);
  assert.equal(searcher.config.streamDownloadProgress, true);
  assert.equal(binding.modelRuntime().servedModelUrl, null);
  searcher.servedModelUrl = binding.config.modelUrl;
  const runtime = binding.modelRuntime(searcher);
  assert.equal(runtime.modelOverride, true);
  assert.equal(runtime.requestedModelUrl, '/models/lc0/lqo_v2.f16.onnx');
  assert.equal(runtime.servedModelUrl, runtime.requestedModelUrl);
  assert.equal(runtime.deviceModelSelection, true);
  assert.equal(runtime.registeredSplitKUrl, null);
  assert.equal(runtime.splitKApplied, false);
  assert.match(runtime.integrity, /verification remains required/);
  binding.retire(searcher);
  searcher.dispose();
});

test('an override matching t3 still follows the production Apple split-K selection and reports the actual served URL', () => {
  const binding = createSoakBigNetBinding('t3', 'https://assets.0x88.app/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx');
  const searcher = binding.acquire();
  const selected = selectLc0OrtModelUrl(binding.config.modelUrl, { providers: ['webgpu'], adapterVendor: 'apple' });
  assert.equal(selected, lc0SplitKUrl(binding.config.modelUrl));
  assert.notEqual(selected, binding.config.modelUrl);
  searcher.servedModelUrl = selected;
  assert.equal(binding.modelRuntime(searcher).splitKApplied, true);
  searcher.servedModelUrl = binding.config.modelUrl; // WASM / non-Apple / original-model fallback.
  assert.equal(binding.modelRuntime(searcher).splitKApplied, false);
  assert.equal(binding.modelRuntime(searcher).servedModelUrl, binding.config.modelUrl);
  binding.retire(searcher);
  searcher.dispose();
});

test('direct diagnostic binding rejects untrusted model URLs as well as query parsing', () => {
  assert.throws(() => createSoakBigNetBinding('lqo', 'https://evil.test/lqo.onnx'), /Model override must/);
});
