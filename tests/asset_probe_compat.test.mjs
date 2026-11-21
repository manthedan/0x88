import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { test } from 'node:test';
import { probeAssetUrls } from '../src/lc0/assetProbe.ts';

test('without AbortSignal.any, asset probes settle, clean up, and still forward cancellation', async (t) => {
  const native = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
  Object.defineProperty(AbortSignal, 'any', { configurable: true, value: undefined });
  t.after(() => Object.defineProperty(AbortSignal, 'any', native));
  let headSignal;
  t.mock.method(globalThis, 'fetch', async (_url, { signal }) => {
    headSignal = signal;
    return new Response(null, { status: 503 });
  });
  let groupSignal;
  const status = await probeAssetUrls(['/cached.wasm'], {
    probeCached: async (_url, signal) => {
      groupSignal = signal;
      return true;
    },
  });
  assert.equal(status, 'present');
  assert.ok(headSignal);
  assert.equal(getEventListeners(groupSignal, 'abort').length, 0);

  t.mock.method(globalThis, 'fetch', (_url, { signal }) => {
    headSignal = signal;
    return new Promise(() => {});
  });
  const controller = new AbortController();
  const pending = probeAssetUrls(['/cancelled.wasm'], { signal: controller.signal });
  controller.abort();
  assert.equal(await pending, 'unknown');
  assert.equal(headSignal.aborted, true);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});
