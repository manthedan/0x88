import assert from 'node:assert/strict';
import test from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';

// Phone cold load whose cache.put failed (largeNetDownload.ts): the page hands the
// prefetch worker's verified bytes to the search worker, which must create the
// session from them instead of downloading the net a second time.
test('search worker init loads a prefetched model from memory, once, without fetching', async (t) => {
  const descriptors = new Map(['self', 'navigator', 'fetch', 'caches'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  t.after(() => {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const define = (key, value) => Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  let handle;
  const replies = new Map();
  define('self', {
    addEventListener(_type, listener) {
      handle = listener;
    },
    postMessage(message) {
      replies.get(message.id)?.(message);
    },
  });
  define('navigator', { userAgent: 'test' });
  define('caches', undefined);
  define('fetch', async () => assert.fail('a prefetched model must not be fetched again'));
  const created = [];
  t.mock.method(Lc0OnnxEvaluator, 'create', async (model) => {
    created.push(new TextDecoder().decode(model));
    throw new Error('stop after session input');
  });
  startSearchWorker();
  const modelUrl = 'https://test.invalid/models/lc0/prefetched.onnx';
  const model = new TextEncoder().encode('abc').buffer;
  const load = {
    model,
    url: modelUrl,
    mode: 'memory',
    cacheStatus: 'unavailable',
    bytes: 3,
    sha256Valid: true,
    verification: 'hashed',
    elapsedMs: 0,
    telemetry: { source: 'network', totalMs: 0 },
  };
  const reply = await new Promise((resolve) => {
    replies.set(1, resolve);
    handle({ data: { type: 'init', id: 1, modelUrl, ep: 'wasm', cacheModel: true, prefetchedModel: { url: modelUrl, load } } });
  });
  assert.equal(reply.type, 'error');
  assert.match(reply.error, /stop after session input/);
  assert.deepEqual(created, ['abc']);
});
