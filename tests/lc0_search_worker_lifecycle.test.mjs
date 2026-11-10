import assert from 'node:assert/strict';
import test from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';

test('worker disposes a replacement after diagnostics fail and keeps its working evaluator', async (t) => {
  const descriptors = new Map(['self', 'navigator', 'fetch'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  let handleMessage;
  const pending = new Map();
  let diagnosticsFail = false;
  const created = [];
  Object.defineProperty(globalThis, 'self', {
    configurable: true,
    value: {
      addEventListener(_type, listener) {
        handleMessage = listener;
      },
      postMessage(message) {
        pending.get(message.id)?.(message);
      },
    },
  });
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      get userAgent() {
        if (diagnosticsFail) throw new Error('diagnostics failed');
        return 'test';
      },
    },
  });
  Object.defineProperty(globalThis, 'fetch', { configurable: true, value: async () => new Response(null, { status: 404 }) });
  t.mock.method(Lc0OnnxEvaluator, 'create', async () => {
    const evaluator = {
      disposals: 0,
      evaluate: async () => ({ bestMove: 'e2e4', fen: 'working evaluator' }),
      dispose() {
        this.disposals += 1;
      },
    };
    created.push(evaluator);
    return evaluator;
  });
  let id = 0;
  const send = (message) =>
    new Promise((resolve) => {
      pending.set(++id, resolve);
      handleMessage({ data: { ...message, id } });
    });
  const init = (modelUrl, evalCacheEntries = 0) => send({ type: 'init', modelUrl, ep: 'wasm', cacheModel: false, evalCacheEntries });
  try {
    startSearchWorker();
    assert.equal((await init('https://test.invalid/old.onnx')).type, 'ready');
    diagnosticsFail = true;
    for (const entries of [0, 2]) {
      const failed = await init(`https://test.invalid/replacement-${entries}.onnx`, entries);
      assert.equal(failed.type, 'error');
      assert.match(failed.error, /diagnostics failed/);
      assert.equal(created.at(-1).disposals, 1);
      assert.equal(created[0].disposals, 0);
      const result = await send({ type: 'evaluate', input: 'unused' });
      assert.equal(result.result.bestMove, 'e2e4');
    }
    diagnosticsFail = false;
    assert.equal((await init('https://test.invalid/replacement-ok.onnx')).type, 'ready');
    assert.equal(created[0].disposals, 1);
    assert.equal(created.at(-1).disposals, 0);
  } finally {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});
