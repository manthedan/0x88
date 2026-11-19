import assert from 'node:assert/strict';
import test from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';

// Analysis Stop disposes a big net whose first download is still running. The
// worker must abort that download so release does not wait it out (the 1.5 s
// disposal timeout, then a hard terminate), and must not commit a partial model
// to Cache Storage. An init queued behind it must not start another download.
test('worker disposal aborts an in-flight model download without writing Cache Storage', async (t) => {
  const descriptors = new Map(['self', 'navigator', 'fetch', 'caches'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  t.after(() => {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const waiting = new Map();
  let handle;
  const define = (key, value) => Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  define('self', {
    addEventListener(_type, listener) {
      handle = listener;
    },
    postMessage(message) {
      if (message.type !== 'downloadProgress') waiting.get(message.id)?.(message);
    },
  });
  define('navigator', { userAgent: 'test' });
  const writes = [];
  define('caches', {
    async open() {
      return {
        match: async () => undefined,
        put: async (request) => writes.push(request.url),
        delete: async () => false,
        keys: async () => [],
      };
    },
    has: async () => false,
  });
  const bytes = 4_000_000;
  const started = Promise.withResolvers();
  let downloadSignal;
  let downloads = 0;
  define('fetch', async (input) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.endsWith('manifest.json'))
      return Response.json({ models: [{ file: 'abort-net.onnx', url: 'https://test.invalid/abort-net.onnx', bytes, sha256: 'a'.repeat(64) }] });
    downloads++;
    downloadSignal = input.signal;
    let sent = false;
    const body = new ReadableStream({
      pull(controller) {
        if (!sent) {
          sent = true;
          controller.enqueue(new Uint8Array(1_000_000));
          started.resolve();
          return;
        }
        // The rest of the model never arrives unless the request is aborted.
        return new Promise((_resolve, reject) => {
          if (input.signal.aborted) reject(input.signal.reason);
          else input.signal.addEventListener('abort', () => reject(input.signal.reason), { once: true });
        });
      },
    });
    return new Response(body, { headers: { 'content-length': String(bytes) } });
  });
  t.mock.method(Lc0OnnxEvaluator, 'create', async () => assert.fail('an aborted download must not create a session'));
  const send = (id, message) =>
    new Promise((resolve) => {
      waiting.set(id, resolve);
      handle({ data: { ...message, id } });
    });

  startSearchWorker();
  const initMessage = {
    type: 'init',
    modelUrl: 'https://test.invalid/abort-net.onnx',
    ep: 'wasm',
    cacheModel: true,
    reportDownloadProgress: true,
  };
  const init = send(1, initMessage);
  await started.promise;
  const queuedInit = send(2, { ...initMessage, evalCacheEntries: 64 });
  const disposed = send(-1, { type: 'dispose' });
  let timer;
  const deadline = new Promise((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error('disposal waited for the download to finish')), 1000);
  });
  try {
    const [initReply, queuedReply, disposeReply] = await Promise.race([Promise.all([init, queuedInit, disposed]), deadline]);
    assert.equal(initReply.type, 'error');
    assert.equal(queuedReply.type, 'error');
    assert.equal(disposeReply.type, 'disposed');
  } finally {
    clearTimeout(timer);
  }
  assert.equal(downloadSignal.aborted, true);
  assert.equal(downloads, 1, 'a retiring worker starts no queued download');
  assert.deepEqual(writes, [], 'no partial model may be committed to Cache Storage');
});
