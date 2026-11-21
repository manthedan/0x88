import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';
import { env } from '../src/nn/ortRuntime.ts';

test('worker loss during replacement init has no request ID and cannot settle init before ready; re-init does not duplicate watchers', async (t) => {
  const descriptors = new Map(['self', 'navigator', 'fetch'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const deviceDescriptor = Object.getOwnPropertyDescriptor(env.webgpu, 'device');
  const loss = Promise.withResolvers();
  const disposing = Promise.withResolvers();
  const release = Promise.withResolvers();
  const then = loss.promise.then.bind(loss.promise);
  const subscription = t.mock.method(loss.promise, 'then', (...args) => then(...args));
  Object.defineProperty(env.webgpu, 'device', { configurable: true, writable: true, value: { lost: loss.promise } });
  const pending = new Map();
  const messages = [];
  let handleMessage;
  Object.defineProperty(globalThis, 'self', {
    configurable: true,
    value: {
      addEventListener(_type, listener) {
        handleMessage = listener;
      },
      postMessage(message) {
        messages.push(message);
        pending.get(message.id)?.(message);
      },
    },
  });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent: 'test' } });
  Object.defineProperty(globalThis, 'fetch', { configurable: true, value: async () => new Response(null, { status: 404 }) });
  let created = 0;
  t.mock.method(Lc0OnnxEvaluator, 'create', async () => {
    const index = created++;
    return {
      evaluate: async () => ({}),
      dispose: async () => {
        if (index === 0) {
          disposing.resolve();
          await release.promise;
        }
      },
    };
  });
  let id = 0;
  const init = (name) =>
    new Promise((resolve) => {
      pending.set(++id, resolve);
      handleMessage({ data: { type: 'init', id, modelUrl: `https://test.invalid/${name}.onnx`, ep: 'wasm', cacheModel: false, allowUnverifiedModel: true } });
    });
  try {
    startSearchWorker();
    assert.equal((await init('old')).type, 'ready');
    const replacing = init('new');
    let settled = false;
    void replacing.then(() => {
      settled = true;
    });
    await disposing.promise;
    loss.resolve({ reason: 'unknown', message: 'reset while init is pending' });
    await new Promise((resolve) => setImmediate(resolve));
    const notifications = messages.filter((message) => message.type === 'deviceLost');
    assert.equal(notifications.length, 1);
    assert.equal(Object.hasOwn(notifications[0], 'id'), false);
    assert.equal(settled, false, 'loss notification must not masquerade as an init reply');
    release.resolve();
    assert.equal((await replacing).type, 'ready');
    assert.equal((await init('third')).type, 'ready');
    assert.equal((await init('fourth')).type, 'ready');
    assert.equal(subscription.mock.callCount(), 1, 'same ORT device has one lost-promise listener across re-inits');
    assert.equal(messages.filter((message) => message.type === 'deviceLost').length, 1);
  } finally {
    release.resolve();
    if (deviceDescriptor) Object.defineProperty(env.webgpu, 'device', deviceDescriptor);
    else delete env.webgpu.device;
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});

test('worker search ignores rootMoves supplied in a message', async (t) => {
  const originalSelf = Object.getOwnPropertyDescriptor(globalThis, 'self');
  const worker = new EventTarget();
  let reply;
  worker.postMessage = (message) => reply.resolve(message);
  Object.defineProperty(globalThis, 'self', { configurable: true, value: worker });
  t.after(() => {
    if (originalSelf) Object.defineProperty(globalThis, 'self', originalSelf);
    else delete globalThis.self;
  });
  startSearchWorker({ createHybridEvaluator: () => ({ evaluate: async () => ({ wdl: [0, 1, 0], legalPriors: [] }) }) });
  const send = (data) => {
    reply = Promise.withResolvers();
    worker.dispatchEvent(new MessageEvent('message', { data }));
    return reply.promise;
  };
  const ready = await send({ type: 'init', id: 1, runtime: 'hybrid', modelUrl: 'unused', packUrl: 'unused', ep: 'wasm', cacheModel: false });
  assert.equal(ready.type, 'ready', ready.error);
  const search = await send({ type: 'search', id: 2, input: START_FEN, visits: 2, rootMoves: [] });
  assert.equal(search.type, 'searchResult', search.error);
  assert.ok(search.result.move, 'the starting position still has a playable move');
});

test('a readback AbortError is an inference failure unless the search was cancelled', async (t) => {
  const originalSelf = Object.getOwnPropertyDescriptor(globalThis, 'self');
  const worker = new EventTarget();
  let reply;
  let entered;
  let evaluation;
  worker.postMessage = (message) => reply.resolve(message);
  Object.defineProperty(globalThis, 'self', { configurable: true, value: worker });
  t.after(() => {
    if (originalSelf) Object.defineProperty(globalThis, 'self', originalSelf);
    else delete globalThis.self;
  });
  startSearchWorker({
    createHybridEvaluator: () => ({
      evaluate: () => {
        entered.resolve();
        return evaluation.promise;
      },
    }),
  });
  const emit = (data) => worker.dispatchEvent(new MessageEvent('message', { data }));
  const send = (data) => {
    reply = Promise.withResolvers();
    emit(data);
    return reply.promise;
  };
  const ready = await send({ type: 'init', id: 1, runtime: 'hybrid', modelUrl: 'readback-test', packUrl: 'unused', ep: 'wasm', cacheModel: false });
  assert.equal(ready.type, 'ready', ready.error);
  for (const [id, cancelled] of [
    [2, false],
    [3, true],
  ]) {
    entered = Promise.withResolvers();
    evaluation = Promise.withResolvers();
    const search = send({ type: 'search', id, input: START_FEN, visits: 2, reuseTree: false });
    await entered.promise;
    if (cancelled) emit({ type: 'cancel', target: id });
    evaluation.reject(new DOMException('GPUBuffer.mapAsync failed', 'AbortError'));
    const result = await search;
    if (cancelled) {
      assert.equal(result.type, 'searchResult');
      assert.equal(result.result.cancelled, true);
    } else {
      assert.deepEqual(result, { type: 'error', id, error: 'GPUBuffer.mapAsync failed' });
    }
  }
});

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
  const init = (modelUrl, evalCacheEntries = 0) =>
    send({ type: 'init', modelUrl, ep: 'wasm', cacheModel: false, allowUnverifiedModel: true, evalCacheEntries });
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
