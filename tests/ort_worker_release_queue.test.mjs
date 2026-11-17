import assert from 'node:assert/strict';
import test from 'node:test';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';
import * as ort from '../src/nn/ortRuntime.ts';

function globals(t, values) {
  const before = Object.getOwnPropertyDescriptor(ort.env.webgpu, 'device');
  const events = [];
  const device = { destroy: () => events.push('destroy') };
  Object.defineProperty(ort.env.webgpu, 'device', { configurable: true, value: device });
  const descriptors = new Map();
  for (const [key, value] of Object.entries(values)) {
    descriptors.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  t.after(() => {
    if (before) Object.defineProperty(ort.env.webgpu, 'device', before);
    else delete ort.env.webgpu.device;
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  return events;
}

for (const kind of ['search', 'maia']) {
  test(`${kind} worker drains its ExclusiveQueue before release/device destroy/ack and seals late admission`, async (t) => {
    const replies = [];
    const waiting = new Map();
    let handle;
    const post = (data) => {
      if (data.type !== 'progress') {
        replies.push(data);
        waiting.get(data.id)?.(data);
      }
    };
    const events = globals(t, {
      self: {
        addEventListener(_type, fn) {
          handle = fn;
        },
        postMessage: post,
      },
      navigator: { userAgent: 'test' },
      fetch: async () => new Response(null, { status: 404 }),
      onmessage: null,
      postMessage: post,
    });
    const entered = Promise.withResolvers();
    const finishRun = Promise.withResolvers();
    const releasing = Promise.withResolvers();
    const finishRelease = Promise.withResolvers();
    const run = async () => {
      events.push('run');
      entered.resolve();
      await finishRun.promise;
      events.push('run-end');
      return kind === 'search' ? { bestMove: 'e2e4' } : { logits_move: { data: new Float32Array(4352) }, logits_value: { data: new Float32Array(3) } };
    };
    const release = async () => {
      events.push('release');
      releasing.resolve();
      await finishRelease.promise;
      events.push('released');
    };
    if (kind === 'search') {
      t.mock.method(Lc0OnnxEvaluator, 'create', async () => ({ evaluate: run, evaluateBatch: run, dispose: release }));
      startSearchWorker();
    } else {
      t.mock.method(ort.InferenceSession, 'create', async () => ({ inputNames: [], outputNames: [], run, release }));
      await import('../src/lc0/maia3Worker.ts');
      handle = globalThis.onmessage;
    }
    const send = (id, message) =>
      new Promise((resolve) => {
        waiting.set(id, resolve);
        handle({ data: { ...message, id } });
      });
    const ready = await send(
      1,
      kind === 'search'
        ? { type: 'init', modelUrl: 'https://test.invalid/model.onnx', ep: 'wasm', cacheModel: false, allowUnverifiedModel: true }
        : { type: 'init', model: 'unused', ep: 'wasm', dequantFold: false },
    );
    assert.equal(ready.type, 'ready');
    const evaluated = send(
      2,
      kind === 'search' ? { type: 'evaluate', input: 'unused' } : { type: 'evaluate', tokens: new ArrayBuffer(3072), eloSelf: 1500, eloOppo: 1500 },
    );
    await entered.promise;
    const disposed = send(-1, { type: 'dispose' });
    handle({ data: { type: 'evaluate', id: 3, input: 'late' } });
    assert.deepEqual(events, ['run']);
    finishRun.resolve();
    await evaluated;
    await releasing.promise;
    assert.deepEqual(events, ['run', 'run-end', 'release']);
    assert.equal(
      replies.some((r) => r.type === 'disposed'),
      false,
    );
    finishRelease.resolve();
    assert.equal((await disposed).type, 'disposed');
    assert.deepEqual(events, ['run', 'run-end', 'release', 'released', 'destroy']);
    assert.equal(ort.env.webgpu.device, undefined);
    assert.equal(
      replies.some((r) => r.id === 3),
      false,
    );
  });
}
