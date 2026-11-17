import assert from 'node:assert/strict';
import test from 'node:test';
import { Bt4WorkerSearcher, LQO_NET } from '../src/lc0/bt4Engine.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';
import * as ort from '../src/nn/ortRuntime.ts';

function globals(t, values) {
  const descriptors = new Map(Object.keys(values).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(values)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  t.after(() => {
    ort.setOrtRuntimeDiagnosticOptionsForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
}

for (const level of ['disabled', 'basic', 'extended', 'layout', 'all']) {
  test(`ortGraphOpt=${level} sets session options, forwarding payload and effective diagnostics`, async (t) => {
    globals(t, { location: { search: `?ortGraphOpt=${level}` } });
    assert.equal(ort.sessionOptions(['wasm']).graphOptimizationLevel, level);
    assert.equal(ort.requestedOrtRuntimeDiagnosticOptions().graphOptimizationLevel, level);
    assert.equal(ort.collectOrtSessionDiagnostics().graphOptimizationLevel, level);
    assert.equal((await ort.collectOrtRuntimeDiagnostics()).graphOptimizationLevel, level);
  });
}

test('graph optimization defaults to all, validates values, and worker pins take precedence over location', (t) => {
  globals(t, { location: { search: '' } });
  assert.equal(ort.sessionOptions(['wasm']).graphOptimizationLevel, 'all');
  location.search = '?ortGraphOpt=invalid';
  assert.equal(ort.sessionOptions(['wasm']).graphOptimizationLevel, 'all');
  location.search = '?ortGraphOpt=disabled';
  ort.setOrtRuntimeDiagnosticOptionsForCurrentThread({ graphOptimizationLevel: 'basic' });
  assert.equal(ort.sessionOptions(['wasm']).graphOptimizationLevel, 'basic');
  ort.setOrtRuntimeDiagnosticOptionsForCurrentThread(null);
  assert.equal(ort.sessionOptions(['wasm']).graphOptimizationLevel, 'disabled');
});

class WorkerStub extends EventTarget {
  posted = [];
  terminated = false;
  postMessage(message) {
    this.posted.push(message);
    if (message.type === 'init')
      queueMicrotask(() =>
        this.dispatchEvent(
          new MessageEvent('message', {
            data: {
              type: 'ready',
              id: message.id,
              backend: 'test',
              servedModelUrl: message.modelUrl,
              inputNames: [],
              outputNames: [],
              ortRuntime: { graphOptimizationLevel: message.ortDiagnostics.graphOptimizationLevel, dequantFoldEnabled: message.dequantFold },
            },
          }),
        ),
      );
  }
  terminate() {
    this.terminated = true;
  }
}

test('big-net worker receives page graph/folding options, keeps device model selection, and retires incompatible cached sessions', async (t) => {
  const workers = [];
  globals(t, {
    location: { search: '?ortGraphOpt=disabled&ortDequantFold=0&ortPreferredOutputLocation=gpu-buffer' },
    Worker: class extends WorkerStub {
      constructor() {
        super();
        workers.push(this);
      }
    },
  });
  const searcher = new Bt4WorkerSearcher({ ...LQO_NET, modelUrl: '/models/lc0/lqo_v2.f16.onnx', diagnosticModel: true });
  t.after(() => searcher.dispose());
  await searcher.init();
  const first = workers[0].posted[0];
  assert.equal(first.ortDiagnostics.graphOptimizationLevel, 'disabled');
  assert.equal(first.ortDiagnostics.preferredOutputLocation, 'gpu-buffer');
  assert.equal(first.dequantFold, false, 'app-side folding is independent and must also reach the worker');
  assert.equal(first.deviceModelSelection, true, 'model overrides keep production split-K rules');
  assert.equal(first.ortIoMode, undefined, 'device-selected big nets inherit captured IO, with dynamic-shape preflight falling back before creation');
  assert.equal(first.allowUnverifiedModel, undefined, 'diagnostic config must not weaken model integrity');
  assert.equal(first.modelUrl, '/models/lc0/lqo_v2.f16.onnx');
  assert.equal(searcher.ortRuntime.graphOptimizationLevel, 'disabled');
  assert.equal(searcher.isInitializedFor(), true);
  location.search = '?ortGraphOpt=basic';
  assert.equal(searcher.isInitializedFor(), false);
  await searcher.init();
  assert.equal(workers.length, 2);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers[1].posted[0].ortDiagnostics.graphOptimizationLevel, 'basic');
  assert.equal(workers[1].posted[0].dequantFold, true, 'default app fold remains enabled');
  assert.equal(searcher.ortRuntime.graphOptimizationLevel, 'basic');
});

test('Maia init forwards page diagnostics and preserves the effective worker settings in its ready payload', async (t) => {
  globals(t, { location: { search: '?ortGraphOpt=layout&ortDequantFold=0' } });
  const worker = new WorkerStub();
  const ready = await Maia3BrowserEvaluator.postInit(worker, 'unused-model');
  assert.equal(worker.posted[0].ortDiagnostics.graphOptimizationLevel, 'layout');
  assert.equal(worker.posted[0].dequantFold, false);
  assert.equal(ready.ortRuntime.graphOptimizationLevel, 'layout');
  assert.equal(ready.ortRuntime.dequantFoldEnabled, false);
});

test('search worker applies forwarded levels without the page URL, reports them, and includes diagnostics/folding in re-init identity', async (t) => {
  let handler;
  let seq = 0;
  const pending = new Map();
  globals(t, {
    location: { search: '' },
    navigator: { userAgent: 'test' },
    fetch: async () => new Response(null, { status: 404 }),
    self: {
      addEventListener(_type, callback) {
        handler = callback;
      },
      postMessage(data) {
        pending.get(data.id)?.(data);
      },
    },
  });
  const levels = [];
  t.mock.method(Lc0OnnxEvaluator, 'create', async () => {
    levels.push(ort.sessionOptions(['wasm']).graphOptimizationLevel);
    return { evaluate: async () => ({}), async dispose() {} };
  });
  startSearchWorker();
  const init = (level, fold) =>
    new Promise((resolve) => {
      pending.set(++seq, resolve);
      handler({
        data: {
          type: 'init',
          id: seq,
          modelUrl: 'https://test.invalid/graph-options.onnx',
          ep: 'wasm',
          cacheModel: false,
          allowUnverifiedModel: true,
          ortDiagnostics: { graphOptimizationLevel: level },
          dequantFold: fold,
        },
      });
    });
  for (const level of ['disabled', 'layout', 'basic']) {
    const ready = await init(level, false);
    assert.equal(ready.type, 'ready');
    assert.equal(ready.ortRuntime.graphOptimizationLevel, level);
    assert.equal(ready.ortRuntime.dequantFoldEnabled, false);
  }
  assert.deepEqual(levels, ['disabled', 'layout', 'basic']);
  await init('basic', false);
  assert.equal(levels.length, 3, 'identical config reuses the session');
  await init('basic', true);
  assert.equal(levels.length, 4, 'changing the independent app-fold flag also re-inits');
});

test('real Maia worker handler applies graph options to ORT session creation without a page query', async (t) => {
  const pending = new Map();
  globals(t, {
    location: { search: '' },
    navigator: { userAgent: 'test' },
    onmessage: null,
    postMessage(data) {
      if (data.type !== 'progress') pending.get(data.id)?.(data);
    },
  });
  const levels = [];
  t.mock.method(ort.InferenceSession, 'create', async (_input, options) => {
    levels.push(options.graphOptimizationLevel);
    return { inputNames: [], outputNames: [], async release() {} };
  });
  await import('../src/lc0/maia3Worker.ts');
  let seq = 0;
  const send = (message) =>
    new Promise((resolve) => {
      pending.set(++seq, resolve);
      globalThis.onmessage({ data: { ...message, id: seq } });
    });
  for (const level of ['disabled', 'extended']) {
    const ready = await send({ type: 'init', model: 'unused', ep: 'wasm', ortDiagnostics: { graphOptimizationLevel: level }, dequantFold: false });
    assert.equal(ready.type, 'ready');
    assert.equal(ready.ortRuntime.graphOptimizationLevel, level);
    assert.equal(ready.ortRuntime.dequantFoldEnabled, false);
  }
  await send({ type: 'dispose' });
  assert.deepEqual(levels, ['disabled', 'extended']);
});
