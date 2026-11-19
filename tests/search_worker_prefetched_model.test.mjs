import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { T3_NET } from '../src/lc0/bt4Engine.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import { LC0_MODEL_DOWNLOAD_NOT_APPROVED } from '../src/lc0/modelCache.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { startSearchWorker } from '../src/lc0/searchWorkerCore.ts';
import { resetWebGpuAdapterProbeForTests, resolveOrtSessionPlacement } from '../src/nn/ortRuntime.ts';

test('a WebGPU worker with an unknown vendor uses transferred split-K bytes', async (t) => {
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
  define('navigator', { gpu: { requestAdapter: async () => ({ features: new Set(['shader-f16']), info: { vendor: '' } }) } });
  resetWebGpuAdapterProbeForTests();
  t.after(resetWebGpuAdapterProbeForTests);
  define('caches', undefined);
  const fetch = t.mock.fn(async (input) => assert.fail(`unexpected fetch: ${input}`));
  define('fetch', fetch);
  const modelUrl = `https://test.invalid${new URL(T3_NET.modelUrl, 'https://test.invalid').pathname}`;
  const ep = 'auto';
  assert.equal(selectLc0OrtModelUrl(modelUrl, await resolveOrtSessionPlacement(ep)), modelUrl);
  const splitKUrl = lc0SplitKUrl(modelUrl);
  assert.ok(splitKUrl);
  const model = new TextEncoder().encode('abc').buffer;
  const create = t.mock.method(Lc0OnnxEvaluator, 'create', async () => {
    throw new Error('stop after session input');
  });
  startSearchWorker();
  const load = {
    model,
    url: splitKUrl,
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
    handle({ data: { type: 'init', id: 1, modelUrl, ep, deviceModelSelection: true, cacheModel: true, prefetchedModel: { url: splitKUrl, load } } });
  });
  assert.equal(fetch.mock.callCount(), 0, 'no second model load');
  assert.equal(create.mock.callCount(), 1);
  assert.equal(create.mock.calls[0].arguments[0], model);
  assert.equal(reply.type, 'error');
  assert.match(reply.error, /stop after session input/);
});

for (const { worker, vendor, providers } of [
  { worker: 'WASM-only', providers: ['wasm'] },
  { worker: 'non-Apple WebGPU', vendor: 'nvidia', providers: ['webgpu', 'wasm'] },
]) {
  test(`a ${worker} worker releases transferred split-K bytes before loading the original`, () => {
    const source = `
import assert from 'node:assert/strict';
import { DEFAULT_LC0_ORT_EVALUATOR_DEPS, lc0SplitKUrl } from ${JSON.stringify(new URL('../src/lc0/lc0SmallModelSelection.ts', import.meta.url).href)};
import { startSearchWorker } from ${JSON.stringify(new URL('../src/lc0/searchWorkerCore.ts', import.meta.url).href)};
import { InferenceSession } from ${JSON.stringify(new URL('../src/nn/ortRuntime.ts', import.meta.url).href)};

const vendor = ${JSON.stringify(vendor)};
const adapter = vendor ? { features: new Set(['shader-f16']), info: { vendor } } : null;
Object.defineProperty(globalThis, 'navigator', { value: { gpu: { requestAdapter: async () => adapter } } });
const reply = Promise.withResolvers();
let handle;
globalThis.self = { addEventListener: (_type, listener) => { handle = listener; }, postMessage: reply.resolve };
const originalUrl = ${JSON.stringify(T3_NET.modelUrl)};
const splitKUrl = lc0SplitKUrl(originalUrl);
assert.ok(splitKUrl);
const loadResult = (url, text) => ({
  model: new TextEncoder().encode(text).buffer,
  url, mode: 'memory', cacheStatus: 'unavailable', verification: 'hashed', sha256Valid: true,
  elapsedMs: 0, telemetry: { source: 'network', totalMs: 0 },
});
let transferred;
function handOffSplitK() {
  const load = loadResult(splitKUrl, 'split-K');
  transferred = new WeakRef(load.model);
  handle({ data: {
    type: 'init', id: 1, modelUrl: originalUrl, ep: 'auto', deviceModelSelection: true,
    cacheModel: true, ortIoMode: 'cpu', dequantFold: false, prefetchedModel: { url: splitKUrl, load },
  } });
}
DEFAULT_LC0_ORT_EVALUATOR_DEPS.load = async (url) => {
  assert.equal(url, originalUrl);
  for (let i = 0; i < 5; i++) { await new Promise(setImmediate); global.gc(); }
  assert.equal(transferred.deref(), undefined, 'split-K buffer is still pinned when loading the original');
  return loadResult(url, 'original');
};
const sessions = [];
InferenceSession.create = async (model, options) => {
  sessions.push({ input: new TextDecoder().decode(model), providers: options.executionProviders });
  return { async release() {} };
};
startSearchWorker();
handOffSplitK();
const message = await reply.promise;
assert.equal(message.type, 'ready', message.error);
assert.deepEqual(sessions, [{ input: 'original', providers: ${JSON.stringify(providers)} }]);
assert.equal(message.servedModelUrl, originalUrl);
`;
    const child = spawnSync(process.execPath, ['--expose-gc', '--input-type=module', '-e', source], { encoding: 'utf8', timeout: 15000 });
    assert.equal(child.status, 0, `${child.error ?? ''}\n${child.stdout}\n${child.stderr}`);
  });
}

// Arena's cache-hit consent: the worker's loader must refuse network bytes, not download them.
test('search worker init with noModelDownload refuses to download an uncached model', async (t) => {
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
  define('caches', { has: async () => true, open: async () => ({ match: async () => undefined }) });
  const modelUrl = 'https://test.invalid/models/lc0/uncached.onnx';
  const sha256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
  define('fetch', async (input) => {
    const href = input.url ?? String(input);
    if (href.endsWith('/manifest.json')) return Response.json({ models: [{ file: 'uncached.onnx', url: modelUrl, bytes: 3, sha256 }] });
    assert.fail(`downloaded ${href}`);
  });
  t.mock.method(Lc0OnnxEvaluator, 'create', async () => assert.fail('no session without model bytes'));
  startSearchWorker();
  const reply = await new Promise((resolve) => {
    replies.set(2, resolve);
    handle({ data: { type: 'init', id: 2, modelUrl, ep: 'wasm', cacheModel: true, noModelDownload: true } });
  });
  assert.equal(reply.type, 'error');
  assert.match(reply.error, new RegExp(LC0_MODEL_DOWNLOAD_NOT_APPROVED));
});
