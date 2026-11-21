import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import { BT4_NET, Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';
import { prefetchLargeNetToCache } from '../src/lc0/largeNetDownload.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import { LC0_MODEL_DOWNLOAD_NOT_APPROVED, loadLc0ModelForOrt, prefetchLc0ModelToCache } from '../src/lc0/modelCache.ts';
import { resetWebGpuAdapterProbeForTests, resolveOrtSessionPlacement } from '../src/nn/ortRuntime.ts';

// Phone cold loads: the net is downloaded, verified and cached by a throwaway
// worker that is terminated before the session worker starts, so the session
// load is a cached load (cold-load-copies analysis, reduction #1).

const modelUrl = 'https://example.test/models/lc0/prefetch.onnx';
const sha256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'; // sha256('abc')
const config = { ...BT4_NET, modelUrl };
const t3Url = `https://example.test/models/lc0/${T3_NET.modelUrl.split('/').at(-1)}`;
const t3SplitKUrl = lc0SplitKUrl(t3Url);
const t3Config = { ...T3_NET, modelUrl: t3Url };
const tick = () => new Promise(setImmediate);

function replaceGlobal(t, name, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else delete globalThis[name];
  });
}

// The real worker module runs in this realm (this test process only): its
// `self` is a stand-in the fake Worker below bridges to.
const workerScope = new EventTarget();
let toPrefetchOwner = () => assert.fail('no prefetch worker is listening');
globalThis.self = workerScope;
await import('../src/lc0/modelPrefetchWorker.ts');

/** Cache Storage, the app manifest and the network; `put` and the worker's transfer can be made to fail. */
function environment(t, { phone = true, putFails = false, postFails = false, holdPrefetch = false, query = '', files = [modelUrl] } = {}) {
  const trace = [];
  const responses = new Map();
  const counts = { downloads: 0, urls: [] };
  workerScope.postMessage = (data, options) => {
    if (postFails && options?.transfer?.length) throw new DOMException('cannot transfer', 'DataCloneError');
    toPrefetchOwner(data);
  };
  replaceGlobal(t, 'location', new URL(`https://example.test/analysis?memoryClass=${phone ? 'phone' : 'desktop'}${query}`));
  replaceGlobal(t, 'localStorage', { getItem: () => null, setItem: () => undefined });
  const cache = {
    async match(request) {
      return responses.get(request.url)?.clone();
    },
    async put(request, response) {
      if (putFails) throw new DOMException('quota', 'QuotaExceededError');
      responses.set(request.url, response);
    },
    async keys(request) {
      return [...responses.keys()].filter((key) => !request || key === request.url).map((key) => new Request(key));
    },
    async delete(request) {
      return responses.delete(request.url);
    },
  };
  replaceGlobal(t, 'caches', { has: async () => true, open: async () => cache });
  t.mock.method(globalThis, 'fetch', async (input) => {
    const href = input.url ?? String(input);
    if (href.endsWith('/models/lc0/manifest.json'))
      return Response.json({ models: files.map((url) => ({ file: url.split('/').at(-1), url, bytes: 3, sha256 })) });
    assert.ok(files.includes(href), href);
    counts.downloads++;
    counts.urls.push(href);
    trace.push('download');
    return new Response('abc');
  });
  // Prefetch workers bridge to the real worker module; search workers answer init.
  const workers = [];
  replaceGlobal(
    t,
    'Worker',
    class extends EventTarget {
      constructor(url) {
        super();
        this.kind = String(url).includes('modelPrefetchWorker') ? 'prefetch' : 'search';
        this.messages = [];
        this.transfers = [];
        this.terminated = false;
        workers.push(this);
        trace.push(`${this.kind}:spawn`);
      }
      postMessage(message, transfer = []) {
        this.messages.push(message);
        this.transfers.push(transfer);
        if (this.kind === 'prefetch') {
          if (holdPrefetch) return; // a download that never finishes on its own
          toPrefetchOwner = (data) => {
            if (!this.terminated) this.dispatchEvent(new MessageEvent('message', { data }));
          };
          workerScope.dispatchEvent(new MessageEvent('message', { data: message }));
        } else if (message.type === 'init') {
          trace.push('search:init');
          queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', { data: { id: message.id, type: 'ready', backend: 'mock-webgpu' } })));
        } else if (message.type === 'dispose') {
          queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', { data: { id: -1, type: 'disposed' } })));
        }
      }
      terminate() {
        this.terminated = true;
        trace.push(`${this.kind}:terminate`);
      }
    },
  );
  return { trace, counts, workers, responses };
}

test('phone cold load: prefetch worker caches the net and is terminated before the session worker loads it from cache', async (t) => {
  const env = environment(t);
  const searcher = new Bt4WorkerSearcher(config);
  const progress = [];
  searcher.onDownloadProgress = (loaded, total) => progress.push([loaded, total]);
  assert.equal(await searcher.init(), 'mock-webgpu');
  assert.deepEqual(env.trace, ['prefetch:spawn', 'download', 'prefetch:terminate', 'search:spawn', 'search:init']);
  assert.deepEqual(env.workers[0].messages[0], { type: 'prefetch', modelUrl, requestPersistentStorage: true, minimumFreeBytesAfterCache: 64 * 1024 * 1024 });
  assert.deepEqual(progress.at(-1), [3, 3], 'the download still reports progress');
  assert.equal(env.responses.has(modelUrl), true);
  // What the session worker's loader now sees: a cache hit, no second download.
  const load = await loadLc0ModelForOrt(modelUrl, { cache: true, signal: AbortSignal.timeout(5000) });
  assert.equal(load.cacheStatus, 'hit');
  assert.equal(load.sha256Valid, true);
  assert.equal(env.counts.downloads, 1);
  await searcher.dispose();
  // Warm: the entry exists. This test realm has no IndexedDB, so it has no verification
  // metadata either; key presence alone decides, and the session loader verifies it.
  env.trace.length = 0;
  const warm = new Bt4WorkerSearcher(config);
  assert.equal(await warm.init(), 'mock-webgpu');
  assert.deepEqual(env.trace, ['search:spawn', 'search:init'], 'no prefetch worker for a cached net');
  await warm.dispose();
});

/** An adapter for the prefetch worker's (and session worker's) own probe. */
function adapter(t, { vendor = 'apple', shaderF16 = true } = {}) {
  replaceGlobal(t, 'navigator', { gpu: { requestAdapter: async () => ({ features: new Set(shaderF16 ? ['shader-f16'] : []), info: { vendor } }) } });
  resetWebGpuAdapterProbeForTests();
  t.after(resetWebGpuAdapterProbeForTests);
}

test('t3 prefetch: the worker picks the file with its own probe without importing ONNX Runtime', async (t) => {
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (context.parentURL?.endsWith('/modelPrefetchWorker.ts')) assert.doesNotMatch(specifier, /ortRuntime|lc0SmallModelSelection|onnxruntime-web/);
      return nextResolve(specifier, context);
    },
  });
  t.after(() => hook.deregister());
  const env = environment(t, { files: [t3Url, t3SplitKUrl] });
  adapter(t);
  const searcher = new Bt4WorkerSearcher(t3Config);
  assert.equal(await searcher.init(), 'mock-webgpu');
  const request = env.workers[0].messages[0];
  assert.equal(request.modelUrl, t3Url, 'the page sends the shipped URL, unresolved');
  assert.equal(request.selectFor, 'auto', 'with the EP the session worker is sent');
  assert.deepEqual(env.counts.urls, [t3SplitKUrl]);
  assert.deepEqual(env.workers[1].messages[0].prefetchedModel, { url: t3SplitKUrl, load: undefined });
  await searcher.dispose();
  // Warm: the worker finds the selected file by key and downloads nothing.
  env.trace.length = 0;
  const warm = new Bt4WorkerSearcher(t3Config);
  assert.equal(await warm.init(), 'mock-webgpu');
  assert.deepEqual(env.trace, ['prefetch:spawn', 'prefetch:terminate', 'search:spawn', 'search:init']);
  assert.equal(env.counts.downloads, 1);
  assert.deepEqual(env.workers.at(-1).messages[0].prefetchedModel, { url: t3SplitKUrl, load: undefined });
  await warm.dispose();
});

test('t3 prefetch uses the EP the session worker will get, not the page preference', async (t) => {
  // ?ortEp=webgpu page on an Apple adapter without shader-f16: the worker is sent `auto`,
  // resolves to WASM and loads the original t3, so that is the file to prefetch.
  const env = environment(t, { query: '&ortEp=webgpu', files: [t3Url, t3SplitKUrl] });
  adapter(t, { shaderF16: false });
  const searcher = new Bt4WorkerSearcher(t3Config);
  assert.equal(await searcher.init(), 'mock-webgpu');
  assert.equal(env.workers[0].messages[0].selectFor, 'auto');
  assert.deepEqual(env.counts.urls, [t3Url]);
  assert.equal(selectLc0OrtModelUrl(t3Url, await resolveOrtSessionPlacement()), t3SplitKUrl, 'the page preference alone would have picked split-K');
  await searcher.dispose();
});

test('desktop init does not prefetch and creates its worker without awaiting', async (t) => {
  const env = environment(t, { phone: false });
  const searcher = new Bt4WorkerSearcher(config);
  const initializing = searcher.init();
  assert.deepEqual(env.trace, ['search:spawn', 'search:init'], 'worker created and init posted synchronously');
  assert.equal(await initializing, 'mock-webgpu');
  assert.equal(env.counts.downloads, 0, 'the search worker owns the download, as before');
  await searcher.dispose();
});

test('dispose (Stop, seat swap) during the prefetch terminates its worker and never starts the session', async (t) => {
  const env = environment(t, { holdPrefetch: true });
  const searcher = new Bt4WorkerSearcher(config);
  const initializing = searcher.init();
  for (let i = 0; i < 50 && env.workers.length === 0; i++) await tick();
  const prefetch = env.workers[0];
  assert.equal(prefetch?.kind, 'prefetch');
  await searcher.dispose();
  await assert.rejects(initializing, /worker disposed/);
  assert.equal(prefetch.terminated, true);
  assert.equal(env.workers.length, 1, 'no session worker after Stop');
});

test('a failed cache.put hands the verified bytes to the session worker: one download', async (t) => {
  const env = environment(t, { putFails: true, files: [t3Url, t3SplitKUrl] });
  adapter(t);
  const searcher = new Bt4WorkerSearcher(t3Config);
  assert.equal(await searcher.init(), 'mock-webgpu');
  assert.deepEqual(env.trace, ['prefetch:spawn', 'download', 'prefetch:terminate', 'search:spawn', 'search:init']);
  assert.equal(env.responses.size, 0);
  const search = env.workers[1];
  const { prefetchedModel } = search.messages[0];
  assert.equal(search.messages[0].modelUrl, t3Url);
  assert.equal(prefetchedModel.url, t3SplitKUrl);
  assert.equal(prefetchedModel.load.mode, 'memory');
  assert.equal(prefetchedModel.load.sha256Valid, true);
  assert.equal(new TextDecoder().decode(prefetchedModel.load.model), 'abc');
  assert.deepEqual(search.transfers[0], [prefetchedModel.load.model], 'transferred, not copied');
  assert.equal(env.counts.downloads, 1);
  await searcher.dispose();
});

test('a reply the prefetch worker cannot post falls back to the normal load instead of hanging', async (t) => {
  const env = environment(t, { putFails: true, postFails: true });
  const warn = t.mock.method(console, 'warn', () => undefined);
  const searcher = new Bt4WorkerSearcher(config);
  assert.equal(await searcher.init(), 'mock-webgpu');
  assert.deepEqual(env.trace, ['prefetch:spawn', 'download', 'prefetch:terminate', 'search:spawn', 'search:init']);
  assert.equal(env.workers[1].messages[0].prefetchedModel, undefined, 'the session worker loads by URL');
  assert.match(String(warn.mock.calls[0].arguments[1]), /cannot transfer/);
  await searcher.dispose();
});

test('a messageerror on the prefetch reply falls back and terminates the worker', async (t) => {
  environment(t);
  t.mock.method(console, 'warn', () => undefined);
  const worker = new EventTarget();
  let terminated = false;
  worker.postMessage = () => queueMicrotask(() => worker.dispatchEvent(new MessageEvent('messageerror')));
  worker.terminate = () => {
    terminated = true;
  };
  const result = await prefetchLargeNetToCache(
    config,
    { signal: new AbortController().signal, ep: 'auto', load: {} },
    { cached: async () => false, spawn: () => worker, reclaimGapMs: () => 0 },
  );
  assert.deepEqual(result, { status: 'fallback' });
  assert.equal(terminated, true);
});

test('prefetch reports put failure and refuses to download what Cache Storage cannot keep', async (t) => {
  const env = environment(t, { putFails: true });
  const manifestEntry = { file: 'prefetch.onnx', url: modelUrl, bytes: 3, sha256 };
  assert.equal((await prefetchLc0ModelToCache(modelUrl, { manifestEntry, channelUrl: null })).mode, 'memory');
  assert.equal(env.counts.downloads, 1);
  replaceGlobal(t, 'caches', undefined);
  await assert.rejects(prefetchLc0ModelToCache(modelUrl, { manifestEntry, channelUrl: null }), /cannot be cached here \(unavailable\)/);
  replaceGlobal(t, 'navigator', { storage: { estimate: async () => ({ quota: 100, usage: 0 }) } });
  replaceGlobal(t, 'caches', { has: async () => true, open: async () => ({ match: async () => undefined }) });
  await assert.rejects(prefetchLc0ModelToCache(modelUrl, { manifestEntry, channelUrl: null }), /cannot be cached here \(quota-limited\)/);
  assert.equal(env.counts.downloads, 1, 'neither refusal downloads');
});

test('the session load waits out the phone reclaim gap counted from the prefetch worker termination', async (t) => {
  environment(t);
  let terminatedAt = Number.NaN;
  const worker = new EventTarget();
  worker.postMessage = () => queueMicrotask(() => worker.dispatchEvent(new MessageEvent('message', { data: { type: 'done', cached: true, modelUrl } })));
  worker.terminate = () => {
    terminatedAt = performance.now();
  };
  const result = await prefetchLargeNetToCache(
    config,
    { signal: new AbortController().signal, ep: 'auto', load: {} },
    { cached: async () => false, spawn: () => worker, reclaimGapMs: () => 40 },
  );
  assert.deepEqual(result, { status: 'prefetched', modelUrl });
  assert.ok(performance.now() - terminatedAt >= 35, 'resolved only after the gap');
});

// Arena's cache-hit consent approves no download (arenaBrowser.ts bigNetFor).
test('a noDownload load serves a cache hit but refuses to download a missing or unreadable entry', async (t) => {
  const env = environment(t);
  const manifestEntry = { file: 'prefetch.onnx', url: modelUrl, bytes: 3, sha256 };
  const options = { cache: true, manifestEntry, channelUrl: null, noDownload: true };
  await assert.rejects(loadLc0ModelForOrt(modelUrl, options), new RegExp(LC0_MODEL_DOWNLOAD_NOT_APPROVED));
  env.responses.set(modelUrl, { clone: () => ({ arrayBuffer: async () => Promise.reject(new TypeError('body unreadable')) }) });
  await assert.rejects(loadLc0ModelForOrt(modelUrl, options), new RegExp(LC0_MODEL_DOWNLOAD_NOT_APPROVED));
  assert.equal(env.counts.downloads, 0);
  env.responses.set(modelUrl, new Response('abc'));
  assert.equal((await loadLc0ModelForOrt(modelUrl, options)).cacheStatus, 'hit');
  assert.equal(env.counts.downloads, 0);
});

test('a noDownload init skips the prefetch worker and tells the session worker not to download', async (t) => {
  const env = environment(t);
  const searcher = new Bt4WorkerSearcher(config);
  assert.equal(await searcher.init({ noDownload: true }), 'mock-webgpu');
  assert.deepEqual(env.trace, ['search:spawn', 'search:init'], 'nothing to prefetch');
  assert.equal(env.workers[0].messages[0].noModelDownload, true);
  await searcher.dispose();
  const approved = new Bt4WorkerSearcher(config);
  await approved.init();
  assert.equal(env.workers.at(-1).messages[0].noModelDownload, undefined);
  await approved.dispose();
});
