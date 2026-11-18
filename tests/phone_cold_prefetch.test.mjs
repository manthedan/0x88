import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BT4_NET, Bt4WorkerSearcher } from '../src/lc0/bt4Engine.ts';
import { prefetchLargeNetToCache } from '../src/lc0/largeNetDownload.ts';
import { loadLc0ModelForOrt, prefetchLc0ModelToCache } from '../src/lc0/modelCache.ts';

// Phone cold loads: the net is downloaded, verified and cached by a throwaway
// worker that is terminated before the session worker starts, so the session
// load is a cached load (cold-load-copies analysis, reduction #1).

const modelUrl = 'https://example.test/models/lc0/prefetch.onnx';
const sha256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'; // sha256('abc')
const config = { ...BT4_NET, modelUrl };
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
workerScope.postMessage = (data) => toPrefetchOwner(data);
globalThis.self = workerScope;
await import('../src/lc0/modelPrefetchWorker.ts');

/** Cache Storage, the app manifest and the network; `put` can be made to fail. */
function environment(t, { phone = true, putFails = false, holdPrefetch = false } = {}) {
  const trace = [];
  const responses = new Map();
  const counts = { downloads: 0 };
  replaceGlobal(t, 'location', new URL(`https://example.test/analysis?memoryClass=${phone ? 'phone' : 'desktop'}`));
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
    if (href.endsWith('/models/lc0/manifest.json')) return Response.json({ models: [{ file: 'prefetch.onnx', url: modelUrl, bytes: 3, sha256 }] });
    assert.equal(href, modelUrl);
    counts.downloads++;
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
        this.terminated = false;
        workers.push(this);
        trace.push(`${this.kind}:spawn`);
      }
      postMessage(message) {
        this.messages.push(message);
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
});

test('phone warm load: a cached net spawns no prefetch worker', async (t) => {
  const env = environment(t);
  const inspected = [];
  const result = await prefetchLargeNetToCache(
    config,
    { signal: new AbortController().signal, load: {} },
    {
      placement: () => assert.fail('BT4 has no device-specific rewrite'),
      inspect: async (url) => (inspected.push(url), { url, cached: true }),
      spawn: () => assert.fail('no worker for a cached net'),
      reclaimGapMs: () => 0,
    },
  );
  assert.equal(result, 'cached');
  assert.deepEqual(inspected, [modelUrl]);
  assert.deepEqual(env.trace, []);
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

test('a failed cache.put falls back to the in-worker load', async (t) => {
  const env = environment(t, { putFails: true });
  const searcher = new Bt4WorkerSearcher(config);
  t.mock.method(console, 'warn', () => undefined);
  assert.equal(await searcher.init(), 'mock-webgpu');
  assert.deepEqual(env.trace, ['prefetch:spawn', 'download', 'prefetch:terminate', 'search:spawn', 'search:init']);
  assert.equal(env.responses.size, 0);
  // The session worker then loads exactly as before the prefetch existed.
  const load = await loadLc0ModelForOrt(modelUrl, { cache: true, signal: AbortSignal.timeout(5000) });
  assert.equal(load.mode, 'memory');
  assert.equal(env.counts.downloads, 2);
  await searcher.dispose();
});

test('prefetch reports put failure and refuses to download what Cache Storage cannot keep', async (t) => {
  const env = environment(t, { putFails: true });
  const manifestEntry = { file: 'prefetch.onnx', url: modelUrl, bytes: 3, sha256 };
  assert.equal(await prefetchLc0ModelToCache(modelUrl, { manifestEntry, channelUrl: null }), false);
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
  worker.postMessage = () => queueMicrotask(() => worker.dispatchEvent(new MessageEvent('message', { data: { type: 'done', cached: true } })));
  worker.terminate = () => {
    terminatedAt = performance.now();
  };
  const result = await prefetchLargeNetToCache(
    config,
    { signal: new AbortController().signal, load: {} },
    { placement: () => assert.fail('unused'), inspect: async (url) => ({ url, cached: false }), spawn: () => worker, reclaimGapMs: () => 40 },
  );
  assert.equal(result, 'prefetched');
  assert.ok(performance.now() - terminatedAt >= 35, 'resolved only after the gap');
});
