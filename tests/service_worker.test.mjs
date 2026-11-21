import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';

const source = await readFile(resolve(process.cwd(), 'public/sw.js'), 'utf8');
const ORIGIN = 'https://0x88.test';

// public/sw.js is a classic worker script, so it is evaluated as one: in a
// context whose globals are a fake worker scope.
class FakeCache {
  entries = new Map();
  key(request) {
    return new URL(typeof request === 'string' ? request : request.url, `${ORIGIN}/sw.js`).href;
  }
  async match(request, { ignoreSearch } = {}) {
    const key = this.key(request);
    if (!ignoreSearch) return this.entries.get(key)?.clone();
    return [...this.entries].find(([url]) => url.split('?')[0] === key.split('?')[0])?.[1].clone();
  }
  async put(request, response) {
    this.entries.set(this.key(request), response);
  }
  async add(url) {
    const response = await this.fetch(url);
    if (!response.ok) throw new TypeError(`bad status ${response.status}`);
    await this.put(url, response);
  }
}

class FakeCacheStorage {
  stores = new Map();
  constructor(fetch) {
    this.fetch = fetch;
  }
  async open(name) {
    if (!this.stores.has(name)) {
      const cache = new FakeCache();
      cache.fetch = (...args) => this.fetch(...args);
      this.stores.set(name, cache);
    }
    return this.stores.get(name);
  }
  async keys() {
    return [...this.stores.keys()];
  }
  async delete(name) {
    return this.stores.delete(name);
  }
}

function basic(body, { status = 200, contentType = 'text/javascript', redirected = false } = {}) {
  const response = new Response(body, { status, headers: { 'content-type': contentType } });
  Object.defineProperty(response, 'type', { value: 'basic' });
  Object.defineProperty(response, 'redirected', { value: redirected });
  return response;
}

function request(path, { mode = 'cors', method = 'GET', headers = {} } = {}) {
  return { url: new URL(path, ORIGIN).href, method, mode, headers: new Headers(headers) };
}

function loadWorker({ version = '1794286670477', fetch = async () => basic(''), caches: sharedCaches } = {}) {
  const listeners = {};
  const state = { skipped: false, fetches: [] };
  const fetchSpy = async (input, init) => {
    state.fetches.push(typeof input === 'string' ? input : input.url);
    return state.fetch(input, init);
  };
  state.fetch = fetch;
  const caches = sharedCaches ?? new FakeCacheStorage(fetchSpy);
  const self = {
    location: new URL(`${ORIGIN}/sw.js?v=${encodeURIComponent(version)}`),
    addEventListener: (type, listener) => {
      listeners[type] = listener;
    },
    skipWaiting: async () => {
      state.skipped = true;
    },
  };
  const context = vm.createContext({ self, caches, fetch: fetchSpy, URL, Headers, Response, setTimeout, clearTimeout, Promise });
  // Top-level consts are lexical, not context properties; expose the ones under test.
  Object.assign(context, vm.runInContext(`${source}\n;({ CACHE_NAME, NAVIGATION_TIMEOUT_MS });`, context, { filename: 'public/sw.js' }));
  return { context, caches, listeners, state };
}

function fetchEvent(req) {
  const event = {
    request: req,
    response: undefined,
    pending: [],
    respondWith(promise) {
      event.response = promise;
    },
    waitUntil(promise) {
      event.pending.push(promise);
    },
  };
  return event;
}

async function dispatchLifecycle(worker, type) {
  const event = fetchEvent(undefined);
  worker.listeners[type](event);
  await Promise.all(event.pending);
}

test('only navigations, hashed build output and small shell files are intercepted', () => {
  const { context } = loadWorker();
  const route = (path, init) => context.routeFor(request(path, init), new URL(path, ORIGIN));
  assert.equal(route('/app/play/', { mode: 'navigate' }), 'navigate');
  assert.equal(route('/_app/immutable/entry/start.C1MMvhS4.js'), 'immutable');
  assert.equal(route('/_app/immutable/assets/app.Bx1.css'), 'immutable');
  assert.equal(route('/favicon.svg'), 'shell');
  assert.equal(route('/manifest.webmanifest'), 'shell');
  // Global CSS is Vite output now; the old unhashed public/ names are not shell files.
  assert.equal(route('/app-shell.css'), null);
  // Engine artifacts and control-plane pointers are left to the browser.
  for (const path of [
    '/ort/ort-wasm-simd-threaded.jsep.wasm',
    '/ort/ort-wasm-simd-threaded.jsep.mjs',
    '/stockfish/stockfish-19.wasm',
    '/stockfish/stockfish-19-lite.js',
    '/lc0/lc0_encoder.wasm',
    '/reckless/reckless-simd128.nnue',
    '/models/lc0/manifest.json',
    '/channels/stable.json',
    '/releases/2026-11-09.engine-upgrades.1.json',
    '/artifacts/sha256/ab/abcdef',
    '/runtimes/centipawn-tvmjs-webgpu/x/tvmjs.bundle.js',
    '/openings/eco.json',
    '/_app/version.json',
    '/_app/immutable/assets/big.wasm',
  ]) {
    assert.equal(route(path), null, path);
  }
  assert.equal(route('/_app/immutable/entry/start.js', { method: 'POST' }), null);
  assert.equal(route('/_app/immutable/entry/start.js', { headers: { range: 'bytes=0-1' } }), null);
  assert.equal(context.routeFor(request('/x'), new URL('https://assets.0x88.app/_app/immutable/x.js')), null);
});

test('the cache is named after the build version from the script URL', () => {
  assert.equal(loadWorker({ version: '1794286670477' }).context.CACHE_NAME, '0x88-app2-1794286670477');
  assert.equal(loadWorker({ version: '../../etc' }).context.CACHE_NAME, '0x88-app2-unversioned');
  assert.equal(loadWorker({ version: '' }).context.CACHE_NAME, '0x88-app2-unversioned');
});

// '0x88-app-200' is what a worker registered at ?v=200 before the prefix bump left behind: its
// pages link stylesheets this script no longer serves, so reinstalling under ?v=200 must drop it.
test('an update waits for old clients; activation keeps the previous build and removes legacy caches', async () => {
  const worker = loadWorker({ version: '200', fetch: async () => basic(SHELL_HTML, { contentType: 'text/html' }) });
  for (const name of ['0x88-app-shell-v2', 'lc0-app-shell-v1', '0x88-app-100', '0x88-app-200', '0x88-app2-100', 'lc0-browser-models-v1', 'other-cache']) {
    await worker.caches.open(name);
  }
  const oldCache = await worker.caches.open('0x88-app2-100');
  const oldChunk = '/_app/immutable/chunks/old.js';
  await oldCache.put(oldChunk, basic('old build'));
  await dispatchLifecycle(worker, 'install');
  assert.equal(worker.state.skipped, false);
  assert.equal(await (await oldCache.match(oldChunk)).text(), 'old build');
  await dispatchLifecycle(worker, 'activate');
  assert.deepEqual(await worker.caches.keys(), ['0x88-app2-100', 'lc0-browser-models-v1', 'other-cache', '0x88-app2-200', '0x88-sw-state']);
});

const SHELL_HTML = [
  '<link rel="icon" href="/favicon.svg" type="image/svg+xml" />',
  '<link rel="manifest" href="/manifest.webmanifest" />',
  '<link href="../../_app/immutable/entry/start.C1MMvhS4.js" rel="modulepreload">',
  '<link href="../../_app/immutable/nodes/2.Ab.js" rel="modulepreload">',
  '<link href="../../_app/immutable/assets/2.Cd.css" rel="stylesheet">',
  '<link href="../../_app/immutable/assets/huge.wasm" rel="preload">',
  '<link href="https://assets.0x88.app/_app/immutable/x.js" rel="modulepreload">',
  '<link href="/ort/ort.mjs" rel="modulepreload">',
].join('\n');

test('offline navigation and lazy imports survive a visit while the update was waiting', async () => {
  const html = (body) => basic(body, { contentType: 'text/html' });
  const a = loadWorker({ version: 'a', fetch: async () => html('A home') });
  await dispatchLifecycle(a, 'install');
  await dispatchLifecycle(a, 'activate');
  const b = loadWorker({ version: 'b', fetch: async () => html('B home'), caches: a.caches });
  await dispatchLifecycle(b, 'install');
  const chunk = '/_app/immutable/chunks/analysis-b.js';
  a.state.fetch = async ({ url }) => (url.endsWith(chunk) ? basic('B chunk') : html('B analysis'));
  for (const req of [request('/app/analysis/', { mode: 'navigate' }), request(chunk)]) {
    const event = fetchEvent(req);
    a.listeners.fetch(event);
    await event.response;
    await Promise.all(event.pending);
  }
  await dispatchLifecycle(b, 'activate');
  b.state.fetch = async () => {
    throw new TypeError('offline');
  };
  for (const [req, expected] of [
    [request('/app/analysis/?fen=offline', { mode: 'navigate' }), 'B analysis'],
    [request(chunk), 'B chunk'],
  ]) {
    const event = fetchEvent(req);
    b.listeners.fetch(event);
    assert.equal(await (await event.response).text(), expected);
  }
  const c = loadWorker({ version: 'c', fetch: async () => html('C home'), caches: a.caches });
  await dispatchLifecycle(c, 'install');
  await dispatchLifecycle(c, 'activate');
  assert.deepEqual(await a.caches.keys(), ['0x88-sw-state', b.context.CACHE_NAME, c.context.CACHE_NAME]);
});

test('chunks reused from the previous cache survive another activation', async () => {
  const html = (body) => basic(body, { contentType: 'text/html' });
  const a = loadWorker({ version: 'a', fetch: async () => html('A home') });
  await dispatchLifecycle(a, 'install');
  await dispatchLifecycle(a, 'activate');
  const chunk = request('/_app/immutable/chunks/shared.js');
  const page = request('/app/analysis/', { mode: 'navigate' });
  await (await a.caches.open(a.context.CACHE_NAME)).put(chunk, basic('shared chunk'));
  const b = loadWorker({ version: 'b', fetch: async () => html('B analysis'), caches: a.caches });
  await dispatchLifecycle(b, 'install');
  await dispatchLifecycle(b, 'activate');
  for (const [req, expected] of [
    [page, 'B analysis'],
    [chunk, 'shared chunk'],
  ]) {
    const event = fetchEvent(req);
    b.listeners.fetch(event);
    assert.equal(await (await event.response).text(), expected);
    assert.equal(event.pending.length, 1, 'cache writes keep the fetch event alive');
    await Promise.all(event.pending);
  }
  assert.ok(!b.state.fetches.includes(chunk.url));
  const c = loadWorker({ version: 'c', fetch: async () => html('C home'), caches: a.caches });
  await dispatchLifecycle(c, 'install');
  await dispatchLifecycle(c, 'activate');
  assert.ok(!(await a.caches.keys()).includes(a.context.CACHE_NAME));
  c.state.fetch = async () => {
    throw new TypeError('offline');
  };
  for (const [req, expected] of [
    [page, 'B analysis'],
    [chunk, 'shared chunk'],
  ]) {
    const event = fetchEvent(req);
    c.listeners.fetch(event);
    assert.equal(await (await event.response).text(), expected);
    await Promise.all(event.pending);
  }
});

test('rollback retains the active worker’s offline pages and chunks despite cache creation order', async () => {
  const html = (body) => basic(body, { contentType: 'text/html' });
  const a = loadWorker({ version: 'a', fetch: async () => html('A home') });
  await dispatchLifecycle(a, 'install');
  await dispatchLifecycle(a, 'activate');
  const b = loadWorker({ version: 'b', fetch: async () => html('B home'), caches: a.caches });
  await dispatchLifecycle(b, 'install');
  await dispatchLifecycle(b, 'activate');
  const rollback = loadWorker({ version: 'a', fetch: async () => html('A home'), caches: a.caches });
  await dispatchLifecycle(rollback, 'install');
  const chunk = '/_app/immutable/chunks/analysis-a.js';
  const page = request('/app/analysis/', { mode: 'navigate' });
  b.state.fetch = async ({ url }) => (url.endsWith(chunk) ? basic('A chunk') : html('A analysis'));
  for (const req of [page, request(chunk)]) {
    const event = fetchEvent(req);
    b.listeners.fetch(event);
    await event.response;
    await Promise.all(event.pending);
  }
  await dispatchLifecycle(rollback, 'activate');
  const restarted = loadWorker({
    version: 'a',
    fetch: async () => {
      throw new TypeError('offline');
    },
    caches: a.caches,
  });
  for (const [req, expected] of [
    [page, 'A analysis'],
    [request(chunk), 'A chunk'],
  ]) {
    const event = fetchEvent(req);
    restarted.listeners.fetch(event);
    assert.equal(await (await event.response).text(), expected);
  }
});

test('activation cannot remove a later installer’s cache', async () => {
  const html = () => basic(SHELL_HTML, { contentType: 'text/html' });
  const b = loadWorker({ version: 'b', fetch: async () => html() });
  await dispatchLifecycle(b, 'install');
  let finishInstall;
  const c = loadWorker({
    version: 'c',
    fetch: () =>
      new Promise((resolve) => {
        finishInstall = () => resolve(html());
      }),
    caches: b.caches,
  });
  const installing = dispatchLifecycle(c, 'install');
  await Promise.resolve();
  await dispatchLifecycle(b, 'activate');
  finishInstall();
  await installing;
  await dispatchLifecycle(c, 'activate');
  assert.ok(await (await b.caches.open(c.context.CACHE_NAME)).match('/app/play/'));
});

test('the shell page is precached with its build chunks and shell files', async () => {
  assert.deepEqual(Array.from(loadWorker().context.shellSubresourceUrls(SHELL_HTML, `${ORIGIN}/app/play/`)), [
    `${ORIGIN}/favicon.svg`,
    `${ORIGIN}/manifest.webmanifest`,
    `${ORIGIN}/_app/immutable/entry/start.C1MMvhS4.js`,
    `${ORIGIN}/_app/immutable/nodes/2.Ab.js`,
    `${ORIGIN}/_app/immutable/assets/2.Cd.css`,
  ]);

  const worker = loadWorker({
    version: '300',
    fetch: async (input) =>
      new URL(input, `${ORIGIN}/sw.js`).pathname === '/app/play/' ? basic(SHELL_HTML, { contentType: 'text/html; charset=UTF-8' }) : basic('x'),
  });
  await dispatchLifecycle(worker, 'install');
  const cache = worker.caches.stores.get('0x88-app2-300');
  assert.deepEqual([...cache.entries.keys()].sort(), [
    `${ORIGIN}/_app/immutable/assets/2.Cd.css`,
    `${ORIGIN}/_app/immutable/entry/start.C1MMvhS4.js`,
    `${ORIGIN}/_app/immutable/nodes/2.Ab.js`,
    `${ORIGIN}/app/play/`,
    `${ORIGIN}/favicon.svg`,
    `${ORIGIN}/manifest.webmanifest`,
  ]);
});

const failedPrecaches = [
  {
    name: 'network failure',
    fetch: async () => {
      throw new TypeError('offline');
    },
  },
  { name: 'HTTP 503', fetch: async () => basic('unavailable', { status: 503, contentType: 'text/html' }) },
  { name: 'non-HTML response', fetch: async () => basic('not a shell') },
  { name: 'redirected response', fetch: async () => basic(SHELL_HTML, { contentType: 'text/html', redirected: true }) },
  {
    name: 'cache storage failure',
    fetch: async () => basic(SHELL_HTML, { contentType: 'text/html' }),
    failPut: true,
  },
];

for (const failure of failedPrecaches) {
  test(`failed essential precache on update keeps the old worker and offline shell: ${failure.name}`, async () => {
    const old = loadWorker({ version: 'old', fetch: async () => basic(SHELL_HTML, { contentType: 'text/html' }) });
    await dispatchLifecycle(old, 'install');
    await dispatchLifecycle(old, 'activate');
    const oldCache = await old.caches.open(old.context.CACHE_NAME);
    const update = loadWorker({ version: 'new', fetch: failure.fetch, caches: old.caches });
    if (failure.failPut) {
      const newCache = await update.caches.open(update.context.CACHE_NAME);
      newCache.put = async () => {
        throw new Error('storage full');
      };
    }
    await assert.rejects(dispatchLifecycle(update, 'install'));
    assert.equal(update.state.skipped, false, 'a rejected install cannot replace the active worker');
    assert.equal(old.caches.stores.get(old.context.CACHE_NAME), oldCache);
    old.state.fetch = async () => {
      throw new TypeError('offline');
    };
    const response = await old.context.networkFirst(request('/app/analysis/', { mode: 'navigate' }));
    assert.equal(await response.text(), SHELL_HTML);
  });

  test(`first install rejects an unusable essential precache: ${failure.name}`, async () => {
    const worker = loadWorker({ fetch: failure.fetch });
    if (failure.failPut) {
      const cache = await worker.caches.open(worker.context.CACHE_NAME);
      cache.put = async () => {
        throw new Error('storage full');
      };
    }
    await assert.rejects(dispatchLifecycle(worker, 'install'));
    assert.equal(worker.state.skipped, false);
  });
}

test('linked chunk precaching is best-effort once the essential navigation shell is stored', async () => {
  const worker = loadWorker({
    fetch: async (input) => {
      if (input === '/app/play/') return basic(SHELL_HTML, { contentType: 'text/html' });
      throw new TypeError('chunk unavailable');
    },
  });
  await dispatchLifecycle(worker, 'install');
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  assert.equal(await (await cache.match('/app/play/')).text(), SHELL_HTML);
});

test('heavy and cross-origin requests are not intercepted', () => {
  const worker = loadWorker();
  for (const path of ['/ort/ort-wasm-simd-threaded.jsep.wasm', '/stockfish/stockfish-19.wasm', 'https://assets.0x88.app/models/x.onnx']) {
    const event = fetchEvent(request(path));
    worker.listeners.fetch(event);
    assert.equal(event.response, undefined, path);
  }
});

test('hashed build output is served cache first', async () => {
  const worker = loadWorker({ fetch: async () => basic('chunk') });
  for (let i = 0; i < 2; i++) {
    const event = fetchEvent(request('/_app/immutable/chunks/a.js'));
    worker.listeners.fetch(event);
    assert.equal(await (await event.response).text(), 'chunk');
    assert.equal(event.pending.length, i === 0 ? 1 : 0);
    await Promise.all(event.pending);
  }
  assert.deepEqual(worker.state.fetches, [`${ORIGIN}/_app/immutable/chunks/a.js`]);
});

test('a slow navigation falls back to the cached page after the timeout', async () => {
  let release;
  const worker = loadWorker({
    fetch: () =>
      new Promise((done) => {
        release = () => done(basic('fresh', { contentType: 'text/html' }));
      }),
  });
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  await cache.put(`${ORIGIN}/app/analysis/`, basic('cached', { contentType: 'text/html' }));
  const background = [];
  const response = await worker.context.networkFirst(request('/app/analysis/', { mode: 'navigate' }), {
    timeoutMs: 20,
    waitUntil: (promise) => background.push(promise),
  });
  assert.equal(await response.text(), 'cached');
  // The late network answer still refreshes the cache.
  release();
  await Promise.all(background);
  assert.equal(await (await cache.match(`${ORIGIN}/app/analysis/`)).text(), 'fresh');
});

for (const [path, mode, contentType] of [
  ['/app/analysis/', 'navigate', 'text/html'],
  ['/manifest.webmanifest', 'cors', 'application/manifest+json'],
]) {
  test(`network response headers return before a slow body is cached: ${contentType}`, async (t) => {
    let stream;
    const body = new ReadableStream({
      start(controller) {
        stream = controller;
        controller.enqueue(new TextEncoder().encode('fresh'));
      },
    });
    let closed = false;
    t.after(() => {
      if (!closed) stream.close();
    });
    const fresh = basic(body, { contentType });
    const worker = loadWorker({ fetch: async () => fresh });
    const cache = await worker.caches.open(worker.context.CACHE_NAME);
    await cache.put(path, basic('cached', { contentType }));
    const put = cache.put.bind(cache);
    cache.put = async (req, response) => put(req, basic(await response.text(), { contentType }));
    const event = fetchEvent(request(path, { mode }));
    const response = await Promise.race([
      worker.context.networkFirst(event.request, { timeoutMs: 20, waitUntil: (promise) => event.waitUntil(promise) }),
      new Promise((done) => setTimeout(() => done('still waiting for the body'), 100)),
    ]);
    assert.equal(response, fresh, 'received headers must win the navigation timeout even while the body is streaming');
    assert.equal(await (await cache.match(path)).text(), 'cached');
    assert.equal(event.pending.length, 1, 'the cache write must extend the fetch event lifetime');
    stream.close();
    closed = true;
    await Promise.all(event.pending);
    assert.equal(await (await cache.match(path)).text(), 'fresh');
  });
}

test('fetch events keep failed cache writes alive without rejecting the network response', async () => {
  const worker = loadWorker({ fetch: async () => basic('fresh', { contentType: 'text/html' }) });
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  cache.put = async () => {
    throw new Error('storage full');
  };
  const event = fetchEvent(request('/app/play/', { mode: 'navigate' }));
  worker.listeners.fetch(event);
  assert.equal(await (await event.response).text(), 'fresh');
  assert.equal(event.pending.length, 1);
  await Promise.all(event.pending);
});

test('a slow navigation with nothing cached keeps waiting for the network', async () => {
  const worker = loadWorker({ fetch: () => new Promise((done) => setTimeout(() => done(basic('late', { contentType: 'text/html' })), 60)) });
  const response = await worker.context.networkFirst(request('/docs/', { mode: 'navigate' }), { timeoutMs: 10 });
  assert.equal(await response.text(), 'late');
});

test('offline navigation prefers an exact query match to an earlier page at the same path', async () => {
  const worker = loadWorker({
    fetch: async () => {
      throw new TypeError('offline');
    },
  });
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  await cache.put('/app/analysis/?fen=first', basic('older page', { contentType: 'text/html' }));
  await cache.put('/app/analysis/?fen=second', basic('exact page', { contentType: 'text/html' }));
  const event = fetchEvent(request('/app/analysis/?fen=second', { mode: 'navigate' }));
  worker.listeners.fetch(event);
  assert.equal(await (await event.response).text(), 'exact page');
});

test('an offline navigation falls back to the page, then to the app shell', async () => {
  const worker = loadWorker({
    fetch: async () => {
      throw new TypeError('offline');
    },
  });
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  await cache.put(`${ORIGIN}/app/play/`, basic('shell', { contentType: 'text/html' }));
  await cache.put(`${ORIGIN}/app/analysis/?fen=old`, basic('analysis', { contentType: 'text/html' }));
  const analysis = await worker.context.networkFirst(request('/app/analysis/?fen=new', { mode: 'navigate' }));
  assert.equal(await analysis.text(), 'analysis');
  const response = await worker.context.networkFirst(request('/app/arena/', { mode: 'navigate' }));
  assert.equal(await response.text(), 'shell');
  await assert.rejects(worker.context.networkFirst(request('/favicon.svg')), /offline/);
});

test('navigation timeout is a few seconds', () => {
  const { context } = loadWorker();
  assert.ok(context.NAVIGATION_TIMEOUT_MS >= 3000 && context.NAVIGATION_TIMEOUT_MS <= 4000);
});

test('only HTML navigations and plain 200 responses are stored', async () => {
  let next;
  const worker = loadWorker({ fetch: async () => next });
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  next = basic('tarball', { contentType: 'application/gzip' });
  await worker.context.networkFirst(request('/stockfish/stockfish-19.0.1-corresponding-source.tar.gz', { mode: 'navigate' }));
  next = basic('page', { contentType: 'text/html', redirected: true });
  await worker.context.networkFirst(request('/app/play/', { mode: 'navigate' }));
  next = basic('missing', { status: 404, contentType: 'text/html' });
  await worker.context.networkFirst(request('/nope/', { mode: 'navigate' }));
  assert.equal(cache.entries.size, 0);
  next = basic('page', { contentType: 'text/html; charset=UTF-8' });
  await worker.context.networkFirst(request('/docs/', { mode: 'navigate' }));
  assert.deepEqual([...cache.entries.keys()], [`${ORIGIN}/docs/`]);
});
