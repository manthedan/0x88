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
  async match(request) {
    return this.entries.get(this.key(request))?.clone();
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

function loadWorker({ version = '1794286670477', fetch = async () => basic('') } = {}) {
  const listeners = {};
  const state = { skipped: false, fetches: [] };
  const fetchSpy = async (input, init) => {
    state.fetches.push(typeof input === 'string' ? input : input.url);
    return state.fetch(input, init);
  };
  state.fetch = fetch;
  const caches = new FakeCacheStorage(fetchSpy);
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
  assert.equal(route('/app-shell.css'), 'shell');
  assert.equal(route('/manifest.webmanifest'), 'shell');
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
  assert.equal(loadWorker({ version: '1794286670477' }).context.CACHE_NAME, '0x88-app-1794286670477');
  assert.equal(loadWorker({ version: '../../etc' }).context.CACHE_NAME, '0x88-app-unversioned');
  assert.equal(loadWorker({ version: '' }).context.CACHE_NAME, '0x88-app-unversioned');
});

test('activate deletes earlier builds and the legacy shell caches, never the model cache', async () => {
  const worker = loadWorker({ version: '200' });
  for (const name of ['0x88-app-shell-v2', 'lc0-app-shell-v1', '0x88-app-100', '0x88-app-200', 'lc0-browser-models-v1', 'other-cache']) {
    await worker.caches.open(name);
  }
  await dispatchLifecycle(worker, 'activate');
  assert.deepEqual(await worker.caches.keys(), ['0x88-app-200', 'lc0-browser-models-v1', 'other-cache']);
});

const SHELL_HTML = [
  '<link rel="stylesheet" href="/chessground.base.css" />',
  '<link rel="stylesheet" href="/app-shell.css" />',
  '<link rel="manifest" href="/manifest.webmanifest" />',
  '<link href="../../_app/immutable/entry/start.C1MMvhS4.js" rel="modulepreload">',
  '<link href="../../_app/immutable/nodes/2.Ab.js" rel="modulepreload">',
  '<link href="../../_app/immutable/assets/2.Cd.css" rel="stylesheet">',
  '<link href="../../_app/immutable/assets/huge.wasm" rel="preload">',
  '<link href="https://assets.0x88.app/_app/immutable/x.js" rel="modulepreload">',
  '<link href="/ort/ort.mjs" rel="modulepreload">',
].join('\n');

test('the shell page is precached with its build chunks and shell files', async () => {
  assert.deepEqual(Array.from(loadWorker().context.shellSubresourceUrls(SHELL_HTML, `${ORIGIN}/app/play/`)), [
    `${ORIGIN}/chessground.base.css`,
    `${ORIGIN}/app-shell.css`,
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
  assert.equal(worker.state.skipped, true);
  const cache = worker.caches.stores.get('0x88-app-300');
  assert.deepEqual([...cache.entries.keys()].sort(), [
    `${ORIGIN}/_app/immutable/assets/2.Cd.css`,
    `${ORIGIN}/_app/immutable/entry/start.C1MMvhS4.js`,
    `${ORIGIN}/_app/immutable/nodes/2.Ab.js`,
    `${ORIGIN}/app-shell.css`,
    `${ORIGIN}/app/play/`,
    `${ORIGIN}/chessground.base.css`,
    `${ORIGIN}/manifest.webmanifest`,
  ]);
});

test('install still completes when the network is unavailable', async () => {
  const worker = loadWorker({
    fetch: async () => {
      throw new TypeError('offline');
    },
  });
  await dispatchLifecycle(worker, 'install');
  assert.equal(worker.state.skipped, true);
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

test('a slow navigation with nothing cached keeps waiting for the network', async () => {
  const worker = loadWorker({ fetch: () => new Promise((done) => setTimeout(() => done(basic('late', { contentType: 'text/html' })), 60)) });
  const response = await worker.context.networkFirst(request('/docs/', { mode: 'navigate' }), { timeoutMs: 10 });
  assert.equal(await response.text(), 'late');
});

test('an offline navigation falls back to the page, then to the app shell', async () => {
  const worker = loadWorker({
    fetch: async () => {
      throw new TypeError('offline');
    },
  });
  const cache = await worker.caches.open(worker.context.CACHE_NAME);
  await cache.put(`${ORIGIN}/app/play/`, basic('shell', { contentType: 'text/html' }));
  const response = await worker.context.networkFirst(request('/app/arena/', { mode: 'navigate' }));
  assert.equal(await response.text(), 'shell');
  await assert.rejects(worker.context.networkFirst(request('/app-shell.css')), /offline/);
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
