// App-shell service worker.
//
// Scope of what this worker touches is an ALLOWLIST, because almost every
// other same-origin GET on this site is an engine artifact: /ort/*.wasm is
// 13-24 MB per file, /stockfish/stockfish-19.wasm is ~99 MB (proxied from the
// asset host), and /lc0/, /reckless/, /models/, /runtimes/, /channels/ and
// /releases/ are runtimes, weights and control-plane pointers with their own
// cache policy (HTTP cache headers, and the verified model cache in
// src/lc0/modelCache.ts, which competes with this one for quota). Those
// requests are not intercepted at all.
//
// Handled:
//   - navigations: network first, falling back to the cached page (or the
//     /app/play/ shell) after NAVIGATION_TIMEOUT_MS or on network failure;
//   - /_app/immutable/*: content-hashed build output (JS and CSS), cache first;
//   - the favicon and web manifest the HTML links: network first.
//
// src/lib/pwa/register.ts registers /sw.js?v=<SvelteKit build version>.
// The browser keeps the update waiting until the active worker controls no
// clients. Activation retains that worker's cache for offline fallback.
//
// The prefix changes whenever cached pages from earlier builds would break
// under this script. A worker registered at an old ?v= URL can reinstall
// with new code under its old version; a new prefix still gives it a fresh
// cache, and activate drops the old one. '0x88-app-' caches hold pages that
// link the pre-Vite global stylesheets (/app-shell.css etc.), which this
// worker neither routes nor precaches. Those files stay in public/ until
// 2026-12-18 for pages an older worker still serves.
const CACHE_PREFIX = '0x88-app2-';
const LEGACY_CACHE_PREFIXES = ['0x88-app-', 'lc0-app-shell-'];
const APP_SHELL_FALLBACK = '/app/play/';
const ACTIVATION_RECORD = '/__sw/previous-cache';
const STATE_CACHE = '0x88-sw-state';
const LAST_ACTIVE_RECORD = '/__sw/active-cache';
const NAVIGATION_TIMEOUT_MS = 3500;
// Stylesheets are not here: they are Vite build output under /_app/immutable/.
const SHELL_ASSETS = new Set(['/favicon.svg', '/manifest.webmanifest']);
// Never cache these even under an allowlisted prefix.
const HEAVY_EXTENSIONS = ['.wasm', '.data', '.nnue', '.onnx', '.bin', '.pack', '.br', '.gz', '.tar'];

function buildVersion(scriptUrl) {
  const raw = new URL(scriptUrl).searchParams.get('v') ?? '';
  return /^[A-Za-z0-9._-]{1,64}$/.test(raw) ? raw : 'unversioned';
}

const CACHE_NAME = `${CACHE_PREFIX}${buildVersion(self.location.href)}`;

function isHeavy(pathname) {
  return HEAVY_EXTENSIONS.some((extension) => pathname.endsWith(extension));
}

/** How a request is served: 'navigate', 'immutable', 'shell', or null to leave it to the browser. */
function routeFor(request, url, origin = self.location.origin) {
  if (request.method !== 'GET') return null;
  if (url.origin !== origin) return null;
  if (request.headers.has('range')) return null;
  if (request.mode === 'navigate') return 'navigate';
  if (isHeavy(url.pathname)) return null;
  if (url.pathname.startsWith('/_app/immutable/')) return 'immutable';
  if (SHELL_ASSETS.has(url.pathname)) return 'shell';
  return null;
}

async function activateCache() {
  const current = await caches.open(CACHE_NAME);
  const names = await caches.keys();
  // CacheStorage.keys() is in creation order. A later installer may already
  // have opened its own cache and must be left alone.
  const older = names.slice(0, names.indexOf(CACHE_NAME));
  const state = await caches.open(STATE_CACHE);
  let previous = await (await state.match(LAST_ACTIVE_RECORD))?.text();
  if (previous === CACHE_NAME) {
    previous = await (await current.match(ACTIVATION_RECORD))?.text();
  } else if (!previous) {
    const builds = older.filter((name) => name.startsWith(CACHE_PREFIX));
    previous = builds[0];
    for (const name of builds.reverse()) {
      if (await (await caches.open(name)).match(ACTIVATION_RECORD)) {
        previous = name;
        break;
      }
    }
  }
  await current.put(ACTIVATION_RECORD, new Response(previous ?? ''));
  await state.put(LAST_ACTIVE_RECORD, new Response(CACHE_NAME));
  await Promise.all(older.filter((name) => name !== previous && (name.startsWith(CACHE_PREFIX) || LEGACY_CACHE_PREFIXES.some((prefix) => name.startsWith(prefix))))
    .map((name) => caches.delete(name)));
}

let previousCachePromise;
function previousCache(cache) {
  return previousCachePromise ??= cache.match(ACTIVATION_RECORD).then(async (response) => {
    const name = await response?.text();
    if (name) return caches.open(name);
  });
}

/** Same-origin subresources of a page worth precaching with it: its build chunks and shell files. */
function shellSubresourceUrls(html, pageUrl) {
  const page = new URL(pageUrl);
  const urls = new Set();
  for (const match of html.matchAll(/<link\b[^>]*\bhref="([^"]+)"[^>]*>/g)) {
    let url;
    try {
      url = new URL(match[1], page);
    } catch {
      continue;
    }
    if (url.origin !== page.origin) continue;
    if (url.pathname.startsWith('/_app/immutable/') ? !isHeavy(url.pathname) : SHELL_ASSETS.has(url.pathname)) urls.add(url.href);
  }
  return [...urls];
}

// A redirected response cannot answer a navigation (browsers reject it from
// respondWith), so one is never stored.
function cacheable(response) {
  return !!response && response.status === 200 && response.type === 'basic' && !response.redirected;
}

function isHtml(response) {
  return (response.headers.get('content-type') ?? '').toLowerCase().startsWith('text/html');
}

async function precacheShell() {
  const cache = await caches.open(CACHE_NAME);
  const response = await fetch(APP_SHELL_FALLBACK, { cache: 'no-cache' });
  // The fallback navigation HTML is essential: installing without it would
  // replace the old worker and let activate delete its working offline shell.
  if (!cacheable(response) || !isHtml(response)) throw new Error('Offline navigation shell could not be precached');
  const html = await response.clone().text();
  await cache.put(APP_SHELL_FALLBACK, response);
  // Linked chunks/styles are best-effort. They improve offline hydration, but
  // a failed chunk fetch must not discard an otherwise usable HTML fallback.
  await Promise.all(shellSubresourceUrls(html, new URL(APP_SHELL_FALLBACK, self.location.origin).href).map((url) => cache.add(url).catch(() => undefined)));
}

async function cacheFirst(request, waitUntil) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const previous = await (await previousCache(cache))?.match(request);
  if (previous) {
    waitUntil(cache.put(request, previous.clone()).catch(() => undefined));
    return previous;
  }
  const response = await fetch(request);
  if (cacheable(response)) waitUntil(cache.put(request, response.clone()).catch(() => undefined));
  return response;
}

async function cachedNavigation(request, cache) {
  const exact = await cache.match(request);
  if (exact || request.mode !== 'navigate') return exact;
  const options = { ignoreSearch: true };
  const samePath = await cache.match(request, options);
  if (samePath) return samePath;
  const previous = await previousCache(cache);
  return (await previous?.match(request)) ??
    (await previous?.match(request, options)) ??
    (await cache.match(APP_SHELL_FALLBACK));
}

/**
 * Network first. A network answer that has not arrived within timeoutMs is
 * raced against the cache: a cached copy is served and the network response
 * still refreshes the cache in the background (via waitUntil); with nothing
 * cached, the request keeps waiting for the network rather than failing.
 */
async function networkFirst(request, { timeoutMs = NAVIGATION_TIMEOUT_MS, waitUntil = () => undefined } = {}) {
  const cache = await caches.open(CACHE_NAME);
  const network = fetch(request).then((response) => {
    // Only pages are kept for navigations: a navigation to a source tarball or
    // another download must not land in the shell cache.
    if (cacheable(response) && (request.mode !== 'navigate' || isHtml(response))) {
      // Hand headers/body to the browser immediately; the timeout races the
      // response headers, never consumption of a slow/streaming cache clone.
      waitUntil(cache.put(request, response.clone()).catch(() => undefined));
    }
    return response;
  });
  let timer;
  const timedOut = new Promise((resolve) => {
    timer = setTimeout(() => resolve('timeout'), timeoutMs);
  });
  try {
    const first = await Promise.race([network, timedOut]);
    if (first !== 'timeout') return first;
  } catch {
    const cached = await cachedNavigation(request, cache);
    if (cached) return cached;
    return network;
  } finally {
    clearTimeout(timer);
  }
  const cached = await cachedNavigation(request, cache);
  if (!cached) return network;
  waitUntil(network.catch(() => undefined));
  return cached;
}

self.addEventListener('install', (event) => {
  event.waitUntil(precacheShell());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(activateCache());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const route = routeFor(event.request, url);
  if (!route) return;
  const waitUntil = (promise) => event.waitUntil(promise);
  event.respondWith(route === 'immutable' ? cacheFirst(event.request, waitUntil) : networkFirst(event.request, { waitUntil }));
});
