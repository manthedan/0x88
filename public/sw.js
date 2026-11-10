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
//   - /_app/immutable/*: content-hashed build output, cache first;
//   - a few small shell files the HTML links: network first.
//
// Everything lives in one cache per build. src/lib/pwa/register.ts registers
// /sw.js?v=<SvelteKit build version>; a new version is a new script URL, so
// each deploy installs a fresh worker whose activate step deletes every older
// build's cache instead of letting hashed chunks accumulate forever.
const CACHE_PREFIX = '0x88-app-';
const LEGACY_CACHE_PREFIXES = ['0x88-app-shell-', 'lc0-app-shell-'];
const APP_SHELL_FALLBACK = '/app/play/';
const NAVIGATION_TIMEOUT_MS = 3500;
const SHELL_ASSETS = new Set([
  '/app-shell.css',
  '/chessground.base.css',
  '/chessground.brown.css',
  '/chessground.cburnett.css',
  '/favicon.svg',
  '/manifest.webmanifest',
]);
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

/** Caches this worker owns or inherited that are not the current build's. */
function staleCacheNames(names, current = CACHE_NAME) {
  return names.filter((name) => name !== current && (name.startsWith(CACHE_PREFIX) || LEGACY_CACHE_PREFIXES.some((prefix) => name.startsWith(prefix))));
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

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (cacheable(response)) cache.put(request, response.clone()).catch(() => undefined);
  return response;
}

async function cachedNavigation(request, cache) {
  return (await cache.match(request)) ?? (request.mode === 'navigate' ? await cache.match(APP_SHELL_FALLBACK) : undefined);
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
  event.waitUntil(precacheShell().then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys().then((names) => Promise.all(staleCacheNames(names).map((name) => caches.delete(name)))));
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const route = routeFor(event.request, url);
  if (!route) return;
  if (route === 'immutable') {
    event.respondWith(cacheFirst(event.request));
    return;
  }
  event.respondWith(networkFirst(event.request, { waitUntil: (promise) => event.waitUntil(promise) }));
});
