/*
 * LC0 browser app-shell service worker.
 *
 * Provides an offline-ish shell using runtime caching (stale-while-revalidate)
 * so the page, its hashed JS/CSS bundles, the ORT WASM runtime, and the small
 * fixture JSON load without the network after a first online visit.
 *
 * The large ONNX model files are deliberately left to the page's own
 * Cache Storage path (src/lc0/modelCache.ts, bucket "lc0-browser-models-v1"),
 * which validates them by sha256. This SW never caches *.onnx so the two
 * caches do not duplicate 40-80 MB blobs or fight over invalidation.
 */
const SHELL_CACHE = 'lc0-app-shell-v1';
const APP_SHELL_FALLBACK = '/lc0-policy-only.html';

self.addEventListener('install', (event) => {
  // Warm the navigation entry; other assets are cached lazily as requested.
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => cache.add(APP_SHELL_FALLBACK).catch(() => undefined))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(
        names.filter((name) => name.startsWith('lc0-app-shell-') && name !== SHELL_CACHE)
          .map((name) => caches.delete(name)),
      ))
      .then(() => self.clients.claim()),
  );
});

function shouldHandle(request, url) {
  if (request.method !== 'GET') return false;
  if (url.origin !== self.location.origin) return false;
  // Range requests (e.g. partial media) and the large sha256-validated models
  // are owned elsewhere; never put them in the shell cache.
  if (request.headers.has('range')) return false;
  if (url.pathname.endsWith('.onnx') || url.pathname.endsWith('.onnx.data')) return false;
  return true;
}

function cacheable(response) {
  // Only cache complete, same-origin OK responses; skip 206/opaque/errors.
  return !!response && response.status === 200 && response.type === 'basic';
}

// Vite emits content-hashed, immutable files under /assets/ (JS bundles and the
// large ORT WASM). Cache-first means they are fetched at most once and never
// re-downloaded in the background, unlike stale-while-revalidate.
function isImmutableAsset(url) {
  return url.pathname.startsWith('/assets/');
}

async function cacheFirst(request) {
  const cache = await caches.open(SHELL_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (cacheable(response)) cache.put(request, response.clone()).catch(() => undefined);
  return response;
}

// Stale-while-revalidate for the navigable HTML and small mutable files (css,
// fixture JSON) so a new deploy propagates after one load while staying offline-
// capable. Falls back to the cached app shell for navigations when offline.
async function staleWhileRevalidate(request) {
  const cache = await caches.open(SHELL_CACHE);
  const cached = await cache.match(request);
  const network = fetch(request)
    .then((response) => {
      if (cacheable(response)) cache.put(request, response.clone()).catch(() => undefined);
      return response;
    })
    .catch(() => undefined);
  const result = cached || (await network);
  if (result) return result;
  if (request.mode === 'navigate') {
    const fallback = await cache.match(APP_SHELL_FALLBACK);
    if (fallback) return fallback;
  }
  return fetch(request);
}

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (!shouldHandle(event.request, url)) return;
  event.respondWith(isImmutableAsset(url) ? cacheFirst(event.request) : staleWhileRevalidate(event.request));
});
