const SHELL_CACHE = '0x88-app-shell-v1';
const APP_SHELL_FALLBACK = '/app/play/';

self.addEventListener('install', (event) => {
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
        names.filter((name) => (
          (name.startsWith('0x88-app-shell-') && name !== SHELL_CACHE)
          || name.startsWith('lc0-app-shell-')
        )).map((name) => caches.delete(name)),
      ))
      .then(() => self.clients.claim()),
  );
});

function shouldHandle(request, url) {
  if (request.method !== 'GET') return false;
  if (url.origin !== self.location.origin) return false;
  if (request.headers.has('range')) return false;
  if (url.pathname.includes('/artifacts/sha256/')) return false;
  if (url.pathname.endsWith('.onnx') || url.pathname.endsWith('.onnx.data')) return false;
  if (url.pathname.endsWith('.bin') || url.pathname.endsWith('.pack')) return false;
  return true;
}

function cacheable(response) {
  return !!response && response.status === 200 && response.type === 'basic';
}

function isImmutableAsset(url) {
  return url.pathname.startsWith('/_app/immutable/') || url.pathname.startsWith('/assets/');
}

async function cacheFirst(request) {
  const cache = await caches.open(SHELL_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (cacheable(response)) cache.put(request, response.clone()).catch(() => undefined);
  return response;
}

async function networkFirst(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const response = await fetch(request);
    if (cacheable(response)) cache.put(request, response.clone()).catch(() => undefined);
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) return cached;
    if (request.mode === 'navigate') {
      const fallback = await cache.match(APP_SHELL_FALLBACK);
      if (fallback) return fallback;
    }
    throw error;
  }
}

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (!shouldHandle(event.request, url)) return;
  event.respondWith(isImmutableAsset(url) ? cacheFirst(event.request) : networkFirst(event.request));
});
