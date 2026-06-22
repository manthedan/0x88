const DEFAULT_APP_ORIGIN = 'https://0x88.app';
const IMMUTABLE_CACHE_CONTROL = 'public, max-age=31536000, immutable';
const EXPOSED_HEADERS = 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, Content-Range, Accept-Ranges';

function artifactHeaders(env, extra = {}) {
  const headers = new Headers(extra);
  headers.set('Access-Control-Allow-Origin', '*');
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  headers.set('Timing-Allow-Origin', env.TIMING_ALLOW_ORIGIN || DEFAULT_APP_ORIGIN);
  headers.set('Access-Control-Expose-Headers', EXPOSED_HEADERS);
  headers.set('Cache-Control', headers.get('Cache-Control') || 'no-store');
  return headers;
}

function withCacheStatus(response, status) {
  const headers = new Headers(response.headers);
  headers.set('Cache-Status', `lc0-artifact-worker; ${status}`);
  headers.set('Age', status === 'hit' ? headers.get('Age') || '1' : '0');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function immutableCache() {
  return globalThis.caches?.default;
}

function metadataCacheRequest(request) {
  const url = new URL(request.url);
  url.search = '__lc0_artifact_head_metadata=v2';
  return new Request(url, { method: 'GET' });
}

function fullBodyCacheRequest(request) {
  const url = new URL(request.url);
  url.search = '';
  return new Request(url, { method: 'GET' });
}

function parseRange(raw, size) {
  if (!raw) return undefined;
  const match = raw.match(/^bytes=(\d*)-(\d*)$/);
  if (!match || (!match[1] && !match[2])) return 'invalid';
  if (!match[1]) {
    const suffixLength = Number(match[2]);
    if (!Number.isSafeInteger(suffixLength) || suffixLength <= 0) return 'invalid';
    const start = Math.max(size - suffixLength, 0);
    const end = size - 1;
    return { start, end, length: end - start + 1 };
  }
  const start = Number(match[1]);
  const end = match[2] ? Number(match[2]) : size - 1;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || start >= size) return 'invalid';
  return { start, end: Math.min(end, size - 1), length: Math.min(end, size - 1) - start + 1 };
}

function keyFromRequest(request) {
  const url = new URL(request.url);
  // R2 publish keys are derived from manifest artifact URLs after encodeURIComponent(file),
  // so keep the URL path's percent-encoded form for object lookup.
  const key = url.pathname.replace(/^\/+/, '');
  return key.startsWith('artifacts/sha256/') ? key : undefined;
}

function objectHeaders(object, env, range) {
  const contentType = object.httpMetadata?.contentType || 'application/octet-stream';
  const cacheControl = object.httpMetadata?.cacheControl || IMMUTABLE_CACHE_CONTROL;
  const headers = artifactHeaders(env, {
    'Content-Type': contentType,
    'Cache-Control': cacheControl,
    ETag: object.httpEtag || object.etag || '',
  });
  headers.set('Accept-Ranges', 'bytes');
  headers.set('X-Artifact-Content-Length', String(object.size));
  if (range) {
    headers.set('Content-Length', String(range.length));
    headers.set('Content-Range', `bytes ${range.start}-${range.end}/${object.size}`);
  } else {
    headers.set('Content-Length', String(object.size));
  }
  return headers;
}

function scheduleCachePut(cache, request, response, ctx) {
  const operation = cache.put(request, response.clone()).catch(() => undefined);
  if (ctx?.waitUntil) ctx.waitUntil(operation);
  else void operation;
}

async function cachedHeadResponse(request, env, head, range, ctx) {
  const cache = immutableCache();
  const cacheRequest = metadataCacheRequest(request);
  if (!range && cache) {
    const cached = await cache.match(cacheRequest);
    if (cached) return withCacheStatus(cached, 'hit');
  }
  const response = new Response(null, { status: range ? 206 : 200, headers: objectHeaders(head, env, range) });
  if (!range && cache) scheduleCachePut(cache, cacheRequest, response, ctx);
  return withCacheStatus(response, cache ? 'miss' : 'fwd');
}

async function cachedFullBodyResponse(request, env, head, ctx) {
  const cache = immutableCache();
  const cacheRequest = fullBodyCacheRequest(request);
  if (cache) {
    const cached = await cache.match(cacheRequest);
    if (cached) return withCacheStatus(cached, 'hit');
  }
  const object = await env.ARTIFACTS.get(keyFromRequest(request));
  if (!object) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });
  const response = new Response(object.body, { status: 200, headers: objectHeaders(head, env) });
  if (cache) scheduleCachePut(cache, cacheRequest, response, ctx);
  return withCacheStatus(response, cache ? 'miss' : 'fwd');
}

async function cachedNoRangeResponse(request) {
  const cache = immutableCache();
  if (!cache || request.headers.get('Range')) return undefined;
  if (request.method === 'HEAD') {
    const cached = await cache.match(metadataCacheRequest(request));
    return cached ? withCacheStatus(cached, 'hit') : undefined;
  }
  if (request.method === 'GET') {
    const cached = await cache.match(fullBodyCacheRequest(request));
    return cached ? withCacheStatus(cached, 'hit') : undefined;
  }
  return undefined;
}

async function rangeResponse(request, env, head, range) {
  // Do not slice a cached full-body response in Worker memory. Some artifacts are hundreds
  // of MiB, so range requests must stay bounded and use R2's range reader directly.
  const object = await env.ARTIFACTS.get(keyFromRequest(request), { range: { offset: range.start, length: range.length } });
  if (!object) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });
  return withCacheStatus(new Response(object.body, { status: 206, headers: objectHeaders(head, env, range) }), 'fwd');
}

export async function handleArtifactRequest(request, env, ctx) {
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: artifactHeaders(env, {
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Type',
        'Access-Control-Max-Age': '86400',
      }),
    });
  }
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response('Method not allowed', { status: 405, headers: artifactHeaders(env, { Allow: 'GET, HEAD, OPTIONS' }) });
  }

  const key = keyFromRequest(request);
  if (!key) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });

  const cached = await cachedNoRangeResponse(request);
  if (cached) return cached;

  const head = await env.ARTIFACTS.head(key);
  if (!head) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });

  const range = parseRange(request.headers.get('Range'), head.size);
  if (range === 'invalid') {
    return new Response('Invalid range', {
      status: 416,
      headers: artifactHeaders(env, { 'Content-Range': `bytes */${head.size}` }),
    });
  }

  if (request.method === 'HEAD') return cachedHeadResponse(request, env, head, range, ctx);
  if (range) return rangeResponse(request, env, head, range);
  return cachedFullBodyResponse(request, env, head, ctx);
}

export default {
  fetch: handleArtifactRequest,
};
