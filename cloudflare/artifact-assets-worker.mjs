const DEFAULT_APP_ORIGIN = 'https://0x88.app';
const IMMUTABLE_CACHE_CONTROL = 'public, max-age=31536000, immutable';
const RELEASE_CACHE_CONTROL = IMMUTABLE_CACHE_CONTROL;
const CHANNEL_CACHE_CONTROL = 'no-cache';
const LOGICAL_ALIAS_CACHE_CONTROL = 'public, max-age=300, stale-while-revalidate=86400';
const CONTROL_CHANNEL_CACHE_CONTROL = 'public, max-age=60, stale-if-error=86400';
const DEFAULT_CHANNEL_KEY = 'channels/stable.json';
const MAX_CONTROL_MANIFEST_BYTES = 8 * 1024 * 1024;
const EXPOSED_HEADERS = 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, X-Artifact-Encoded-Length, X-Artifact-Decoded-SHA256, X-Artifact-Encoded-SHA256, Content-Range, Accept-Ranges';

function artifactHeaders(env, extra = {}) {
  const headers = new Headers(extra);
  headers.set('Access-Control-Allow-Origin', '*');
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  headers.set('Timing-Allow-Origin', env.TIMING_ALLOW_ORIGIN || DEFAULT_APP_ORIGIN);
  headers.set('Access-Control-Expose-Headers', EXPOSED_HEADERS);
  headers.set('Cache-Control', headers.get('Cache-Control') || 'no-store');
  return headers;
}

function appendVary(headers, value) {
  const values = (headers.get('Vary') || '').split(',').map((entry) => entry.trim()).filter(Boolean);
  if (!values.some((entry) => entry.toLowerCase() === value.toLowerCase())) values.push(value);
  headers.set('Vary', values.join(', '));
}

function withCacheStatus(response, status, { cacheControl, negotiated = false } = {}) {
  const headers = new Headers(response.headers);
  headers.set('Cache-Status', `lc0-artifact-worker; ${status}`);
  headers.set('Age', status === 'hit' ? headers.get('Age') || '1' : '0');
  if (cacheControl) headers.set('Cache-Control', cacheControl);
  if (negotiated) appendVary(headers, 'Accept-Encoding');
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
    // R2 returns an already encoded representation. Without manual mode the
    // Workers runtime can strip Content-Encoding without decoding the body.
    encodeBody: headers.has('Content-Encoding') ? 'manual' : 'automatic',
  });
}

function edgeCache() {
  return globalThis.caches?.default;
}

function canonicalCacheRequest(request, key, kind = 'body') {
  const url = new URL(request.url);
  url.pathname = `/${key}`;
  url.search = kind === 'body' ? '' : `__lc0_artifact_${kind}=v3`;
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
  // R2 publish keys are derived from manifest URLs, so retain percent encoding.
  const key = url.pathname.replace(/^\/+/, '');
  if (key.startsWith('artifacts/sha256/')) return key;
  if (key.startsWith('releases/') && key.endsWith('.json')) return key;
  if (key.startsWith('channels/') && key.endsWith('.json')) return key;
  return undefined;
}

function isImmutableArtifactKey(key) {
  return key.startsWith('artifacts/sha256/');
}

function cacheControlForKey(key) {
  if (isImmutableArtifactKey(key)) return IMMUTABLE_CACHE_CONTROL;
  if (key.startsWith('releases/')) return RELEASE_CACHE_CONTROL;
  if (key.startsWith('channels/')) return CHANNEL_CACHE_CONTROL;
  return 'no-store';
}

function shouldPreventTransform(key, contentType) {
  return key.endsWith('.wasm')
    || key.endsWith('.onnx')
    || key.endsWith('.data')
    || key.endsWith('.gz')
    || contentType === 'application/wasm'
    || contentType === 'application/octet-stream';
}

function preventTransform(cacheControl) {
  return /\bno-transform\b/i.test(cacheControl) ? cacheControl : `${cacheControl}, no-transform`;
}

function representationFromKey(key) {
  const identity = key.match(/^artifacts\/sha256\/([a-f0-9]{64})\/identity$/);
  if (identity) {
    return { encoding: 'identity', url: `/${key}`, sha256: identity[1], rawSha256: identity[1] };
  }
  const br = key.match(/^artifacts\/sha256\/([a-f0-9]{64})\/br\/([a-f0-9]{64})$/);
  if (br) {
    return { encoding: 'br', url: `/${key}`, sha256: br[2], rawSha256: br[1] };
  }
  return { encoding: 'identity', url: `/${key}` };
}

function descriptorForDirectKey(key) {
  const representation = representationFromKey(key);
  return {
    key,
    representation,
    raw: representation.rawSha256 ? { sha256: representation.rawSha256 } : undefined,
    logicalAlias: false,
    negotiated: false,
  };
}

function identityDescriptor(descriptor) {
  if (descriptor.representation?.encoding !== 'br') return descriptor;
  const rawSha256 = descriptor.raw?.sha256 || descriptor.representation.rawSha256;
  if (!rawSha256) return descriptor;
  const identity = descriptor.representations?.find((entry) => entry.encoding === 'identity') || {
    encoding: 'identity',
    url: `/artifacts/sha256/${rawSha256}/identity`,
    sha256: rawSha256,
    bytes: descriptor.raw?.bytes,
    rawSha256,
  };
  const key = artifactKeyFromUrl(identity.url);
  return key ? { ...descriptor, key, representation: identity, negotiated: false } : descriptor;
}

function representationQuality(header, encoding) {
  if (!header) return encoding === 'identity' ? 1 : 0;
  let wildcard;
  let explicit;
  for (const part of header.split(',')) {
    const [rawName, ...parameters] = part.trim().split(';');
    const name = rawName.toLowerCase();
    let quality = 1;
    for (const parameter of parameters) {
      const match = parameter.trim().match(/^q=(0(?:\.\d{0,3})?|1(?:\.0{0,3})?)$/i);
      if (match) quality = Number(match[1]);
    }
    if (name === encoding) explicit = quality;
    else if (name === '*') wildcard = quality;
  }
  if (explicit !== undefined) return explicit;
  if (encoding === 'identity') return wildcard === 0 ? 0 : 1;
  return wildcard ?? 0;
}

function selectRepresentation(artifact, request) {
  const representations = Array.isArray(artifact.representations) ? artifact.representations : [];
  const identity = representations.find((entry) => entry.encoding === 'identity');
  const br = representations.find((entry) => entry.encoding === 'br');
  const acceptEncoding = request.headers.get('Accept-Encoding');
  if (request.headers.get('Range')) {
    return identity && representationQuality(acceptEncoding, 'identity') > 0 ? identity : undefined;
  }
  const identityQuality = identity ? representationQuality(acceptEncoding, 'identity') : -1;
  const brQuality = br ? representationQuality(acceptEncoding, 'br') : -1;
  if (brQuality > 0 && brQuality >= identityQuality) return br;
  if (identityQuality > 0) return identity;
  return undefined;
}

function artifactKeyFromUrl(raw) {
  try {
    const url = new URL(raw, 'https://assets.invalid');
    const key = url.pathname.replace(/^\/+/, '');
    return key.startsWith('artifacts/sha256/') ? key : undefined;
  } catch {
    return undefined;
  }
}

function validateV2Representation(artifact, representation) {
  const key = artifactKeyFromUrl(representation?.url);
  const rawSha256 = artifact?.raw?.sha256?.toLowerCase();
  if (!key || !rawSha256 || !Number.isFinite(artifact.raw.bytes)) return undefined;
  if (representation.encoding === 'identity') {
    if (key !== `artifacts/sha256/${rawSha256}/identity`) return undefined;
    if (representation.sha256?.toLowerCase() !== rawSha256 || representation.bytes !== artifact.raw.bytes) return undefined;
  } else if (representation.encoding === 'br') {
    const encodedSha256 = representation.sha256?.toLowerCase();
    if (!encodedSha256 || key !== `artifacts/sha256/${rawSha256}/br/${encodedSha256}`) return undefined;
  } else {
    return undefined;
  }
  return { ...representation, sha256: representation.sha256.toLowerCase(), rawSha256 };
}

function logicalPathMatches(artifact, logicalUrl) {
  if (artifact?.logicalUrl === logicalUrl) return true;
  const basename = logicalUrl.split('/').pop();
  if (artifact?.file === basename) return true;
  if (typeof artifact?.name !== 'string') return false;
  return artifact.name === logicalUrl || artifact.name === basename || `/${artifact.name.replace(/^\/+/, '')}` === logicalUrl;
}

async function readJsonObjectCached(env, key, request, cacheControl) {
  const cache = edgeCache();
  const cacheRequest = canonicalCacheRequest(request, key, 'control');
  if (cache) {
    const cached = await cache.match(cacheRequest);
    if (cached) return cached.json();
  }
  const object = await env.ARTIFACTS.get(key);
  if (!object) return undefined;
  if (!Number.isFinite(object.size) || object.size > MAX_CONTROL_MANIFEST_BYTES) {
    throw new Error(`Artifact control manifest is too large: ${key}`);
  }
  const response = new Response(object.body, {
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': cacheControl,
      'Content-Length': String(object.size),
    },
  });
  const cacheCopy = cache ? response.clone() : undefined;
  const value = await response.json();
  if (cache && cacheCopy) await cache.put(cacheRequest, cacheCopy).catch(() => undefined);
  return value;
}

async function descriptorFromStableReleaseLogicalPath(request, env) {
  const logicalUrl = new URL(request.url).pathname;
  if (!logicalUrl.startsWith('/models/') && !logicalUrl.startsWith('/stockfish/') && !logicalUrl.startsWith('/berserk/') && !logicalUrl.startsWith('/plentychess/') && !logicalUrl.startsWith('/stormphrax/') && !logicalUrl.startsWith('/viridithas/') && !logicalUrl.startsWith('/monty/') && !logicalUrl.startsWith('/reckless/') && !logicalUrl.startsWith('/runtimes/')) return undefined;
  const channelKey = env.ARTIFACT_CHANNEL_KEY || DEFAULT_CHANNEL_KEY;
  const channel = await readJsonObjectCached(env, channelKey, request, CONTROL_CHANNEL_CACHE_CONTROL);
  const releasePath = channel?.releaseManifestUrl || channel?.releaseUrl;
  if (!releasePath) return undefined;
  const releaseKey = String(releasePath).replace(/^\/+/, '');
  if (!releaseKey.startsWith('releases/') || !releaseKey.endsWith('.json')) return undefined;
  const release = await readJsonObjectCached(env, releaseKey, request, IMMUTABLE_CACHE_CONTROL);
  const artifacts = release?.artifacts ?? [];
  const exactArtifact = artifacts.find((entry) => entry.logicalUrl === logicalUrl);
  const fallbackArtifacts = exactArtifact ? [] : artifacts.filter((entry) => !entry.logicalUrl && logicalPathMatches(entry, logicalUrl));
  const artifact = exactArtifact ?? (fallbackArtifacts.length === 1 ? fallbackArtifacts[0] : undefined);
  if (!artifact) return undefined;

  if (artifact.raw && Array.isArray(artifact.representations)) {
    const selected = selectRepresentation(artifact, request);
    if (!selected) return { unacceptableEncoding: true };
    const representation = validateV2Representation(artifact, selected);
    const key = representation && artifactKeyFromUrl(representation.url);
    if (!representation || !key) return undefined;
    return {
      key,
      representation,
      representations: artifact.representations.map((entry) => validateV2Representation(artifact, entry)).filter(Boolean),
      raw: { sha256: artifact.raw.sha256.toLowerCase(), bytes: artifact.raw.bytes },
      contentType: artifact.contentType,
      logicalAlias: true,
      negotiated: !request.headers.get('Range'),
    };
  }

  if (!artifact.artifactUrl) return undefined;
  const key = artifactKeyFromUrl(artifact.artifactUrl);
  if (!key) return undefined;
  return {
    key,
    representation: { encoding: 'identity', url: artifact.artifactUrl, sha256: artifact.sha256, bytes: artifact.bytes },
    raw: Number.isFinite(artifact.bytes) ? { sha256: artifact.sha256, bytes: artifact.bytes } : undefined,
    contentType: artifact.contentType,
    logicalAlias: true,
    negotiated: false,
  };
}

function resolvedMetadata(descriptor, object) {
  const custom = object?.customMetadata || {};
  const representation = descriptor.representation || representationFromKey(descriptor.key);
  const encoding = representation.encoding || custom.encoding || 'identity';
  const customEncodedBytes = typeof custom.encodedBytes === 'string' && custom.encodedBytes.trim() !== ''
    ? Number(custom.encodedBytes)
    : undefined;
  const encodedBytes = Number.isFinite(representation.bytes)
    ? representation.bytes
    : Number.isFinite(customEncodedBytes)
      ? customEncodedBytes
      : object?.size;
  const customDecodedBytes = typeof custom.decodedBytes === 'string' && custom.decodedBytes.trim() !== ''
    ? Number(custom.decodedBytes)
    : undefined;
  const decodedBytes = Number.isFinite(descriptor.raw?.bytes)
    ? descriptor.raw.bytes
    : Number.isFinite(customDecodedBytes)
      ? customDecodedBytes
      : encoding === 'identity' ? encodedBytes : undefined;
  return {
    encoding,
    encodedBytes,
    decodedBytes,
    decodedSha256: descriptor.raw?.sha256 || representation.rawSha256 || custom.decodedSha256,
    encodedSha256: representation.sha256 || custom.encodedSha256,
    contentType: descriptor.contentType || object?.httpMetadata?.contentType || custom.contentType,
  };
}

function objectHeaders(descriptor, object, env, range, cacheControlOverride) {
  const metadata = resolvedMetadata(descriptor, object);
  const contentType = metadata.contentType || (descriptor.key.endsWith('.json') ? 'application/json; charset=utf-8' : 'application/octet-stream');
  const baseCacheControl = cacheControlOverride || cacheControlForKey(descriptor.key);
  const cacheControl = shouldPreventTransform(descriptor.key, contentType) ? preventTransform(baseCacheControl) : baseCacheControl;
  const headers = artifactHeaders(env, {
    'Content-Type': contentType,
    'Cache-Control': cacheControl,
    ETag: object?.httpEtag || object?.etag || (metadata.encodedSha256 ? `"${metadata.encodedSha256}"` : ''),
  });
  if (metadata.encoding !== 'identity') headers.set('Content-Encoding', metadata.encoding);
  headers.set('Accept-Ranges', 'bytes');
  if (isImmutableArtifactKey(descriptor.key)) {
    if (Number.isFinite(metadata.decodedBytes)) headers.set('X-Artifact-Content-Length', String(metadata.decodedBytes));
    if (!range && Number.isFinite(metadata.encodedBytes)) headers.set('X-Artifact-Encoded-Length', String(metadata.encodedBytes));
    if (metadata.decodedSha256) headers.set('X-Artifact-Decoded-SHA256', metadata.decodedSha256);
    if (metadata.encodedSha256) headers.set('X-Artifact-Encoded-SHA256', metadata.encodedSha256);
  }
  if (range) {
    headers.delete('Content-Encoding');
    headers.set('Content-Length', String(range.length));
    headers.set('Content-Range', `bytes ${range.start}-${range.end}/${metadata.decodedBytes ?? object.size}`);
  } else if (Number.isFinite(metadata.encodedBytes)) {
    headers.set('Content-Length', String(metadata.encodedBytes));
  }
  return headers;
}

function scheduleCachePut(cache, request, response, ctx) {
  const operation = cache.put(request, response.clone()).catch(() => undefined);
  if (ctx?.waitUntil) ctx.waitUntil(operation);
  else void operation;
}

function applyDescriptorHeaders(response, descriptor, env, range) {
  const cachedHeaders = response.headers;
  const object = {
    size: Number(cachedHeaders.get('Content-Length')),
    httpEtag: cachedHeaders.get('ETag') || undefined,
    httpMetadata: {
      contentType: cachedHeaders.get('Content-Type') || undefined,
      contentEncoding: cachedHeaders.get('Content-Encoding') || undefined,
    },
    customMetadata: {
      decodedBytes: cachedHeaders.get('X-Artifact-Content-Length') || '',
      encodedBytes: cachedHeaders.get('X-Artifact-Encoded-Length') || '',
      decodedSha256: cachedHeaders.get('X-Artifact-Decoded-SHA256') || '',
      encodedSha256: cachedHeaders.get('X-Artifact-Encoded-SHA256') || '',
    },
  };
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: objectHeaders(descriptor, object, env, range),
  });
}

async function fullBodyResponse(descriptor, request, env, ctx) {
  const cache = edgeCache();
  const cacheRequest = canonicalCacheRequest(request, descriptor.key);
  if (cache && isImmutableArtifactKey(descriptor.key)) {
    const cached = await cache.match(cacheRequest);
    if (cached) return withCacheStatus(applyDescriptorHeaders(cached, descriptor, env), 'hit', {
      cacheControl: descriptor.logicalAlias ? LOGICAL_ALIAS_CACHE_CONTROL : undefined,
      negotiated: descriptor.negotiated,
    });
  }

  const object = await env.ARTIFACTS.get(descriptor.key);
  if (!object) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });
  const canonical = new Response(object.body, { status: 200, headers: objectHeaders(descriptor, object, env) });
  if (cache && isImmutableArtifactKey(descriptor.key)) scheduleCachePut(cache, cacheRequest, canonical, ctx);
  return withCacheStatus(canonical, cache && isImmutableArtifactKey(descriptor.key) ? 'miss' : 'fwd', {
    cacheControl: descriptor.logicalAlias ? LOGICAL_ALIAS_CACHE_CONTROL : undefined,
    negotiated: descriptor.negotiated,
  });
}

async function headResponse(descriptor, request, env) {
  const rangeHeader = request.headers.get('Range');
  if (rangeHeader && descriptor.representation?.encoding === 'br' && !descriptor.logicalAlias) {
    return new Response(null, { status: 416, headers: artifactHeaders(env, { 'Accept-Ranges': 'none' }) });
  }
  if (rangeHeader && descriptor.logicalAlias) descriptor = identityDescriptor(descriptor);
  const cache = edgeCache();
  const cacheRequest = canonicalCacheRequest(request, descriptor.key, 'head');
  if (!rangeHeader && cache && isImmutableArtifactKey(descriptor.key)) {
    const cached = await cache.match(cacheRequest);
    if (cached) return withCacheStatus(applyDescriptorHeaders(cached, descriptor, env), 'hit', {
      cacheControl: descriptor.logicalAlias ? LOGICAL_ALIAS_CACHE_CONTROL : undefined,
      negotiated: descriptor.negotiated,
    });
  }

  // Expected release metadata cannot prove that the immutable R2 object exists.
  const head = await env.ARTIFACTS.head(descriptor.key);
  if (!head) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });
  const totalBytes = descriptor.raw?.bytes ?? head.size;
  const range = parseRange(rangeHeader, totalBytes);
  if (range === 'invalid') {
    return new Response('Invalid range', {
      status: 416,
      headers: artifactHeaders(env, { 'Content-Range': `bytes */${totalBytes}` }),
    });
  }
  const canonical = new Response(null, { status: range ? 206 : 200, headers: objectHeaders(descriptor, head, env, range) });
  if (!range && cache && isImmutableArtifactKey(descriptor.key)) await cache.put(cacheRequest, canonical.clone()).catch(() => undefined);
  return withCacheStatus(canonical, !range && cache && isImmutableArtifactKey(descriptor.key) ? 'miss' : 'fwd', {
    cacheControl: descriptor.logicalAlias ? LOGICAL_ALIAS_CACHE_CONTROL : undefined,
    negotiated: descriptor.negotiated && !range,
  });
}

async function rangeResponse(descriptor, request, env) {
  if (descriptor.representation?.encoding === 'br' && !descriptor.logicalAlias) {
    return new Response('Range is not supported for an encoded representation', {
      status: 416,
      headers: artifactHeaders(env, { 'Accept-Ranges': 'none' }),
    });
  }
  if (descriptor.logicalAlias) descriptor = identityDescriptor(descriptor);
  let size = descriptor.raw?.bytes || descriptor.representation?.bytes;
  let head;
  if (!Number.isFinite(size)) {
    head = await env.ARTIFACTS.head(descriptor.key);
    if (!head) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });
    size = head.size;
  } else {
    head = {
      size,
      httpEtag: descriptor.representation?.sha256 ? `"${descriptor.representation.sha256}"` : undefined,
      httpMetadata: descriptor.contentType ? { contentType: descriptor.contentType } : undefined,
    };
  }
  const range = parseRange(request.headers.get('Range'), size);
  if (range === 'invalid') {
    return new Response('Invalid range', {
      status: 416,
      headers: artifactHeaders(env, { 'Content-Range': `bytes */${size}` }),
    });
  }
  const object = await env.ARTIFACTS.get(descriptor.key, { range: { offset: range.start, length: range.length } });
  if (!object) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });
  return withCacheStatus(new Response(object.body, { status: 206, headers: objectHeaders(descriptor, head, env, range) }), 'fwd', {
    cacheControl: descriptor.logicalAlias ? LOGICAL_ALIAS_CACHE_CONTROL : undefined,
  });
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

  const directKey = keyFromRequest(request);
  let descriptor = directKey ? descriptorForDirectKey(directKey) : await descriptorFromStableReleaseLogicalPath(request, env);
  if (descriptor?.unacceptableEncoding) {
    return new Response('No acceptable artifact representation', { status: 406, headers: artifactHeaders(env, { Vary: 'Accept-Encoding' }) });
  }
  if (!descriptor) return new Response('Not found', { status: 404, headers: artifactHeaders(env) });

  if (request.method === 'HEAD') return headResponse(descriptor, request, env);
  if (request.headers.get('Range')) return rangeResponse(descriptor, request, env);
  return fullBodyResponse(descriptor, request, env, ctx);
}

export default {
  fetch: handleArtifactRequest,
};
