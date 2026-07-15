import assert from 'node:assert/strict';
import { test } from 'node:test';
import { handleArtifactRequest } from '../cloudflare/artifact-assets-worker.mjs';

const KEY = 'artifacts/sha256/abc/test.bin';
const ESCAPED_KEY = 'artifacts/sha256/abc/model%20v1.onnx';
const RELEASE_KEY = 'releases/2026-06-22.test.json';
const CHANNEL_KEY = 'channels/stable.json';
const BODY = new TextEncoder().encode('abcdefghijklmnopqrstuvwxyz');
const RELEASE_BODY = new TextEncoder().encode('{"schema":"release"}');
const CHANNEL_BODY = new TextEncoder().encode('{"schema":"channel"}');

function fakeEnv() {
  const entries = new Map([
    [KEY, { body: BODY, contentType: 'application/octet-stream', cacheControl: 'public, max-age=31536000, immutable' }],
    [ESCAPED_KEY, { body: BODY, contentType: 'application/octet-stream', cacheControl: 'public, max-age=31536000, immutable' }],
    [RELEASE_KEY, { body: RELEASE_BODY, contentType: 'application/json; charset=utf-8', cacheControl: 'public, max-age=31536000, immutable' }],
    [CHANNEL_KEY, { body: CHANNEL_BODY, contentType: 'application/json; charset=utf-8', cacheControl: 'public, max-age=31536000, immutable' }],
  ]);
  return {
    TIMING_ALLOW_ORIGIN: 'https://0x88.app',
    ARTIFACTS: {
      async head(key) {
        const entry = entries.get(key);
        return entry ? {
          size: entry.body.byteLength,
          httpEtag: '"fake-etag"',
          httpMetadata: { contentType: entry.contentType, cacheControl: entry.cacheControl },
        } : null;
      },
      async get(key, options) {
        const entry = entries.get(key);
        if (!entry) return null;
        const range = options?.range;
        const body = range ? entry.body.slice(range.offset, range.offset + range.length) : entry.body;
        return {
          size: entry.body.byteLength,
          httpEtag: '"fake-etag"',
          httpMetadata: { contentType: entry.contentType, cacheControl: entry.cacheControl },
          body,
        };
      },
    },
  };
}

async function text(response) {
  return new TextDecoder().decode(await response.arrayBuffer());
}

class FakeCache {
  constructor() { this.store = new Map(); }
  async match(request) { return this.store.get(request.url); }
  async put(request, response) { this.store.set(request.url, response); }
}

async function withFakeEdgeCache(run) {
  const previous = globalThis.caches;
  globalThis.caches = { default: new FakeCache() };
  try {
    return await run();
  } finally {
    globalThis.caches = previous;
  }
}

test('artifact assets worker serves full immutable artifacts with required headers', async () => {
  const response = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`), fakeEnv());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
  assert.equal(response.headers.get('Cross-Origin-Resource-Policy'), 'cross-origin');
  assert.equal(response.headers.get('Timing-Allow-Origin'), 'https://0x88.app');
  assert.match(response.headers.get('Access-Control-Expose-Headers'), /Content-Length/);
  assert.match(response.headers.get('Access-Control-Expose-Headers'), /X-Artifact-Content-Length/);
  assert.match(response.headers.get('Access-Control-Expose-Headers'), /X-Artifact-Encoded-Length/);
  assert.equal(response.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
  assert.equal(response.headers.get('X-Artifact-Encoded-Length'), String(BODY.byteLength));
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=31536000, immutable, no-transform');
  assert.equal(response.headers.get('Cache-Status'), 'lc0-artifact-worker; fwd');
  assert.equal(response.headers.get('Accept-Ranges'), 'bytes');
  assert.equal(response.headers.get('Content-Length'), String(BODY.byteLength));
  assert.equal(await text(response), 'abcdefghijklmnopqrstuvwxyz');
});

test('artifact assets worker caches HEAD metadata without R2 body fetches', async () => {
  await withFakeEdgeCache(async () => {
    const request = new Request(`https://assets.example/${KEY}`, { method: 'HEAD' });
    const first = await handleArtifactRequest(request, fakeEnv());
    assert.equal(first.headers.get('Cache-Status'), 'lc0-artifact-worker; miss');
    assert.equal(first.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
    const second = await handleArtifactRequest(request, fakeEnv());
    assert.equal(second.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(second.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
  });
});

test('artifact assets worker preserves encoded length through normalized cached HEAD metadata', async () => {
  const previous = globalThis.caches;
  let cached;
  globalThis.caches = { default: {
    async match() { return cached; },
    async put(_request, response) {
      const headers = new Headers(response.headers);
      headers.set('Content-Length', '0');
      cached = new Response(null, { status: response.status, headers });
    },
  } };
  try {
    const request = new Request(`https://assets.example/${V2_BR_KEY}`, { method: 'HEAD' });
    const env = fakeV2Env();
    const first = await handleArtifactRequest(request, env);
    assert.equal(first.headers.get('Content-Encoding'), 'br');
    assert.equal(first.headers.get('X-Artifact-Encoded-Length'), String(V2_BR_BODY.byteLength));
    const second = await handleArtifactRequest(request, env);
    assert.equal(second.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(second.headers.get('Content-Length'), String(V2_BR_BODY.byteLength));
    assert.equal(second.headers.get('X-Artifact-Encoded-Length'), String(V2_BR_BODY.byteLength));
  } finally {
    globalThis.caches = previous;
  }
});

test('artifact assets worker serves cached full artifacts without an R2 head hit', async () => {
  await withFakeEdgeCache(async () => {
    const request = new Request(`https://assets.example/${KEY}`);
    const first = await handleArtifactRequest(request, fakeEnv());
    assert.equal(first.headers.get('Cache-Status'), 'lc0-artifact-worker; miss');
    assert.equal(await text(first), 'abcdefghijklmnopqrstuvwxyz');
    const unavailableEnv = { ARTIFACTS: { head: async () => { throw new Error('R2 unavailable'); } } };
    const second = await handleArtifactRequest(request, unavailableEnv);
    assert.equal(second.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(await text(second), 'abcdefghijklmnopqrstuvwxyz');
  });
});

test('artifact assets worker serves cached HEAD metadata without an R2 head hit', async () => {
  await withFakeEdgeCache(async () => {
    const request = new Request(`https://assets.example/${KEY}`, { method: 'HEAD' });
    const first = await handleArtifactRequest(request, fakeEnv());
    assert.equal(first.headers.get('Cache-Status'), 'lc0-artifact-worker; miss');
    const unavailableEnv = { ARTIFACTS: { head: async () => { throw new Error('R2 unavailable'); } } };
    const second = await handleArtifactRequest(request, unavailableEnv);
    assert.equal(second.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(second.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
  });
});

test('artifact assets worker serves valid byte ranges', async () => {
  const response = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`, {
    headers: { Range: 'bytes=2-5' },
  }), fakeEnv());
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('Content-Range'), `bytes 2-5/${BODY.byteLength}`);
  assert.equal(response.headers.get('Content-Length'), '4');
  assert.equal(await text(response), 'cdef');
});

test('artifact assets worker serves suffix byte ranges', async () => {
  const response = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`, {
    headers: { Range: 'bytes=-3' },
  }), fakeEnv());
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('Content-Range'), `bytes ${BODY.byteLength - 3}-${BODY.byteLength - 1}/${BODY.byteLength}`);
  assert.equal(await text(response), 'xyz');
});

test('artifact assets worker does not materialize cached full bodies for ranges', async () => {
  await withFakeEdgeCache(async () => {
    const full = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`), fakeEnv());
    assert.equal(full.headers.get('Cache-Status'), 'lc0-artifact-worker; miss');
    assert.equal(await text(full), 'abcdefghijklmnopqrstuvwxyz');
    const range = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`, {
      headers: { Range: 'bytes=0-2' },
    }), fakeEnv());
    assert.equal(range.headers.get('Cache-Status'), 'lc0-artifact-worker; fwd');
    assert.equal(await text(range), 'abc');
  });
});

test('artifact assets worker still serves when cache population fails', async () => {
  const previous = globalThis.caches;
  globalThis.caches = { default: { match: async () => undefined, put: async () => { throw new Error('cache unavailable'); } } };
  try {
    const response = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`), fakeEnv());
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('Cache-Status'), 'lc0-artifact-worker; miss');
    assert.equal(await text(response), 'abcdefghijklmnopqrstuvwxyz');
  } finally {
    globalThis.caches = previous;
  }
});

test('artifact assets worker serves immutable release and mutable channel manifests with distinct policies', async () => {
  const release = await handleArtifactRequest(new Request(`https://assets.example/${RELEASE_KEY}`), fakeEnv());
  assert.equal(release.status, 200);
  assert.equal(release.headers.get('Content-Type'), 'application/json; charset=utf-8');
  assert.equal(release.headers.get('Cache-Control'), 'public, max-age=31536000, immutable');
  assert.equal(release.headers.get('X-Artifact-Content-Length'), null);
  assert.equal(await text(release), '{"schema":"release"}');

  const channel = await handleArtifactRequest(new Request(`https://assets.example/${CHANNEL_KEY}`), fakeEnv());
  assert.equal(channel.status, 200);
  assert.equal(channel.headers.get('Cache-Control'), 'no-cache');
  assert.equal(channel.headers.get('X-Artifact-Content-Length'), null);
  assert.equal(await text(channel), '{"schema":"channel"}');
});

test('artifact assets worker uses GET metadata for mutable manifest bodies', async () => {
  const oldBody = new TextEncoder().encode('{"schema":"old"}');
  const newBody = new TextEncoder().encode('{"schema":"new-channel"}');
  const env = {
    TIMING_ALLOW_ORIGIN: 'https://0x88.app',
    ARTIFACTS: {
      async head() {
        return {
          size: oldBody.byteLength,
          httpEtag: '"old"',
          httpMetadata: { contentType: 'application/json; charset=utf-8', cacheControl: 'public, max-age=31536000, immutable' },
        };
      },
      async get() {
        return {
          size: newBody.byteLength,
          httpEtag: '"new"',
          httpMetadata: { contentType: 'application/json; charset=utf-8', cacheControl: 'public, max-age=31536000, immutable' },
          body: newBody,
        };
      },
    },
  };
  const response = await handleArtifactRequest(new Request(`https://assets.example/${CHANNEL_KEY}`), env);
  assert.equal(response.headers.get('ETag'), '"new"');
  assert.equal(response.headers.get('Content-Length'), String(newBody.byteLength));
  assert.equal(await text(response), '{"schema":"new-channel"}');
});

test('artifact assets worker does not edge-cache mutable channel manifests', async () => {
  await withFakeEdgeCache(async () => {
    const request = new Request(`https://assets.example/${CHANNEL_KEY}`);
    const first = await handleArtifactRequest(request, fakeEnv());
    assert.equal(first.headers.get('Cache-Status'), 'lc0-artifact-worker; fwd');
    const second = await handleArtifactRequest(request, fakeEnv());
    assert.equal(second.headers.get('Cache-Status'), 'lc0-artifact-worker; fwd');
  });
});

test('artifact assets worker preserves percent-encoded R2 keys', async () => {
  const response = await handleArtifactRequest(new Request(`https://assets.example/${ESCAPED_KEY}`), fakeEnv());
  assert.equal(response.status, 200);
  assert.equal(await text(response), 'abcdefghijklmnopqrstuvwxyz');
});

test('artifact assets worker serves stable logical asset paths through the channel release manifest with short caching', async () => {
  const channelBody = new TextEncoder().encode(JSON.stringify({ releaseManifestUrl: '/releases/stable-test.json' }));
  const releaseBody = new TextEncoder().encode(JSON.stringify({
    artifacts: [{
      logicalUrl: '/stormphrax/stormphrax-emscripten.js',
      artifactUrl: `https://assets.example/${KEY}`,
    }],
  }));
  const entries = new Map([
    [KEY, { body: BODY, contentType: 'text/javascript; charset=utf-8' }],
    ['channels/stable.json', { body: channelBody, contentType: 'application/json; charset=utf-8' }],
    ['releases/stable-test.json', { body: releaseBody, contentType: 'application/json; charset=utf-8' }],
  ]);
  const env = {
    TIMING_ALLOW_ORIGIN: 'https://0x88.app',
    ARTIFACTS: {
      async head(key) {
        const entry = entries.get(key);
        return entry ? { size: entry.body.byteLength, httpEtag: '"etag"', httpMetadata: { contentType: entry.contentType } } : null;
      },
      async get(key, options) {
        const entry = entries.get(key);
        if (!entry) return null;
        const range = options?.range;
        const body = range ? entry.body.slice(range.offset, range.offset + range.length) : entry.body;
        return { size: entry.body.byteLength, httpEtag: '"etag"', httpMetadata: { contentType: entry.contentType }, body };
      },
    },
  };
  await withFakeEdgeCache(async () => {
    const request = new Request('https://assets.example/stormphrax/stormphrax-emscripten.js');
    const response = await handleArtifactRequest(request, env);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('Cache-Control'), 'public, max-age=300, stale-while-revalidate=86400');
    assert.equal(response.headers.get('Cache-Status'), 'lc0-artifact-worker; miss');
    assert.equal(response.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
    assert.equal(response.headers.get('Content-Type'), 'text/javascript; charset=utf-8');
    assert.equal(await text(response), 'abcdefghijklmnopqrstuvwxyz');
    const cached = await handleArtifactRequest(request, env);
    assert.equal(cached.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(await text(cached), 'abcdefghijklmnopqrstuvwxyz');
  });
});

test('artifact assets worker binds engine manifests to the selected release', async () => {
  const manifestKey = `artifacts/sha256/${'a'.repeat(64)}/stormphrax.manifest.json`;
  const channelBody = new TextEncoder().encode(JSON.stringify({ releaseManifestUrl: '/releases/stable-test.json' }));
  const releaseBody = new TextEncoder().encode(JSON.stringify({
    sourceManifests: ['public/stormphrax/stormphrax-emscripten-single-thread.manifest.json'],
    artifacts: [{
      logicalUrl: '/stormphrax/stormphrax-emscripten-single-thread.manifest.json',
      artifactUrl: `https://assets.example/${manifestKey}`,
    }],
  }));
  const manifestBody = new TextEncoder().encode('{}');
  const entries = new Map([
    ['channels/stable.json', { body: channelBody, contentType: 'application/json; charset=utf-8' }],
    ['releases/stable-test.json', { body: releaseBody, contentType: 'application/json; charset=utf-8' }],
    [manifestKey, { body: manifestBody, contentType: 'application/json; charset=utf-8' }],
  ]);
  const env = {
    ARTIFACTS: {
      async head(key) {
        const entry = entries.get(key);
        return entry ? { size: entry.body.byteLength, httpEtag: '"etag"', httpMetadata: { contentType: entry.contentType } } : null;
      },
      async get(key) {
        const entry = entries.get(key);
        return entry ? { size: entry.body.byteLength, httpEtag: '"etag"', httpMetadata: { contentType: entry.contentType }, body: entry.body } : null;
      },
    },
  };
  const response = await handleArtifactRequest(new Request('https://assets.example/stormphrax/stormphrax-emscripten-single-thread.manifest.json', { method: 'HEAD' }), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=300, stale-while-revalidate=86400');
  assert.equal(response.headers.get('X-Artifact-Content-Length'), String(manifestBody.byteLength));
  const unrelated = await handleArtifactRequest(new Request('https://assets.example/stormphrax/future.manifest.json', { method: 'HEAD' }), env);
  assert.equal(unrelated.status, 404);
});

test('artifact assets worker rejects non-artifact paths and invalid ranges', async () => {
  const notFound = await handleArtifactRequest(new Request('https://assets.example/models/raw.onnx'), fakeEnv());
  assert.equal(notFound.status, 404);
  assert.equal(notFound.headers.get('Cache-Control'), 'no-store');
  const badRange = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`, {
    headers: { Range: 'bytes=999-1000' },
  }), fakeEnv());
  assert.equal(badRange.status, 416);
  assert.equal(badRange.headers.get('Content-Range'), `bytes */${BODY.byteLength}`);
});

const V2_RAW_SHA = 'a'.repeat(64);
const V2_BR_SHA = 'b'.repeat(64);
const V2_IDENTITY_KEY = `artifacts/sha256/${V2_RAW_SHA}/identity`;
const V2_BR_KEY = `artifacts/sha256/${V2_RAW_SHA}/br/${V2_BR_SHA}`;
const V2_BR_BODY = new TextEncoder().encode('compressed-body');

function fakeV2Env() {
  const counts = { head: new Map(), get: new Map() };
  const channel = new TextEncoder().encode(JSON.stringify({
    schema: 'lc0-webgpu.artifact-channel.v2',
    releaseId: 'v2-test',
    releaseUrl: '/releases/v2-test.json',
  }));
  const artifact = {
    name: 'model-a',
    logicalUrl: '/models/lc0/model-a.onnx',
    contentType: 'application/octet-stream',
    raw: { sha256: V2_RAW_SHA, bytes: BODY.byteLength },
    representations: [
      { encoding: 'identity', url: `/${V2_IDENTITY_KEY}`, sha256: V2_RAW_SHA, bytes: BODY.byteLength },
      { encoding: 'br', url: `/${V2_BR_KEY}`, sha256: V2_BR_SHA, bytes: V2_BR_BODY.byteLength },
    ],
  };
  const release = new TextEncoder().encode(JSON.stringify({
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'v2-test',
    artifacts: [artifact, { ...artifact, name: 'model-b', file: 'model-b.onnx', logicalUrl: undefined, contentType: 'application/wasm' }],
  }));
  const entries = new Map([
    ['channels/stable.json', { body: channel, contentType: 'application/json; charset=utf-8' }],
    ['releases/v2-test.json', { body: release, contentType: 'application/json; charset=utf-8' }],
    [V2_IDENTITY_KEY, { body: BODY, contentType: 'application/octet-stream' }],
    [V2_BR_KEY, { body: V2_BR_BODY, contentType: 'application/octet-stream', contentEncoding: 'br' }],
  ]);
  const increment = (map, key) => map.set(key, (map.get(key) || 0) + 1);
  return {
    counts,
    ARTIFACTS: {
      async head(key) {
        increment(counts.head, key);
        const entry = entries.get(key);
        return entry ? {
          size: entry.body.byteLength,
          httpEtag: '"v2-etag"',
          httpMetadata: { contentType: entry.contentType, contentEncoding: entry.contentEncoding },
        } : null;
      },
      async get(key, options) {
        increment(counts.get, key);
        const entry = entries.get(key);
        if (!entry) return null;
        const range = options?.range;
        const body = range ? entry.body.slice(range.offset, range.offset + range.length) : entry.body;
        return {
          size: entry.body.byteLength,
          httpEtag: '"v2-etag"',
          httpMetadata: { contentType: entry.contentType, contentEncoding: entry.contentEncoding },
          body,
        };
      },
    },
  };
}

test('artifact assets worker negotiates a v2 Brotli representation with decoded integrity metadata', async () => {
  const response = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', {
    headers: { 'Accept-Encoding': 'br' },
  }), fakeV2Env());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Content-Encoding'), 'br');
  assert.equal(response.headers.get('Vary'), 'Accept-Encoding');
  assert.equal(response.headers.get('Content-Length'), String(V2_BR_BODY.byteLength));
  assert.equal(response.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
  assert.equal(response.headers.get('X-Artifact-Encoded-Length'), String(V2_BR_BODY.byteLength));
  assert.equal(response.headers.get('X-Artifact-Decoded-SHA256'), V2_RAW_SHA);
  assert.equal(response.headers.get('X-Artifact-Encoded-SHA256'), V2_BR_SHA);
  assert.equal(await text(response), 'compressed-body');
});

test('artifact assets worker forces v2 Range requests to the identity representation', async () => {
  const env = fakeV2Env();
  const response = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', {
    headers: { 'Accept-Encoding': 'br', Range: 'bytes=2-5' },
  }), env);
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('Content-Encoding'), null);
  assert.equal(response.headers.get('Content-Range'), `bytes 2-5/${BODY.byteLength}`);
  assert.equal(await text(response), 'cdef');
  assert.equal(env.counts.get.get(V2_IDENTITY_KEY), 1);
  assert.equal(env.counts.get.get(V2_BR_KEY), undefined);
});

test('artifact assets worker shares canonical v2 body and control caches across logical aliases', async () => {
  await withFakeEdgeCache(async () => {
    const env = fakeV2Env();
    const first = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx'), env);
    assert.equal(await text(first), 'abcdefghijklmnopqrstuvwxyz');
    const second = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-b.onnx'), env);
    assert.equal(second.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(second.headers.get('Content-Type'), 'application/wasm');
    assert.equal(second.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
    assert.equal(await text(second), 'abcdefghijklmnopqrstuvwxyz');
    assert.equal(env.counts.get.get('channels/stable.json'), 1);
    assert.equal(env.counts.get.get('releases/v2-test.json'), 1);
    assert.equal(env.counts.get.get(V2_IDENTITY_KEY), 1);
    assert.equal([...env.counts.head.values()].reduce((sum, value) => sum + value, 0), 0);
  });
});

test('artifact assets worker reapplies logical metadata after a direct canonical body cache fill', async () => {
  await withFakeEdgeCache(async () => {
    const env = fakeV2Env();
    const direct = await handleArtifactRequest(new Request(`https://assets.example/${V2_BR_KEY}`), env);
    assert.equal(direct.headers.get('X-Artifact-Content-Length'), null);
    assert.equal(await text(direct), 'compressed-body');
    const alias = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', {
      headers: { 'Accept-Encoding': 'br' },
    }), env);
    assert.equal(alias.headers.get('Cache-Status'), 'lc0-artifact-worker; hit');
    assert.equal(alias.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
    assert.equal(alias.headers.get('X-Artifact-Decoded-SHA256'), V2_RAW_SHA);
    assert.equal(await text(alias), 'compressed-body');

    const directHeadRequest = new Request(`https://assets.example/${V2_BR_KEY}`, { method: 'HEAD' });
    const firstHead = await handleArtifactRequest(directHeadRequest, env);
    const cachedHead = await handleArtifactRequest(directHeadRequest, env);
    assert.equal(firstHead.headers.get('X-Artifact-Content-Length'), null);
    assert.equal(cachedHead.headers.get('X-Artifact-Content-Length'), null);
  });
});

test('artifact assets worker rejects Range on an explicit Brotli representation without reading identity', async () => {
  const env = fakeV2Env();
  const response = await handleArtifactRequest(new Request(`https://assets.example/${V2_BR_KEY}`, {
    headers: { Range: 'bytes=0-3' },
  }), env);
  assert.equal(response.status, 416);
  assert.equal(response.headers.get('Accept-Ranges'), 'none');
  assert.equal(env.counts.get.get(V2_IDENTITY_KEY), undefined);
  assert.equal(env.counts.get.get(V2_BR_KEY), undefined);

  const head = await handleArtifactRequest(new Request(`https://assets.example/${V2_BR_KEY}`, {
    method: 'HEAD',
    headers: { Range: 'bytes=0-3' },
  }), env);
  assert.equal(head.status, 416);
  assert.equal((await head.arrayBuffer()).byteLength, 0);
});

test('artifact assets worker verifies R2 existence before answering a logical HEAD', async () => {
  const env = fakeV2Env();
  const originalHead = env.ARTIFACTS.head;
  env.ARTIFACTS.head = async (key) => key === V2_IDENTITY_KEY ? null : originalHead(key);
  const response = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', { method: 'HEAD' }), env);
  assert.equal(response.status, 404);
});

test('artifact assets worker keeps ranged HEAD responses bodyless and avoids R2 GET', async () => {
  const env = fakeV2Env();
  const response = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', {
    method: 'HEAD',
    headers: { Range: 'bytes=0-3' },
  }), env);
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('Content-Range'), `bytes 0-3/${BODY.byteLength}`);
  assert.equal((await response.arrayBuffer()).byteLength, 0);
  assert.equal(env.counts.head.get(V2_IDENTITY_KEY), 1);
  assert.equal(env.counts.get.get(V2_IDENTITY_KEY), undefined);
});

test('artifact assets worker performs one body-cache lookup and no HEAD on a full GET miss', async () => {
  const previous = globalThis.caches;
  let matches = 0;
  globalThis.caches = { default: {
    async match() { matches += 1; return undefined; },
    async put() {},
  } };
  try {
    const env = fakeEnv();
    let heads = 0;
    let gets = 0;
    const originalHead = env.ARTIFACTS.head;
    const originalGet = env.ARTIFACTS.get;
    env.ARTIFACTS.head = async (...args) => { heads += 1; return originalHead(...args); };
    env.ARTIFACTS.get = async (...args) => { gets += 1; return originalGet(...args); };
    const response = await handleArtifactRequest(new Request(`https://assets.example/${KEY}`), env);
    assert.equal(await text(response), 'abcdefghijklmnopqrstuvwxyz');
    assert.equal(matches, 1);
    assert.equal(heads, 0);
    assert.equal(gets, 1);
  } finally {
    globalThis.caches = previous;
  }
});

test('artifact assets worker rejects ambiguous name-only v2 basename aliases', async () => {
  const env = fakeV2Env();
  const originalGet = env.ARTIFACTS.get;
  env.ARTIFACTS.get = async (key, options) => {
    if (key !== 'releases/v2-test.json') return originalGet(key, options);
    const raw = { sha256: V2_RAW_SHA, bytes: BODY.byteLength };
    const representations = [{ encoding: 'identity', url: `/${V2_IDENTITY_KEY}`, sha256: V2_RAW_SHA, bytes: BODY.byteLength }];
    const body = new TextEncoder().encode(JSON.stringify({
      schema: 'lc0-webgpu.artifact-release.v2',
      artifacts: [
        { name: 'one', file: 'model-a.onnx', raw, representations },
        { name: 'two', file: 'model-a.onnx', raw, representations },
      ],
    }));
    return { size: body.byteLength, httpMetadata: { contentType: 'application/json' }, body };
  };
  const response = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx'), env);
  assert.equal(response.status, 404);
});

test('artifact assets worker returns 406 when a v2 client rejects every available encoding', async () => {
  const response = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', {
    headers: { 'Accept-Encoding': 'identity;q=0, br;q=0' },
  }), fakeV2Env());
  assert.equal(response.status, 406);
  assert.equal(response.headers.get('Vary'), 'Accept-Encoding');

  const range = await handleArtifactRequest(new Request('https://assets.example/models/lc0/model-a.onnx', {
    headers: { 'Accept-Encoding': 'br, identity;q=0', Range: 'bytes=0-3' },
  }), fakeV2Env());
  assert.equal(range.status, 406);
  assert.equal(range.headers.get('Vary'), 'Accept-Encoding');
});
