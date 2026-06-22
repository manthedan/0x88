import assert from 'node:assert/strict';
import { test } from 'node:test';
import { handleArtifactRequest } from '../cloudflare/artifact-assets-worker.mjs';

const KEY = 'artifacts/sha256/abc/test.bin';
const ESCAPED_KEY = 'artifacts/sha256/abc/model%20v1.onnx';
const BODY = new TextEncoder().encode('abcdefghijklmnopqrstuvwxyz');

function fakeEnv() {
  const object = {
    size: BODY.byteLength,
    httpEtag: '"fake-etag"',
    httpMetadata: { contentType: 'application/octet-stream', cacheControl: 'public, max-age=31536000, immutable' },
  };
  return {
    TIMING_ALLOW_ORIGIN: 'https://0x88.app',
    ARTIFACTS: {
      async head(key) {
        return key === KEY || key === ESCAPED_KEY ? object : null;
      },
      async get(key, options) {
        if (key !== KEY && key !== ESCAPED_KEY) return null;
        const range = options?.range;
        const body = range ? BODY.slice(range.offset, range.offset + range.length) : BODY;
        return { ...object, body };
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
  assert.equal(response.headers.get('X-Artifact-Content-Length'), String(BODY.byteLength));
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=31536000, immutable');
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

test('artifact assets worker preserves percent-encoded R2 keys', async () => {
  const response = await handleArtifactRequest(new Request(`https://assets.example/${ESCAPED_KEY}`), fakeEnv());
  assert.equal(response.status, 200);
  assert.equal(await text(response), 'abcdefghijklmnopqrstuvwxyz');
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
