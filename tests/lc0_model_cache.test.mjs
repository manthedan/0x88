import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  clearLc0ModelCache,
  loadLc0ModelForOrt,
  sha256Hex,
  verifyLc0ModelBytes,
} from '../src/lc0/modelCache.ts';

// Known SHA-256 test vector: the empty input hashes to this digest.
const EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
// "abc" -> ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

function bytesOf(text) {
  return new TextEncoder().encode(text).buffer;
}

test('sha256Hex matches known vectors', async () => {
  assert.equal(await sha256Hex(new Uint8Array(0)), EMPTY_SHA256);
  assert.equal(await sha256Hex(bytesOf('abc')), ABC_SHA256);
});

test('verifyLc0ModelBytes accepts matching length and sha256', async () => {
  const check = await verifyLc0ModelBytes(bytesOf('abc'), { expectedBytes: 3, expectedSha256: ABC_SHA256 });
  assert.equal(check.ok, true);
  assert.equal(check.sha256Checked, true);
  assert.equal(check.sha256, ABC_SHA256);
});

test('verifyLc0ModelBytes rejects a byte-length mismatch before hashing', async () => {
  const check = await verifyLc0ModelBytes(bytesOf('abc'), { expectedBytes: 99, expectedSha256: ABC_SHA256 });
  assert.equal(check.ok, false);
  assert.equal(check.sha256Checked, false);
  assert.match(check.reason, /byte length mismatch/);
});

test('verifyLc0ModelBytes rejects a sha256 mismatch', async () => {
  const check = await verifyLc0ModelBytes(bytesOf('abc'), { expectedSha256: EMPTY_SHA256 });
  assert.equal(check.ok, false);
  assert.equal(check.sha256Checked, true);
  assert.match(check.reason, /sha256 mismatch/);
});

test('verifyLc0ModelBytes accepts bytes with no expected sha256 (length-only)', async () => {
  const check = await verifyLc0ModelBytes(bytesOf('abc'), { expectedBytes: 3 });
  assert.equal(check.ok, true);
  assert.equal(check.sha256Checked, false);
});

// --- loadLc0ModelForOrt integration with a fake Cache Storage + fetch ---

class FakeCache {
  constructor() { this.store = new Map(); }
  async match(req) { return this.store.get(req.url ?? req); }
  async put(req, res) { this.store.set(req.url ?? req, res); }
  async delete(req) { return this.store.delete(req.url ?? req); }
  async keys() { return [...this.store.keys()]; }
}

class FakeCacheStorage {
  constructor() { this.named = new Map(); }
  async open(name) { if (!this.named.has(name)) this.named.set(name, new FakeCache()); return this.named.get(name); }
  async has(name) { return this.named.has(name); }
  async delete(name) { return this.named.delete(name); }
}

const MODEL_URL = 'http://localhost/models/lc0/test.onnx';
const MODEL_ARTIFACT_URL = 'http://localhost/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx';
const MANIFEST_URL = 'http://localhost/models/lc0/manifest.json';
const CHANNEL_URL = 'http://localhost/channels/stable.json';
const RELEASE_URL = 'http://localhost/releases/test-release.json';

async function withMockedEnv(run, { serveBytes, manifestSha256, manifestBytes, manifestArtifactUrl, channelArtifacts }) {
  const prev = { caches: globalThis.caches, fetch: globalThis.fetch, location: globalThis.location };
  globalThis.caches = new FakeCacheStorage();
  globalThis.location = { href: 'http://localhost/' };
  const fetchLog = { model: 0, modelRequestCaches: [], urls: [] };
  globalThis.fetch = async (input) => {
    const url = typeof input === 'string' ? input : input.url;
    fetchLog.urls.push(url);
    if (url === MANIFEST_URL || url === '/models/lc0/manifest.json') {
      const manifest = { models: [{ file: 'test.onnx', url: MODEL_URL, artifactUrl: manifestArtifactUrl, bytes: manifestBytes, sha256: manifestSha256 }] };
      return new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } });
    }
    if (url === CHANNEL_URL) {
      return new Response(JSON.stringify({ schema: 'lc0_browser.artifact_channel_manifest.v1', releaseManifestUrl: '/releases/test-release.json' }), { headers: { 'content-type': 'application/json' } });
    }
    if (url === RELEASE_URL) {
      return new Response(JSON.stringify({ schema: 'lc0_browser.artifact_release_manifest.v1', artifacts: channelArtifacts ?? [] }), { headers: { 'content-type': 'application/json' } });
    }
    if (url === MODEL_URL || url === MODEL_ARTIFACT_URL) {
      fetchLog.model += 1;
      fetchLog.modelRequestCaches.push(typeof input === 'string' ? undefined : input.cache);
      const served = serveBytes(input);
      return served instanceof Response ? served : new Response(served);
    }
    return new Response(null, { status: 404 });
  };
  try {
    return await run(fetchLog);
  } finally {
    globalThis.caches = prev.caches;
    globalThis.fetch = prev.fetch;
    globalThis.location = prev.location;
  }
}

test('loadLc0ModelForOrt caches on miss then serves a validated hit', async () => {
  await withMockedEnv(async (fetchLog) => {
    const miss = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
    assert.equal(miss.cacheStatus, 'miss');
    assert.equal(miss.sha256Valid, true);
    assert.equal(miss.sha256, ABC_SHA256);
    assert.equal(miss.telemetry.source, 'network');
    assert.equal(miss.telemetry.requestCache, 'force-cache');
    assert.equal(miss.telemetry.preallocatedDownload, true);
    assert.equal(typeof miss.telemetry.downloadMs, 'number');
    assert.equal(typeof miss.telemetry.hashMs, 'number');
    assert.equal(typeof miss.telemetry.cacheWriteMs, 'number');
    assert.equal(fetchLog.model, 1);
    assert.deepEqual(fetchLog.modelRequestCaches, ['force-cache']);

    const hit = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
    assert.equal(hit.cacheStatus, 'hit');
    assert.equal(hit.sha256Valid, true);
    assert.equal(hit.telemetry.source, 'cache-storage');
    assert.equal(typeof hit.telemetry.cacheReadMs, 'number');
    assert.equal(typeof hit.telemetry.hashMs, 'number');
    assert.equal(fetchLog.model, 1, 'a validated cache hit does not refetch');
  }, { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 });
});

test('loadLc0ModelForOrt resolves stable model URLs through manifest artifactUrl', async () => {
  await withMockedEnv(async (fetchLog) => {
    const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: '/models/lc0/manifest.json' });
    assert.equal(direct.mode, 'url');
    assert.equal(direct.model, MODEL_ARTIFACT_URL);
    assert.equal(direct.url, MODEL_ARTIFACT_URL);
    assert.equal(direct.logicalUrl, MODEL_URL);
    assert.equal(fetchLog.model, 0, 'URL mode resolves the immutable URL without downloading bytes');

    const cached = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
    assert.equal(cached.url, MODEL_ARTIFACT_URL);
    assert.equal(cached.logicalUrl, MODEL_URL);
    assert.equal(cached.telemetry.requestCache, 'force-cache');
    assert.deepEqual(fetchLog.modelRequestCaches, ['force-cache']);
  }, { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3, manifestArtifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx' });
});

test('loadLc0ModelForOrt keeps default model manifest lookups on the app shell origin', async () => {
  await withMockedEnv(async (fetchLog) => {
    const assetModelUrl = 'https://assets.0x88.app/models/lc0/test.onnx';
    const direct = await loadLc0ModelForOrt(assetModelUrl, { cache: false, channelUrl: '/channels/stable.json' });
    assert.equal(direct.model, MODEL_ARTIFACT_URL);
    assert(fetchLog.urls.includes('/models/lc0/manifest.json'));
    assert(!fetchLog.urls.includes('https://assets.0x88.app/models/lc0/manifest.json'));
  }, {
    serveBytes: () => bytesOf('abc'),
    manifestSha256: ABC_SHA256,
    manifestBytes: 3,
    channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
});

test('loadLc0ModelForOrt resolves stable model URLs through channel release manifests when configured', async () => {
  await withMockedEnv(async (fetchLog) => {
    const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: MANIFEST_URL, channelUrl: '/channels/stable.json' });
    assert.equal(direct.model, MODEL_ARTIFACT_URL);
    assert.equal(direct.logicalUrl, MODEL_URL);

    const cached = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
    assert.equal(cached.url, MODEL_ARTIFACT_URL);
    assert.equal(cached.logicalUrl, MODEL_URL);
    assert.deepEqual(fetchLog.modelRequestCaches, ['force-cache']);
  }, {
    serveBytes: () => bytesOf('abc'),
    manifestSha256: ABC_SHA256,
    manifestBytes: 3,
    channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
});

test('loadLc0ModelForOrt validates channel artifacts against release metadata', async () => {
  await withMockedEnv(async () => {
    const result = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
    assert.equal(result.sha256Valid, true);
    assert.equal(result.sha256, ABC_SHA256);
    assert.equal(result.expectedSha256, ABC_SHA256);
  }, {
    serveBytes: () => bytesOf('abc'),
    manifestSha256: EMPTY_SHA256,
    manifestBytes: 0,
    channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
});

test('loadLc0ModelForOrt resolves artifact URLs even when Cache Storage is unavailable', async () => {
  await withMockedEnv(async () => {
    const savedCaches = globalThis.caches;
    try {
      delete globalThis.caches;
      const result = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
      assert.equal(result.cacheStatus, 'unavailable');
      assert.equal(result.model, MODEL_ARTIFACT_URL);
      assert.equal(result.logicalUrl, MODEL_URL);
    } finally {
      globalThis.caches = savedCaches;
    }
  }, {
    serveBytes: () => bytesOf('abc'),
    manifestSha256: ABC_SHA256,
    manifestBytes: 3,
    channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
});

test('loadLc0ModelForOrt skips Cache Storage admission when quota is too low', async () => {
  await withMockedEnv(async (fetchLog) => {
    const prevNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    let persistCalls = 0;
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        storage: {
          estimate: async () => ({ usage: 900, quota: 1000 }),
          persist: async () => { persistCalls += 1; return true; },
        },
      },
    });
    try {
      const result = await loadLc0ModelForOrt(MODEL_URL, {
        cache: true,
        manifestUrl: MANIFEST_URL,
        requestPersistentStorage: true,
        minimumFreeBytesAfterCache: 128,
      });
      assert.equal(result.mode, 'url');
      assert.equal(result.cacheStatus, 'quota-limited');
      assert.equal(result.model, MODEL_URL);
      assert.equal(fetchLog.model, 0, 'quota fallback does not fetch bytes into memory');
      assert.equal(persistCalls, 1);
    } finally {
      if (prevNavigator) Object.defineProperty(globalThis, 'navigator', prevNavigator);
      else delete globalThis.navigator;
    }
  }, { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 });
});

test('loadLc0ModelForOrt rejects a corrupt download and does not cache it', async () => {
  await withMockedEnv(async (fetchLog) => {
    await assert.rejects(
      () => loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL }),
      /validation failed/,
    );
    assert.equal(fetchLog.model, 1);
    const cache = await globalThis.caches.open('lc0-browser-models-v1');
    assert.equal(await cache.match(MODEL_URL), undefined, 'corrupt bytes are not persisted');
  }, { serveBytes: () => bytesOf('xyz'), manifestSha256: ABC_SHA256, manifestBytes: 3 });
});

test('loadLc0ModelForOrt evicts and revalidates a stale cache entry when the model content changes', async () => {
  // Seed the cache with "abc" under the original manifest.
  await withMockedEnv(async () => {
    const seed = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
    assert.equal(seed.cacheStatus, 'miss');
    // The seeded cache lives on globalThis.caches; reuse it for the next phase.
    const seededCaches = globalThis.caches;

    // Now the model content changes: the manifest sha256 is bumped and the
    // server returns the new bytes. The stale cached "abc" must be evicted.
    let phase = 0;
    const requestCaches = [];
    const prevFetch = globalThis.fetch;
    globalThis.fetch = async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === MANIFEST_URL) {
        const manifest = { models: [{ file: 'test.onnx', url: MODEL_URL, bytes: 4, sha256: await sha256Hex(bytesOf('abcd')) }] };
        return new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } });
      }
      if (url === MODEL_URL) {
        phase += 1;
        requestCaches.push(typeof input === 'string' ? undefined : input.cache);
        return new Response(bytesOf('abcd'));
      }
      return new Response(null, { status: 404 });
    };
    globalThis.caches = seededCaches;

    const revalidated = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
    assert.equal(revalidated.cacheStatus, 'miss');
    assert.equal(revalidated.revalidated, true);
    assert.equal(revalidated.sha256, await sha256Hex(bytesOf('abcd')));
    assert.equal(revalidated.telemetry.requestCache, 'reload');
    assert.deepEqual(requestCaches, ['reload'], 'stale-cache recovery bypasses the browser HTTP cache');
    assert.equal(phase, 1, 'the new content is fetched exactly once after eviction');
    globalThis.fetch = prevFetch;
  }, { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 });
});

test('loadLc0ModelForOrt streams progress downloads into a preallocated buffer', async () => {
  const progress = [];
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('a'));
      controller.enqueue(new TextEncoder().encode('b'));
      controller.enqueue(new TextEncoder().encode('c'));
      controller.close();
    },
  });
  await withMockedEnv(async () => {
    const loaded = await loadLc0ModelForOrt(MODEL_URL, {
      cache: false,
      manifestUrl: MANIFEST_URL,
      onProgress: (loadedBytes, totalBytes) => progress.push([loadedBytes, totalBytes]),
    });
    assert.equal(loaded.mode, 'memory');
    assert.equal(loaded.cacheStatus, 'disabled');
    assert.equal(loaded.telemetry.source, 'memory');
    assert.equal(loaded.telemetry.preallocatedDownload, true);
    assert.equal(typeof loaded.telemetry.downloadMs, 'number');
    assert.deepEqual(new Uint8Array(loaded.model), new TextEncoder().encode('abc'));
    assert.deepEqual(progress, [[0, 3], [1, 3], [2, 3], [3, 3]]);
  }, {
    serveBytes: () => new Response(stream, { headers: { 'content-length': '3' } }),
    manifestSha256: ABC_SHA256,
    manifestBytes: 3,
  });
});

test('clearLc0ModelCache reports removed entries and is safe when empty', async () => {
  await withMockedEnv(async () => {
    await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
    const cleared = await clearLc0ModelCache();
    assert.equal(cleared.cleared, true);
    assert.equal(cleared.removedEntries, 1);
    const again = await clearLc0ModelCache();
    assert.equal(again.cleared, false);
    assert.equal(again.removedEntries, 0);
  }, { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 });
});
