import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLc0OnnxEvaluatorFromUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import {
  clearLc0ModelCache,
  clearLc0ModelCacheEntry,
  inspectLc0ModelDownload,
  loadLc0ModelForOrt,
  modelFileName,
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

test('verifyLc0ModelBytes requires an explicit opt-out for length-only checks', async () => {
  const check = await verifyLc0ModelBytes(bytesOf('abc'), { expectedBytes: 3 });
  assert.equal(check.ok, false);
  assert.equal(check.sha256Checked, false);
  assert.equal((await verifyLc0ModelBytes(bytesOf('abc'), { expectedBytes: 3 }, true)).ok, true);
});

// --- loadLc0ModelForOrt integration with a fake Cache Storage + fetch ---

class FakeCache {
  constructor() {
    this.store = new Map();
  }
  async match(req) {
    return this.store.get(req.url ?? req)?.clone();
  }
  async put(req, res) {
    this.store.set(req.url ?? req, res);
  }
  async delete(req) {
    return this.store.delete(req.url ?? req);
  }
  async keys(request) {
    return [...this.store.keys()].filter((url) => !request || url === request.url).map((url) => new Request(url));
  }
}

class FakeCacheStorage {
  constructor() {
    this.named = new Map();
  }
  async open(name) {
    if (!this.named.has(name)) this.named.set(name, new FakeCache());
    return this.named.get(name);
  }
  async has(name) {
    return this.named.has(name);
  }
  async delete(name) {
    return this.named.delete(name);
  }
}

class MemoryMetadataStore {
  constructor() {
    this.entries = new Map();
  }
  async get(key) {
    return this.entries.get(key);
  }
  async put(entry) {
    this.entries.set(entry.key, { ...entry });
  }
  async delete(key) {
    this.entries.delete(key);
  }
  async list(cacheName) {
    return [...this.entries.values()].filter((entry) => entry.cacheName === cacheName);
  }
  async clear(cacheName) {
    for (const [key, entry] of this.entries) if (entry.cacheName === cacheName) this.entries.delete(key);
  }
}

const MODEL_URL = 'http://localhost/models/lc0/test.onnx';
const MODEL_ARTIFACT_URL = 'http://localhost/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx';
const MANIFEST_URL = 'http://localhost/models/lc0/manifest.json';
let fixtureNumber = 0;
let CHANNEL_URL;
let RELEASE_URL;
const MODEL_V2_BR_URL = `http://localhost/artifacts/sha256/${ABC_SHA256}/br/${'b'.repeat(64)}`;

async function withMockedEnv(
  run,
  { serveBytes, manifestSha256, manifestBytes, manifestArtifactUrl, channelArtifacts, channelManifest, releaseManifest, artifactUrls = [] },
) {
  const prev = { caches: globalThis.caches, fetch: globalThis.fetch, location: globalThis.location };
  CHANNEL_URL = `http://localhost/channels/stable.json?fixture=${++fixtureNumber}`;
  RELEASE_URL = `http://localhost/releases/test-release-${fixtureNumber}.json`;
  globalThis.caches = new FakeCacheStorage();
  globalThis.location = { href: 'http://localhost/' };
  const fetchLog = { manifest: 0, model: 0, modelRequestCaches: [], modelCacheModes: [], urls: [] };
  globalThis.fetch = async (input) => {
    const url = typeof input === 'string' ? input : input.url;
    fetchLog.urls.push(url);
    if (url === MANIFEST_URL || url === '/models/lc0/manifest.json') {
      fetchLog.manifest += 1;
      const manifest = { models: [{ file: 'test.onnx', url: MODEL_URL, artifactUrl: manifestArtifactUrl, bytes: manifestBytes, sha256: manifestSha256 }] };
      return new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } });
    }
    if (url === CHANNEL_URL) {
      return new Response(
        JSON.stringify(
          channelManifest
            ? { ...channelManifest, releaseUrl: new URL(RELEASE_URL).pathname }
            : { schema: 'lc0_browser.artifact_channel_manifest.v1', releaseManifestUrl: new URL(RELEASE_URL).pathname },
        ),
        { headers: { 'content-type': 'application/json' } },
      );
    }
    if (url === RELEASE_URL) {
      return new Response(JSON.stringify(releaseManifest ?? { schema: 'lc0_browser.artifact_release_manifest.v1', artifacts: channelArtifacts ?? [] }), {
        headers: { 'content-type': 'application/json' },
      });
    }
    if (url === MODEL_URL || url === MODEL_ARTIFACT_URL || artifactUrls.includes(url)) {
      fetchLog.model += 1;
      const requestCache = typeof input === 'string' ? undefined : input.cache;
      fetchLog.modelRequestCaches.push(requestCache);
      fetchLog.modelCacheModes.push(requestCache);
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

test('offline lookup restores a verified immutable model, but never trusts replaced bodies or downloads with a stale pin', async () => {
  const options = { cache: true, manifestUrl: MANIFEST_URL, channelUrl: null, metadataStore: new MemoryMetadataStore(), requireManifestEntry: true };
  await withMockedEnv(
    async () => {
      const onlineFetch = globalThis.fetch;
      assert.equal((await loadLc0ModelForOrt(MODEL_URL, options)).cacheStatus, 'miss');
      const cache = await caches.open('lc0-browser-models-v1');
      globalThis.fetch = async (url) => {
        assert.equal(url, MANIFEST_URL, 'offline pins must never authorize a binary fetch');
        throw new TypeError('offline');
      };
      assert.equal((await inspectLc0ModelDownload(MODEL_URL, options)).cached, true);
      const offline = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(offline.url, MODEL_ARTIFACT_URL);
      assert.equal(offline.verification, 'hashed');
      assert.equal(new TextDecoder().decode(offline.model), 'abc');

      globalThis.fetch = async (url) => (url === MANIFEST_URL ? Response.json({ models: [] }) : onlineFetch(url));
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, options), /App-origin model integrity metadata is unavailable/);
      globalThis.fetch = async (url) => {
        assert.equal(url, MANIFEST_URL);
        throw new TypeError('offline');
      };
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, { ...options, forceReload: true }), /App-origin model integrity metadata is unavailable/);
      await cache.put(MODEL_ARTIFACT_URL, new Response('bad'));
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, options), /No verified cached model available offline/);
      assert.equal(await cache.match(MODEL_ARTIFACT_URL), undefined);
      await cache.put(MODEL_ARTIFACT_URL, new Response('abc'));
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, options), /App-origin model integrity metadata is unavailable/);
    },
    { serveBytes: () => bytesOf('abc'), manifestArtifactUrl: MODEL_ARTIFACT_URL, manifestBytes: 3, manifestSha256: ABC_SHA256 },
  );
});

test('shipped pins recover old content-addressed records offline without trusting unrelated models or corrupt cached bytes', async (t) => {
  globalThis.__SHIPPED_MODEL_PINS__ = {
    '/models/lc0/manifest.json': { '/models/lc0/test.onnx': { expectedSha256: ABC_SHA256, expectedBytes: 3 } },
  };
  t.after(() => delete globalThis.__SHIPPED_MODEL_PINS__);
  const metadataStore = new MemoryMetadataStore();
  const options = { cache: true, manifestUrl: MANIFEST_URL, channelUrl: null, metadataStore, requireManifestEntry: true };
  await withMockedEnv(
    async () => {
      await loadLc0ModelForOrt(MODEL_URL, options);
      const legacy = [...metadataStore.entries.values()][0];
      delete legacy.provenance;
      assert.equal(legacy.url, MODEL_ARTIFACT_URL);
      const requests = [];
      globalThis.fetch = async (input) => {
        requests.push(input);
        throw new TypeError('offline');
      };
      assert.equal((await inspectLc0ModelDownload(MODEL_URL, options)).cached, true);
      const restored = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(restored.verification, 'hashed');
      assert.equal(restored.url, MODEL_ARTIFACT_URL);
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL.replace('test.onnx', 'unknown.onnx'), options), /App-origin model integrity metadata is unavailable/);
      await (await caches.open('lc0-browser-models-v1')).put(MODEL_ARTIFACT_URL, new Response('bad'));
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, options), /No verified cached model available offline/);
      assert.ok(requests.every((input) => input === MANIFEST_URL));
    },
    { serveBytes: () => bytesOf('abc'), manifestArtifactUrl: MODEL_ARTIFACT_URL, manifestBytes: 3, manifestSha256: ABC_SHA256 },
  );
});

test('exact manifest URL pins precede filename fallback and tolerate non-UTF-8 URL octets', async () => {
  const url = MODEL_URL.replace('test.onnx', '%E9.onnx');
  assert.equal(modelFileName(url), '%E9.onnx');
  await withMockedEnv(
    async () => {
      const original = globalThis.fetch;
      let bodies = 0;
      globalThis.fetch = async (input, options) => {
        const requested = input.url ?? String(input);
        if (requested === MANIFEST_URL)
          return Response.json({
            models: [
              { file: '%E9.onnx', url: MODEL_URL, bytes: 3, sha256: '0'.repeat(64) },
              { file: 'legacy-name.onnx', url, bytes: 3, sha256: ABC_SHA256 },
            ],
          });
        if (requested === url) {
          bodies++;
          return new Response('abc');
        }
        return original(input, options);
      };
      const loaded = await loadLc0ModelForOrt(url, { cache: false, manifestUrl: MANIFEST_URL, channelUrl: null });
      assert.equal(loaded.sha256Valid, true);
      assert.equal(new TextDecoder().decode(loaded.model), 'abc');
      assert.equal(bodies, 1);
    },
    { serveBytes: () => bytesOf('abc'), manifestBytes: 3, manifestSha256: ABC_SHA256 },
  );
});

test('loadLc0ModelForOrt caches on miss then serves a validated hit', async () => {
  await withMockedEnv(
    async (fetchLog) => {
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
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('loadLc0ModelForOrt resolves stable model URLs through manifest artifactUrl', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: '/models/lc0/manifest.json' });
      assert.equal(direct.mode, 'memory');
      assert.deepEqual(direct.model, bytesOf('abc'));
      assert.equal(direct.sha256Valid, true);
      assert.equal(direct.url, MODEL_ARTIFACT_URL);
      assert.equal(direct.logicalUrl, MODEL_URL);
      assert.equal(fetchLog.model, 1, 'cache-disabled loads verify the immutable bytes before returning them');

      const cached = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
      assert.equal(cached.url, MODEL_ARTIFACT_URL);
      assert.equal(cached.logicalUrl, MODEL_URL);
      assert.equal(cached.telemetry.requestCache, 'force-cache');
      assert.deepEqual(fetchLog.modelRequestCaches, ['default', 'force-cache']);
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      manifestArtifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx',
    },
  );
});

test('loadLc0ModelForOrt keeps default model manifest lookups on the app shell origin', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const assetModelUrl = 'https://assets.0x88.app/models/lc0/test.onnx';
      const direct = await loadLc0ModelForOrt(assetModelUrl, { cache: false, channelUrl: CHANNEL_URL.replace('http://localhost', '') });
      assert.equal(direct.url, MODEL_ARTIFACT_URL);
      assert.equal(direct.sha256Valid, true);
      assert(fetchLog.urls.includes('/models/lc0/manifest.json'));
      assert(!fetchLog.urls.includes('https://assets.0x88.app/models/lc0/manifest.json'));
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      channelArtifacts: [
        {
          logicalUrl: '/models/lc0/test.onnx',
          artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx',
          bytes: 3,
          sha256: ABC_SHA256,
        },
      ],
    },
  );
});

test('loadLc0ModelForOrt resolves stable model URLs through channel release manifests when configured', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL.replace('http://localhost', '') });
      assert.equal(direct.url, MODEL_ARTIFACT_URL);
      assert.equal(direct.sha256Valid, true);
      assert.equal(direct.logicalUrl, MODEL_URL);

      const cached = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
      assert.equal(cached.url, MODEL_ARTIFACT_URL);
      assert.equal(cached.logicalUrl, MODEL_URL);
      assert.deepEqual(fetchLog.modelRequestCaches, ['default', 'force-cache']);
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      channelArtifacts: [
        {
          logicalUrl: '/models/lc0/test.onnx',
          artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx',
          bytes: 3,
          sha256: ABC_SHA256,
        },
      ],
    },
  );
});

test('concurrent and subsequent lookups share channel and release fetches; channels expire independently', async (t) => {
  let now = Date.now();
  t.mock.method(Date, 'now', () => now);
  await withMockedEnv(
    async (fetchLog) => {
      const options = { manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL };
      const inspect = () => inspectLc0ModelDownload(MODEL_URL, options);
      await Promise.all([inspect(), inspect(), inspect()]);
      await inspect();
      const count = (url) => fetchLog.urls.filter((requested) => requested === url).length;
      assert.equal(count(CHANNEL_URL), 1);
      assert.equal(count(RELEASE_URL), 1);
      now += 31_000;
      await inspect();
      assert.equal(count(CHANNEL_URL), 2);
      assert.equal(count(RELEASE_URL), 1);
      now += 31_000;
      await inspect();
      assert.equal(count(CHANNEL_URL), 3);
      assert.equal(count(RELEASE_URL), 2);
    },
    { manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('loadLc0ModelForOrt resolves v2 channels to immutable Brotli bodies with decoded integrity', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
      assert.equal(direct.url, MODEL_V2_BR_URL);
      assert.equal(direct.sha256Valid, true);

      const cached = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
      assert.equal(cached.url, MODEL_V2_BR_URL);
      assert.equal(cached.sha256Valid, true);
      assert.deepEqual(fetchLog.modelRequestCaches, ['default', 'force-cache']);
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      artifactUrls: [MODEL_V2_BR_URL],
      channelManifest: { schema: 'lc0-webgpu.artifact-channel.v2', releaseUrl: '/releases/test-release.json' },
      releaseManifest: {
        schema: 'lc0-webgpu.artifact-release.v2',
        artifacts: [
          {
            name: 'test-model',
            file: 'test.onnx',
            raw: { bytes: 3, sha256: ABC_SHA256 },
            representations: [
              { encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 },
              { encoding: 'br', url: `/artifacts/sha256/${ABC_SHA256}/br/${'b'.repeat(64)}`, bytes: 2, sha256: 'b'.repeat(64) },
            ],
          },
        ],
      },
    },
  );
});

test('loadLc0ModelForOrt does not resolve ambiguous name-only v2 basenames', async () => {
  await withMockedEnv(
    async () => {
      const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
      assert.equal(direct.url, MODEL_URL);
      assert.equal(direct.sha256Valid, true);
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      channelManifest: { schema: 'lc0-webgpu.artifact-channel.v2', releaseUrl: '/releases/test-release.json' },
      releaseManifest: {
        schema: 'lc0-webgpu.artifact-release.v2',
        artifacts: ['one', 'two'].map((name) => ({
          name,
          file: 'test.onnx',
          raw: { bytes: 3, sha256: ABC_SHA256 },
          representations: [{ encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 }],
        })),
      },
    },
  );
});

test('loadLc0ModelForOrt rejects channel metadata that contradicts the app manifest', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      await assert.rejects(
        loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL }),
        /channel artifact integrity mismatch.*model manifest/,
      );
      assert.equal(fetchLog.model, 0, 'neither the artifact nor a fallback body is fetched');
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: EMPTY_SHA256,
      manifestBytes: 0,
      channelArtifacts: [
        {
          logicalUrl: '/models/lc0/test.onnx',
          artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx',
          bytes: 3,
          sha256: ABC_SHA256,
        },
      ],
    },
  );
});

for (const version of [1, 2]) {
  for (const conflict of ['sha256', 'bytes']) {
    for (const cache of [false, true]) {
      test(`v${version} channel ${conflict} conflict fails closed before body/cache access (cache=${cache})`, async () => {
        const bytes = conflict === 'bytes' ? 4 : 3;
        const sha256 = conflict === 'sha256' ? EMPTY_SHA256 : ABC_SHA256;
        await withMockedEnv(
          async (fetchLog) => {
            await assert.rejects(
              loadLc0ModelForOrt(MODEL_URL, { cache, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL }),
              /channel artifact integrity mismatch/,
            );
            assert.equal(fetchLog.model, 0);
            assert.equal(caches.named.size, 0, 'mismatching metadata cannot consult or poison the cache');
          },
          {
            serveBytes: () => {
              throw new Error('must not fetch a conflicting body');
            },
            manifestSha256: ABC_SHA256,
            manifestBytes: 3,
            channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: MODEL_ARTIFACT_URL, bytes, sha256 }],
            ...(version === 2
              ? {
                  channelManifest: { releaseUrl: '/releases/test-release.json' },
                  releaseManifest: {
                    artifacts: [
                      {
                        logicalUrl: '/models/lc0/test.onnx',
                        raw: { bytes, sha256 },
                        representations: [{ encoding: 'br', url: MODEL_V2_BR_URL, bytes: 2, sha256: 'b'.repeat(64) }],
                      },
                    ],
                  },
                }
              : {}),
          },
        );
      });
    }
  }
}

test('missing legacy channel pins inherit app pins, and matching hash case is normalized for comparison', async () => {
  for (const pins of [{}, { sha256: ABC_SHA256.toUpperCase(), bytes: 3 }]) {
    await withMockedEnv(
      async () => {
        const loaded = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
        assert.equal(loaded.expectedSha256, ABC_SHA256);
        assert.equal(loaded.expectedBytes, 3);
        assert.equal(loaded.sha256Valid, true);
        assert.equal(loaded.url, MODEL_ARTIFACT_URL);
      },
      {
        serveBytes: () => bytesOf('abc'),
        manifestSha256: ABC_SHA256,
        manifestBytes: 3,
        channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: MODEL_ARTIFACT_URL, ...pins }],
      },
    );
  }
});

test('a corrupt channel body cannot pass inherited app pins or enter the cache', async () => {
  await withMockedEnv(
    async () => {
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL }), /sha256 mismatch/);
      assert.equal((await caches.open('lc0-browser-models-v1')).store.size, 0);
    },
    {
      serveBytes: () => bytesOf('bad'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      channelArtifacts: [{ logicalUrl: '/models/lc0/test.onnx', artifactUrl: MODEL_ARTIFACT_URL }],
    },
  );
});

test('metadata-trusted parse corruption evicts only that entry, refetches with reload once, and preserves a zero-network warm hit', async (t) => {
  t.mock.method(console, 'warn', () => {});
  await withMockedEnv(
    async (fetchLog) => {
      const metadataStore = new MemoryMetadataStore();
      const options = { cache: true, manifestUrl: MANIFEST_URL, metadataStore };
      await loadLc0ModelForOrt(MODEL_URL, options);
      const cache = await caches.open('lc0-browser-models-v1');
      const unrelated = 'http://localhost/models/lc0/other.onnx';
      await cache.put(unrelated, new Response(bytesOf('keep')));
      await cache.put(MODEL_URL, new Response(bytesOf('bad'))); // Same length; metadata still attests the old verified body.
      const creations = [];
      const result = await createLc0OnnxEvaluatorFromUrl(
        MODEL_URL,
        { load: options },
        {
          placement: async () => ({ providers: ['wasm'] }),
          load: loadLc0ModelForOrt,
          createEvaluator: async (model) => {
            const text = new TextDecoder().decode(model);
            creations.push(text);
            if (text === 'bad') throw new Error('ERROR_CODE: 7, ERROR_MESSAGE: Protobuf parsing failed');
            return { async dispose() {} };
          },
        },
      );
      assert.deepEqual(creations, ['bad', 'abc']);
      assert.equal(result.modelLoad.revalidated, true);
      assert.equal(result.modelLoad.verification, 'hashed');
      assert.equal(result.modelLoad.sha256Valid, true);
      assert.equal(fetchLog.model, 2);
      assert.deepEqual(fetchLog.modelRequestCaches, ['force-cache', 'reload']);
      assert.equal(await (await cache.match(unrelated)).text(), 'keep', 'other cached models survive');
      const warm = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(warm.verification, 'metadata');
      assert.equal(fetchLog.model, 2, 'repaired warm load performs zero model downloads');
      assert.equal(metadataStore.entries.size, 1);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('a corrupt recovery download fails SHA-256 before a second create and is not cached', async (t) => {
  t.mock.method(console, 'warn', () => {});
  let downloads = 0,
    creates = 0;
  await withMockedEnv(
    async (fetchLog) => {
      const metadataStore = new MemoryMetadataStore();
      const load = { cache: true, manifestUrl: MANIFEST_URL, metadataStore };
      await loadLc0ModelForOrt(MODEL_URL, load);
      const cache = await caches.open('lc0-browser-models-v1');
      await cache.put(MODEL_URL, new Response(bytesOf('bad')));
      await assert.rejects(
        createLc0OnnxEvaluatorFromUrl(
          MODEL_URL,
          { load },
          {
            placement: async () => ({ providers: ['wasm'] }),
            load: loadLc0ModelForOrt,
            createEvaluator: async () => {
              creates++;
              throw new Error('INVALID_PROTOBUF');
            },
          },
        ),
        /sha256 mismatch/,
      );
      assert.equal(creates, 1);
      assert.equal(fetchLog.model, 2);
      assert.equal(cache.store.size, 0, 'neither poisoned old bytes nor corrupt fresh bytes remain cached');
      assert.equal(metadataStore.entries.size, 0);
    },
    { serveBytes: () => bytesOf(++downloads === 1 ? 'abc' : 'bad'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('forced recovery still uses HTTP reload and hashes when Cache Storage is unavailable', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      delete globalThis.caches;
      const fresh = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, forceReload: true });
      assert.equal(fresh.verification, 'hashed');
      assert.equal(fresh.sha256Valid, true);
      assert.deepEqual(fetchLog.modelRequestCaches, ['reload']);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('forceReload bypasses both the Cache Storage body and a coalesced ordinary flight', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const options = { cache: true, manifestUrl: MANIFEST_URL };
      await loadLc0ModelForOrt(MODEL_URL, options);
      const [warm, fresh] = await Promise.all([loadLc0ModelForOrt(MODEL_URL, options), loadLc0ModelForOrt(MODEL_URL, { ...options, forceReload: true })]);
      assert.equal(warm.cacheStatus, 'hit');
      assert.equal(fresh.cacheStatus, 'miss');
      assert.equal(fresh.verification, 'hashed');
      assert.deepEqual(fetchLog.modelRequestCaches, ['force-cache', 'reload']);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('loadLc0ModelForOrt resolves artifact URLs even when Cache Storage is unavailable', async () => {
  await withMockedEnv(
    async () => {
      const savedCaches = globalThis.caches;
      try {
        delete globalThis.caches;
        const result = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, channelUrl: CHANNEL_URL });
        assert.equal(result.cacheStatus, 'unavailable');
        assert.equal(result.url, MODEL_ARTIFACT_URL);
        assert.equal(result.sha256Valid, true);
        assert.equal(result.logicalUrl, MODEL_URL);
      } finally {
        globalThis.caches = savedCaches;
      }
    },
    {
      serveBytes: () => bytesOf('abc'),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
      channelArtifacts: [
        {
          logicalUrl: '/models/lc0/test.onnx',
          artifactUrl: '/artifacts/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad/test.onnx',
          bytes: 3,
          sha256: ABC_SHA256,
        },
      ],
    },
  );
});

test('loadLc0ModelForOrt skips Cache Storage admission when quota is too low', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const prevNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
      let persistCalls = 0;
      Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: {
          storage: {
            estimate: async () => ({ usage: 900, quota: 1000 }),
            persist: async () => {
              persistCalls += 1;
              return true;
            },
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
        assert.equal(result.mode, 'memory');
        assert.equal(result.cacheStatus, 'quota-limited');
        assert.deepEqual(result.model, bytesOf('abc'));
        assert.equal(result.sha256Valid, true);
        assert.equal(fetchLog.model, 1, 'quota fallback still verifies the bytes in memory');
        assert.equal(persistCalls, 0);
      } finally {
        if (prevNavigator) Object.defineProperty(globalThis, 'navigator', prevNavigator);
        else delete globalThis.navigator;
      }
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('concurrent model loads share manifest, download, and verification work', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const firstProgress = [];
      const secondProgress = [];
      const [first, second] = await Promise.all([
        loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, onProgress: (loaded) => firstProgress.push(loaded) }),
        loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, onProgress: (loaded) => secondProgress.push(loaded) }),
      ]);
      assert.equal(first.cacheStatus, 'miss');
      assert.equal(second.cacheStatus, 'miss');
      assert.equal(fetchLog.manifest, 1);
      assert.equal(fetchLog.model, 1);
      assert.strictEqual(first.model, second.model, 'in-flight callers share the verified byte buffer');
      assert.deepEqual(firstProgress, [0, 3]);
      assert.deepEqual(secondProgress, firstProgress, 'all in-flight callers receive shared download progress');

      const warm = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
      assert.equal(warm.cacheStatus, 'hit');
      assert.equal(fetchLog.model, 1);
      assert.equal(fetchLog.manifest, 2, 'settled single-flight state does not pin mutable manifests forever');
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('metadata-trusted warm hit skips the full SHA-256 pass', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const metadataStore = new MemoryMetadataStore();
      const options = { cache: true, manifestUrl: MANIFEST_URL, metadataStore };
      const miss = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(miss.verification, 'hashed');
      const hit = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(hit.cacheStatus, 'hit');
      assert.equal(hit.verification, 'metadata');
      assert.equal(hit.hashMs, 0);
      assert.equal(hit.sha256, ABC_SHA256);
      assert.equal(fetchLog.model, 1);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('loadLc0ModelForOrt rejects a corrupt download and does not cache it', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      await assert.rejects(() => loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL }), /validation failed/);
      assert.equal(fetchLog.model, 1);
      const cache = await globalThis.caches.open('lc0-browser-models-v1');
      assert.equal(await cache.match(MODEL_URL), undefined, 'corrupt bytes are not persisted');
    },
    { serveBytes: () => bytesOf('xyz'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('loadLc0ModelForOrt evicts and revalidates a stale cache entry when the model content changes', async () => {
  // Seed the cache with "abc" under the original manifest.
  await withMockedEnv(
    async () => {
      const seed = await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
      assert.equal(seed.cacheStatus, 'miss');
      // The seeded cache lives on globalThis.caches; reuse it for the next phase.
      const seededCaches = globalThis.caches;

      // Now the model content changes: the manifest sha256 is bumped and the
      // server returns the new bytes. The stale cached "abc" must be evicted.
      let phase = 0;
      const requestCaches = [];
      let recoveryCacheMode;
      const prevFetch = globalThis.fetch;
      globalThis.fetch = async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === MANIFEST_URL) {
          const manifest = { models: [{ file: 'test.onnx', url: MODEL_URL, bytes: 4, sha256: await sha256Hex(bytesOf('abcd')) }] };
          return new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } });
        }
        if (url === MODEL_URL) {
          phase += 1;
          recoveryCacheMode = typeof input === 'string' ? undefined : input.cache;
          requestCaches.push(recoveryCacheMode);
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
      assert.equal(recoveryCacheMode, 'reload', 'corruption recovery bypasses stale HTTP cache content');
      globalThis.fetch = prevFetch;
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
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
  await withMockedEnv(
    async () => {
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
      assert.deepEqual(progress, [
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 3],
      ]);
    },
    {
      serveBytes: () => new Response(stream, { headers: { 'content-length': '3' } }),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
    },
  );
});

test('loadLc0ModelForOrt aborts caller-owned streamed downloads', async () => {
  const controller = new AbortController();
  const progress = [];
  await withMockedEnv(
    async () => {
      await assert.rejects(
        () =>
          loadLc0ModelForOrt(MODEL_URL, {
            cache: false,
            manifestUrl: MANIFEST_URL,
            signal: controller.signal,
            onProgress: (loadedBytes, totalBytes) => {
              progress.push([loadedBytes, totalBytes]);
              if (loadedBytes === 0) controller.abort();
            },
          }),
        (error) => error?.name === 'AbortError',
      );
      assert.deepEqual(progress, [[0, 3]]);
    },
    {
      serveBytes: (input) =>
        new Response(
          new ReadableStream({
            start(streamController) {
              input.signal.addEventListener('abort', () => streamController.error(input.signal.reason), { once: true });
            },
          }),
          { headers: { 'content-length': '3' } },
        ),
      manifestSha256: ABC_SHA256,
      manifestBytes: 3,
    },
  );
});

test('loadLc0ModelForOrt aborts while manifest metadata is pending without cancelling the shared request', async () => {
  await withMockedEnv(
    async () => {
      const mockedFetch = globalThis.fetch;
      let releaseManifest;
      const manifestGate = new Promise((resolve) => {
        releaseManifest = resolve;
      });
      globalThis.fetch = async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === MANIFEST_URL) await manifestGate;
        return mockedFetch(input);
      };
      const controller = new AbortController();
      const cancelled = loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: MANIFEST_URL, signal: controller.signal });
      const unsignalled = loadLc0ModelForOrt(MODEL_URL, { cache: false, manifestUrl: MANIFEST_URL });
      controller.abort();
      await assert.rejects(cancelled, (error) => error?.name === 'AbortError');
      releaseManifest();
      const loaded = await unsignalled;
      assert.equal(loaded.url, MODEL_URL);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

test('loadLc0ModelForOrt rechecks cancellation after asynchronous verification', async () => {
  const controller = new AbortController();
  const cryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  const nativeCrypto = globalThis.crypto;
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        async digest(algorithm, data) {
          const digest = await nativeCrypto.subtle.digest(algorithm, data);
          controller.abort();
          return digest;
        },
      },
    },
  });
  try {
    await withMockedEnv(
      async () => {
        await assert.rejects(
          () =>
            loadLc0ModelForOrt(MODEL_URL, {
              cache: false,
              manifestUrl: MANIFEST_URL,
              signal: controller.signal,
              onProgress: () => undefined,
            }),
          (error) => error?.name === 'AbortError',
        );
      },
      { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
    );
  } finally {
    if (cryptoDescriptor) Object.defineProperty(globalThis, 'crypto', cryptoDescriptor);
    else delete globalThis.crypto;
  }
});

test('clearLc0ModelCacheEntry removes only the selected model and its metadata', async () => {
  await withMockedEnv(
    async () => {
      const metadataStore = new MemoryMetadataStore();
      await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL, metadataStore });
      assert.equal(metadataStore.entries.size, 1);
      assert.equal(await clearLc0ModelCacheEntry(MODEL_URL, 'lc0-browser-models-v1', metadataStore), true);
      assert.equal(metadataStore.entries.size, 0);
      const cache = await globalThis.caches.open('lc0-browser-models-v1');
      assert.equal(await cache.match(MODEL_URL), undefined);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});

for (const operation of ['open', 'match', 'put', 'keys']) {
  test(`optional Cache Storage ${operation} failures keep verified model bytes without refetching`, async () => {
    await withMockedEnv(
      async (fetchLog) => {
        const cache = await globalThis.caches.open('lc0-browser-models-v1');
        const owner = operation === 'open' ? globalThis.caches : cache;
        owner[operation] = async () => {
          throw new DOMException('storage denied', 'QuotaExceededError');
        };
        const result = await loadLc0ModelForOrt(MODEL_URL, { cache: true });
        assert.equal(result.mode, 'memory');
        assert.equal(result.cacheStatus, 'unavailable');
        assert.equal(result.sha256Valid, true);
        assert.deepEqual(result.model, bytesOf('abc'));
        assert.equal(fetchLog.model, 1);
      },
      { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
    );
  });
}

test('optional storage fallback never swallows integrity or cancellation errors', async () => {
  await withMockedEnv(
    async () => {
      globalThis.caches.open = async () => {
        throw Error('denied');
      };
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, { cache: true }), /sha256 mismatch/);
      const controller = new AbortController();
      globalThis.caches.open = async () => {
        controller.abort();
        throw Error('denied');
      };
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, { cache: true, signal: controller.signal }), { name: 'AbortError' });
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: EMPTY_SHA256, manifestBytes: 3 },
  );
});

test('external manifest entries use the shared metadata fast path and rehash on mismatch', async () => {
  await withMockedEnv(
    async (fetchLog) => {
      const metadataStore = new MemoryMetadataStore();
      const options = {
        cache: true,
        channelUrl: null,
        manifestEntry: { file: 'test.onnx', url: MODEL_URL, bytes: 3, sha256: ABC_SHA256 },
        metadataStore,
      };
      assert.equal((await loadLc0ModelForOrt(MODEL_URL, options)).verification, 'hashed');
      const cache = await globalThis.caches.open('lc0-browser-models-v1');
      const original = cache.store.get(MODEL_URL);
      // Real Cache Storage clones every response read.
      cache.match = async () => original.clone();
      const warm = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(warm.verification, 'metadata');
      assert.equal(warm.hashMs, 0);
      const entry = [...metadataStore.entries.values()][0];
      entry.sha256 = EMPTY_SHA256;
      assert.equal((await loadLc0ModelForOrt(MODEL_URL, options)).verification, 'hashed');
      assert.equal(fetchLog.manifest, 0);
      assert.equal(fetchLog.model, 1);
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, { ...options, manifestEntry: { ...options.manifestEntry, sha256: EMPTY_SHA256 } }), /sha256 mismatch/);
    },
    { serveBytes: () => bytesOf('abc') },
  );
});

test('clearLc0ModelCache reports removed entries and is safe when empty', async () => {
  await withMockedEnv(
    async () => {
      await loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestUrl: MANIFEST_URL });
      const cleared = await clearLc0ModelCache();
      assert.equal(cleared.cleared, true);
      assert.equal(cleared.removedEntries, 1);
      const again = await clearLc0ModelCache();
      assert.equal(again.cleared, false);
      assert.equal(again.removedEntries, 0);
    },
    { serveBytes: () => bytesOf('abc'), manifestSha256: ABC_SHA256, manifestBytes: 3 },
  );
});
