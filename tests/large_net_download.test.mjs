import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';
import { inspectLargeNetDownload } from '../src/lc0/largeNetDownload.ts';
import { lc0SplitKUrl } from '../src/lc0/lc0SmallModelSelection.ts';
import { inspectLc0ModelDownload, loadLc0ModelForOrt } from '../src/lc0/modelCache.ts';

const url = 'https://example.test/models/lc0/custom.onnx';
const sha256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
const manifestEntry = { file: 'custom.onnx', url, bytes: 3, sha256 };

function environment(t) {
  const responses = new Map();
  const records = new Map();
  let exists = false;
  const counts = { opens: 0, downloads: 0, metadataWrites: 0, bodyReads: 0 };
  const cache = {
    async match(request) {
      counts.bodyReads++;
      return responses.get(request.url)?.clone();
    },
    async put(request, response) {
      responses.set(request.url, response);
    },
    async keys(request) {
      return [...responses.keys()].filter((key) => !request || key === request.url).map((key) => new Request(key));
    },
  };
  const metadataStore = {
    async get(key) {
      return records.get(key);
    },
    async put(entry) {
      counts.metadataWrites++;
      records.set(entry.key, entry);
    },
    async list() {
      return [...records.values()];
    },
  };
  t.mock.method(globalThis, 'fetch', async () => {
    counts.downloads++;
    return new Response('abc');
  });
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {
      has: async () => exists,
      open: async () => {
        counts.opens++;
        exists = true;
        return cache;
      },
    },
  });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'caches', previous);
    else delete globalThis.caches;
  });
  const options = { cache: true, channelUrl: null, manifestEntry, metadataStore };
  return { responses, records, counts, options };
}

test('download-state probe does not create caches or fetch model bytes on a miss', async (t) => {
  const f = environment(t);
  assert.deepEqual(await inspectLc0ModelDownload(url, f.options), { url, downloadBytes: 3, cached: false });
  assert.deepEqual(f.counts, { opens: 0, downloads: 0, metadataWrites: 0, bodyReads: 0 });
});

test('verified Cache Storage presence is read-only, without reading the cached model body', async (t) => {
  const f = environment(t);
  await loadLc0ModelForOrt(url, f.options);
  const before = { ...f.counts };
  assert.deepEqual(await inspectLc0ModelDownload(url, f.options), { url, downloadBytes: 3, cached: true });
  assert.equal(f.counts.downloads, before.downloads);
  assert.equal(f.counts.bodyReads, before.bodyReads);
  assert.equal(f.counts.metadataWrites, before.metadataWrites);
  f.responses.clear();
  assert.equal((await inspectLc0ModelDownload(url, f.options)).cached, false, 'orphan metadata is not a cache hit');
});

for (const invalid of ['missing', 'stale', 'hash', 'size', 'schema', 'runtime', 'url', 'cache', 'key']) {
  test(`unverified ${invalid} cache state cannot suppress a warning`, async (t) => {
    const f = environment(t);
    await loadLc0ModelForOrt(url, f.options);
    const metadata = [...f.records.values()][0];
    const changes = {
      stale: { verifiedAt: 0 },
      hash: { sha256: '0'.repeat(64) },
      size: { observedBytes: 4 },
      schema: { schemaVersion: -1 },
      runtime: { runtimeVersion: 'old' },
      url: { url: `${url}?other` },
      cache: { cacheName: 'other' },
      key: { key: 'other' },
    };
    if (invalid === 'missing') f.records.clear();
    else f.records.set(metadata.key, { ...metadata, ...changes[invalid] });
    assert.equal((await inspectLc0ModelDownload(url, f.options)).cached, false);
    assert.equal(f.counts.downloads, 1, 'the UI probe must never repair/fetch a model');
  });
}

test('channel resolution uses the selected Brotli URL and encoded download size, but raw integrity pins', async (t) => {
  const f = environment(t);
  const channelUrl = 'https://example.test/channel.json';
  const releaseUrl = 'https://example.test/release.json';
  const artifactUrl = 'https://example.test/artifacts/abc/br/def';
  const fetchBytes = globalThis.fetch;
  t.mock.method(globalThis, 'fetch', async (request) => {
    const href = request.url ?? request;
    if (href === channelUrl) return Response.json({ releaseManifestUrl: releaseUrl });
    if (href === releaseUrl)
      return Response.json({
        artifacts: [
          {
            logicalUrl: '/models/lc0/custom.onnx',
            raw: { bytes: 3, sha256 },
            representations: [{ encoding: 'br', url: artifactUrl, bytes: 2, sha256: '0'.repeat(64) }],
          },
        ],
      });
    assert.equal(href, artifactUrl);
    return fetchBytes(request);
  });
  const options = { ...f.options, channelUrl };
  await loadLc0ModelForOrt(url, options);
  assert.equal(f.responses.has(artifactUrl), true);
  assert.deepEqual(await inspectLc0ModelDownload(url, options), { url: artifactUrl, downloadBytes: 2, cached: true });
  assert.equal(f.counts.downloads, 1);
  f.options.manifestEntry = { ...manifestEntry, bytes: 4 };
  await assert.rejects(inspectLc0ModelDownload(url, { ...options, manifestEntry: f.options.manifestEntry }), /integrity mismatch/);
});

for (const placement of [
  { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' },
  { providers: ['wasm'], adapterVendor: 'apple' },
  { providers: ['webgpu'], adapterVendor: 'nvidia' },
]) {
  test(`dialog inspection follows the production t3 device rule: ${JSON.stringify(placement)}`, async () => {
    const selected = placement.providers[0] === 'webgpu' && placement.adapterVendor === 'apple' ? lc0SplitKUrl(BIG_NETS.t3.modelUrl) : BIG_NETS.t3.modelUrl;
    const inspected = [];
    const result = await inspectLargeNetDownload(BIG_NETS.t3, undefined, {
      placement: async () => placement,
      inspect: async (url) => {
        inspected.push(url);
        return { url, downloadBytes: 171021068, cached: true };
      },
    });
    assert.equal(result.url, selected);
    assert.equal(result.cached, true);
    const original = BIG_NETS.t3.modelUrl;
    assert.deepEqual(inspected, selected === original ? [original] : [selected, original]);
  });
}

test('the t3 probe includes the fallback even if its cached copy could be evicted by a split-K download', async () => {
  const original = BIG_NETS.t3.modelUrl;
  const splitK = lc0SplitKUrl(original);
  const sizes = new Map([
    [original, 160_000_000],
    [splitK, 170_000_000],
  ]);
  for (const [cached, expectedUrl, expectedSize] of [
    [[splitK], original, 160_000_000],
    [[original], splitK, 330_000_000],
    [[], splitK, 330_000_000],
  ]) {
    const result = await inspectLargeNetDownload(BIG_NETS.t3, undefined, {
      placement: async () => ({ providers: ['webgpu', 'wasm'], adapterVendor: 'apple' }),
      inspect: async (url) => ({ url, downloadBytes: sizes.get(url), cached: cached.includes(url) }),
    });
    assert.equal(result.cached, false);
    assert.equal(result.mayDownloadFallback, true);
    assert.equal(result.url, expectedUrl);
    assert.equal(result.downloadBytes, expectedSize);
  }
});

test('Queen Odds inspection needs no GPU allocation/probe and never substitutes a different model', async () => {
  const result = await inspectLargeNetDownload(BIG_NETS.lqo, undefined, {
    placement: () => assert.fail('Queen Odds has no device-specific rewrite'),
    inspect: async (url) => ({ url, downloadBytes: 95662716, cached: false }),
  });
  assert.equal(result.downloadBytes, 95662716);
  assert.equal(result.url, BIG_NETS.lqo.modelUrl);
});

test('already-aborted download-state inspection performs no fetches', async (t) => {
  const f = environment(t);
  const abort = new AbortController();
  abort.abort();
  await assert.rejects(inspectLc0ModelDownload(url, { ...f.options, signal: abort.signal }), { name: 'AbortError' });
  assert.equal(f.counts.downloads, 0);
});
