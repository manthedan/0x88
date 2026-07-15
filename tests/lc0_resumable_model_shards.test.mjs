import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, open, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, test } from 'node:test';
import { pathToFileURL } from 'node:url';
import {
  clearResumableLc0ModelShardCache,
  loadResumableLc0ModelForOrt,
} from '../src/lc0/modelCache.ts';
import { createBenchmarkRunDirectory } from '../scripts/bench_resumable_model_shards.mjs';
import {
  generateResumableModelShards,
  writeFileAtomically,
} from '../scripts/generate_resumable_model_shards.mjs';
import {
  FileModelShardStore,
  localFileFetch,
  writeReconstructedModelAtomically,
} from '../scripts/reconstruct_resumable_model_shards.mjs';

const MIB = 1024 * 1024;
const CHUNK_BYTES = 16 * MIB;
const MODEL_BYTES = 5 * CHUNK_BYTES + 12345;
const temporaryRoots = new Set();

afterEach(async () => {
  const roots = [...temporaryRoots];
  temporaryRoots.clear();
  await Promise.all(roots.map((root) => rm(root, { recursive: true, force: true })));
});

async function writeDeterministicModel(path, bytes = MODEL_BYTES) {
  const file = await open(path, 'w');
  const hash = createHash('sha256');
  const block = Buffer.allocUnsafe(MIB);
  try {
    for (let offset = 0; offset < bytes; offset += block.byteLength) {
      const length = Math.min(block.byteLength, bytes - offset);
      const blockNumber = offset / block.byteLength;
      for (let index = 0; index < length; index += 1) {
        block[index] = (index * 31 + (index >>> 8) * 17 + blockNumber * 29 + (blockNumber >>> 3) * 7 + 23) & 0xff;
      }
      const slice = block.subarray(0, length);
      hash.update(slice);
      await file.write(slice);
    }
  } finally {
    await file.close();
  }
  return hash.digest('hex');
}

async function assertMatchesFile(actual, expectedPath) {
  const file = await open(expectedPath, 'r');
  try {
    for (let offset = 0; offset < actual.byteLength; offset += MIB) {
      const expected = Buffer.allocUnsafe(Math.min(MIB, actual.byteLength - offset));
      const { bytesRead } = await file.read(expected, 0, expected.byteLength, offset);
      assert.equal(bytesRead, expected.byteLength);
      assert.equal(Buffer.compare(Buffer.from(actual, offset, expected.byteLength), expected), 0);
    }
  } finally {
    await file.close();
  }
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'lc0-resumable-shards-'));
  temporaryRoots.add(root);
  const modelPath = join(root, 'fixture.onnx');
  const expectedSha256 = await writeDeterministicModel(modelPath);
  const generated = await generateResumableModelShards({
    inputPath: modelPath,
    outputDir: join(root, 'published'),
    chunkBytes: CHUNK_BYTES,
  });
  return {
    root,
    modelPath,
    expectedSha256,
    manifestPath: generated.manifestPath,
    manifest: generated.manifest,
    cacheDir: join(root, 'cache'),
  };
}

function loaderOptions(entry, overrides = {}) {
  return {
    researchOnly: true,
    concurrency: 3,
    corruptionRetries: 1,
    fetchFn: localFileFetch(),
    shardStore: new FileModelShardStore(entry.cacheDir),
    ...overrides,
  };
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function singleShardManifest(bytes, url = 'shards/model.bin') {
  const hash = sha256(bytes);
  return {
    schema: 'lc0_browser.resumable_model_shards.v1',
    chunkBytes: CHUNK_BYTES,
    decoded: { bytes: bytes.byteLength, sha256: hash },
    shards: [{ bytes: bytes.byteLength, sha256: hash, url }],
  };
}

function responseWithUrl(body, init, url) {
  const response = new Response(body, init);
  Object.defineProperty(response, 'url', { value: url });
  return response;
}

class MemoryShardStore {
  constructor(entries = []) {
    this.entries = new Map(entries.map((entry) => [entry.sha256, {
      bytes: entry.bytes.slice(0),
      lastUsedAt: entry.lastUsedAt ?? 0,
    }]));
    this.deleted = [];
    this.touched = [];
  }

  async get(sha256) {
    return this.entries.get(sha256)?.bytes.slice(0);
  }

  async put(sha256, bytes) {
    this.entries.set(sha256, { bytes: bytes.slice(0), lastUsedAt: Date.now() });
  }

  async delete(sha256) {
    this.deleted.push(sha256);
    this.entries.delete(sha256);
  }

  async touch(sha256) {
    const entry = this.entries.get(sha256);
    if (!entry) return;
    this.touched.push(sha256);
    entry.lastUsedAt = Date.now();
  }

  async list() {
    return [...this.entries].map(([sha256, entry]) => ({
      sha256,
      bytes: entry.bytes.byteLength,
      lastUsedAt: entry.lastUsedAt,
    }));
  }
}

class FakeShardCache {
  constructor() {
    this.entries = new Map();
    this.bodyReads = 0;
    this.bodyStreamReads = 0;
    this.bodyPuts = 0;
    this.metadataPuts = 0;
    this.failMetadataPuts = false;
    this.deleted = [];
  }

  seed(request, bytes, headers = {}) {
    this.entries.set(request.url ?? request, {
      bytes: Buffer.from(bytes),
      headers: new Headers(headers),
    });
  }

  async match(request) {
    const url = request.url ?? request;
    const entry = this.entries.get(url);
    if (!entry) return undefined;
    const response = new Response(entry.bytes, { headers: entry.headers });
    if (!url.includes('/metadata/')) {
      const arrayBuffer = response.arrayBuffer.bind(response);
      response.arrayBuffer = async () => {
        this.bodyReads += 1;
        return arrayBuffer();
      };
      const body = response.body;
      const cache = this;
      Object.defineProperty(response, 'body', {
        value: new ReadableStream({
          async pull(controller) {
            const reader = body.getReader();
            const { done, value } = await reader.read();
            reader.releaseLock();
            if (done) controller.close();
            else {
              controller.enqueue(value);
              cache.bodyStreamReads += 1;
            }
          },
        }),
      });
    }
    return response;
  }

  async put(request, response) {
    const url = request.url ?? request;
    if (url.includes('/metadata/') && this.failMetadataPuts) {
      this.metadataPuts += 1;
      throw new Error('forced metadata cache.put failure');
    }
    const bytes = Buffer.from(await response.arrayBuffer());
    this.entries.set(url, { bytes, headers: new Headers(response.headers) });
    if (url.includes('/metadata/')) this.metadataPuts += 1;
    else this.bodyPuts += 1;
  }

  async delete(request) {
    const url = request.url ?? request;
    this.deleted.push(url);
    return this.entries.delete(url);
  }

  async keys() {
    return [...this.entries.keys()].map((url) => new Request(url));
  }
}

class FakeShardCacheStorage {
  constructor(cache) {
    this.cache = cache;
  }

  async open() {
    return this.cache;
  }

  async delete() {
    this.cache.entries.clear();
    return true;
  }
}

test('resumable shard generation uses ordered 16 MiB content hashes and reconstructs byte-identically', async () => {
  const entry = await fixture();
  assert.equal(entry.manifest.chunkBytes, CHUNK_BYTES);
  assert.equal(entry.manifest.decoded.bytes, MODEL_BYTES);
  assert.equal(entry.manifest.decoded.sha256, entry.expectedSha256);
  assert.equal(entry.manifest.shards.length, 6);
  assert(entry.manifest.shards.slice(0, -1).every((shard) => shard.bytes === CHUNK_BYTES));
  assert(entry.manifest.shards.every((shard) => shard.url === `shards/sha256/${shard.sha256}.bin`));

  const firstShardPath = new URL(entry.manifest.shards[0].url, pathToFileURL(entry.manifestPath));
  await writeFile(firstShardPath, Buffer.alloc(1, 0xa5));
  const sizeRepaired = await generateResumableModelShards({
    inputPath: entry.modelPath,
    outputDir: join(entry.root, 'published'),
    chunkBytes: CHUNK_BYTES,
  });
  assert.equal(sizeRepaired.uniqueBytesWritten, CHUNK_BYTES);
  assert.equal((await readFile(firstShardPath)).byteLength, CHUNK_BYTES);

  await writeFile(firstShardPath, Buffer.alloc(CHUNK_BYTES, 0xa5));
  const hashRepaired = await generateResumableModelShards({
    inputPath: entry.modelPath,
    outputDir: join(entry.root, 'published'),
    chunkBytes: CHUNK_BYTES,
  });
  assert.equal(hashRepaired.uniqueBytesWritten, CHUNK_BYTES);
  assert.equal(
    createHash('sha256').update(await readFile(firstShardPath)).digest('hex'),
    entry.manifest.shards[0].sha256,
  );

  const result = await loadResumableLc0ModelForOrt(pathToFileURL(entry.manifestPath).href, loaderOptions(entry));
  assert.equal(result.sha256, entry.expectedSha256);
  assert.equal(result.bytes, MODEL_BYTES);
  assert.equal(result.downloadedShards, 6);
  assert.equal(result.reusedShards, 0);
  assert(result.peakTemporaryBytes <= 4 * CHUNK_BYTES);
  await assertMatchesFile(result.model, entry.modelPath);
});

test('manifest publishing atomically preserves the previous file when replacement fails', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-resumable-manifest-atomic-'));
  temporaryRoots.add(root);
  const manifestPath = join(root, 'model.resumable.json');
  const previous = '{"schema":"previous-valid-manifest"}\n';
  await writeFile(manifestPath, previous);
  await assert.rejects(
    writeFileAtomically(manifestPath, '{"schema":"replacement"}\n', async () => {
      throw new Error('forced rename failure');
    }),
    /forced rename failure/,
  );
  assert.equal(await readFile(manifestPath, 'utf8'), previous);
  assert.deepEqual(
    (await readdir(root)).filter((name) => name.startsWith('model.resumable.json.tmp-')),
    [],
  );
});

test('reconstructed output atomically preserves the previous file when replacement fails', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-resumable-output-atomic-'));
  temporaryRoots.add(root);
  const outputPath = join(root, 'model.onnx');
  const previous = Buffer.from('previous valid model');
  await writeFile(outputPath, previous);
  await assert.rejects(
    writeReconstructedModelAtomically(outputPath, Buffer.from('replacement model'), async () => {
      throw new Error('forced output rename failure');
    }),
    /forced output rename failure/,
  );
  assert.deepEqual(await readFile(outputPath), previous);
  assert.deepEqual(
    (await readdir(root)).filter((name) => name.startsWith('model.onnx.tmp-')),
    [],
  );
});

test('oversized shard responses are cancelled before unbounded buffering', async () => {
  const entry = await fixture();
  const fileFetch = localFileFetch();
  let cancelled = false;
  let shardFetches = 0;
  await assert.rejects(
    loadResumableLc0ModelForOrt(pathToFileURL(entry.manifestPath).href, loaderOptions(entry, {
      concurrency: 1,
      corruptionRetries: 0,
      fetchFn: async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (!url.endsWith('.bin')) return fileFetch(input, init);
        shardFetches += 1;
        let sentExpectedBytes = false;
        return new Response(new ReadableStream({
          pull(controller) {
            if (!sentExpectedBytes) {
              sentExpectedBytes = true;
              controller.enqueue(new Uint8Array(CHUNK_BYTES));
            } else {
              controller.enqueue(Uint8Array.of(1));
            }
          },
          cancel() {
            cancelled = true;
          },
        }));
      },
    })),
    /corruption persisted/,
  );
  assert.equal(shardFetches, 1);
  assert.equal(cancelled, true);
});

test('local file fetch streams JSON and shard bodies while propagating aborts', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-resumable-local-fetch-'));
  temporaryRoots.add(root);
  const manifestPath = join(root, 'model.resumable.json');
  const shardPath = join(root, 'model.bin');
  const manifest = {
    schema: 'lc0_browser.resumable_model_shards.v1',
    chunkBytes: CHUNK_BYTES,
    decoded: { bytes: 1, sha256: sha256(Buffer.of(1)) },
    shards: [{ bytes: 1, sha256: sha256(Buffer.of(1)), url: 'model.bin' }],
  };
  await writeFile(manifestPath, JSON.stringify(manifest));
  await writeFile(shardPath, Buffer.of(1, 2));
  const fetchFile = localFileFetch();
  const manifestResponse = await fetchFile(pathToFileURL(manifestPath).href);
  assert.equal(manifestResponse.url, '');
  assert.equal(manifestResponse.headers.get('content-type'), 'application/json');
  assert.equal(manifestResponse.headers.get('content-length'), String(Buffer.byteLength(JSON.stringify(manifest))));
  assert.deepEqual(await manifestResponse.json(), manifest);
  const shardResponse = await fetchFile(pathToFileURL(shardPath).href);
  assert.equal(shardResponse.url, '');
  assert.deepEqual(Buffer.from(await shardResponse.arrayBuffer()), Buffer.of(1, 2));

  const abortBytes = Buffer.alloc(CHUNK_BYTES, 0x35);
  manifest.decoded = { bytes: abortBytes.byteLength, sha256: sha256(abortBytes) };
  manifest.shards = [{
    bytes: abortBytes.byteLength,
    sha256: manifest.decoded.sha256,
    url: 'model.bin',
  }];
  await writeFile(manifestPath, JSON.stringify(manifest));
  await writeFile(shardPath, abortBytes);
  await (await fetchFile(pathToFileURL(manifestPath).href)).json();
  const controller = new AbortController();
  const abortedShardResponse = await fetchFile(pathToFileURL(shardPath).href, { signal: controller.signal });
  controller.abort();
  await assert.rejects(abortedShardResponse.arrayBuffer(), { name: 'AbortError' });
});

test('oversized local manifests are rejected by the loader stream bound', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-resumable-local-manifest-bound-'));
  temporaryRoots.add(root);
  const manifestPath = join(root, 'oversized.resumable.json');
  const manifestBody = `${JSON.stringify(singleShardManifest(Buffer.of(1)))}${' '.repeat(4096)}`;
  await writeFile(manifestPath, manifestBody);
  await assert.rejects(
    loadResumableLc0ModelForOrt(pathToFileURL(manifestPath).href, {
      researchOnly: true,
      maxManifestBytes: 1024,
      fetchFn: localFileFetch(),
      shardStore: new MemoryShardStore(),
    }),
    /manifest exceeded configured limit 1024 bytes/,
  );
});

test('filesystem shard cache bounds oversized reads and propagates aborts', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-resumable-file-cache-'));
  temporaryRoots.add(root);
  const store = new FileModelShardStore(join(root, 'cache'));

  await t.test('oversized cached entries are deleted and redownloaded', async () => {
    const bytes = Buffer.alloc(CHUNK_BYTES, 0x36);
    const manifest = singleShardManifest(bytes);
    const manifestUrl = 'https://models.example/research/filesystem-cache.resumable.json';
    const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
    await store.put(manifest.shards[0].sha256, bytes.buffer);
    await writeFile(store.path(manifest.shards[0].sha256), Buffer.alloc(CHUNK_BYTES + 1, 0x37));
    let shardFetches = 0;
    const result = await loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      concurrency: 1,
      corruptionRetries: 0,
      shardStore: store,
      fetchFn: async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === manifestUrl) return new Response(JSON.stringify(manifest));
        if (url === shardUrl) {
          shardFetches += 1;
          return new Response(bytes);
        }
        return new Response(null, { status: 404 });
      },
    });
    assert.equal(shardFetches, 1);
    assert.equal(result.corruptShardsEvicted, 1);
    assert.deepEqual(await readFile(store.path(manifest.shards[0].sha256)), bytes);
  });

  await t.test('aborted cached reads reject without deleting valid entries', async () => {
    const bytes = Buffer.alloc(2 * CHUNK_BYTES, 0x38);
    const hash = sha256(bytes);
    await store.put(hash, bytes.buffer);
    const controller = new AbortController();
    const read = store.getBounded(hash, bytes.byteLength, controller.signal);
    setImmediate(() => controller.abort());
    await assert.rejects(read, { name: 'AbortError' });
    assert.equal((await store.list()).some((entry) => entry.sha256 === hash), true);
  });
});

test('manifest allocation limits reject amplification before shard access', async (t) => {
  const manifestUrl = 'https://models.example/research/allocation-limits.resumable.json';

  await t.test('duplicate shard references respect the configured reference limit', async () => {
    const bytes = Buffer.alloc(CHUNK_BYTES, 0x39);
    const descriptor = singleShardManifest(bytes).shards[0];
    const manifest = {
      schema: 'lc0_browser.resumable_model_shards.v1',
      chunkBytes: CHUNK_BYTES,
      decoded: {
        bytes: 3 * CHUNK_BYTES,
        sha256: sha256(Buffer.concat([bytes, bytes, bytes])),
      },
      shards: [descriptor, descriptor, descriptor],
    };
    let fetches = 0;
    const store = new MemoryShardStore();
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        maxShardReferences: 2,
        shardStore: store,
        fetchFn: async () => {
          fetches += 1;
          return new Response(JSON.stringify(manifest));
        },
      }),
      /shard references exceed configured limit 2/,
    );
    assert.equal(fetches, 1);
    assert.equal(store.entries.size, 0);
  });

  await t.test('decoded bytes respect the configured allocation limit', async () => {
    const bytes = Buffer.alloc(CHUNK_BYTES, 0x3a);
    const manifest = singleShardManifest(bytes);
    let fetches = 0;
    const store = new MemoryShardStore();
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        maxDecodedBytes: CHUNK_BYTES - 1,
        shardStore: store,
        fetchFn: async () => {
          fetches += 1;
          return new Response(JSON.stringify(manifest));
        },
      }),
      new RegExp(`decoded bytes exceed configured limit ${CHUNK_BYTES - 1}`),
    );
    assert.equal(fetches, 1);
    assert.equal(store.entries.size, 0);
  });

  await t.test('invalid allocation limit options reject before the manifest fetch', async () => {
    for (const options of [
      { maxManifestBytes: -1 },
      { maxManifestBytes: 1.5 },
      { maxDecodedBytes: -1 },
      { maxDecodedBytes: 1.5 },
      { maxShardReferences: -1 },
      { maxShardReferences: Number.NaN },
    ]) {
      let fetches = 0;
      await assert.rejects(
        loadResumableLc0ModelForOrt(manifestUrl, {
          researchOnly: true,
          ...options,
          fetchFn: async () => {
            fetches += 1;
            return new Response(null, { status: 500 });
          },
        }),
        /must be a non-negative safe integer or Infinity/,
      );
      assert.equal(fetches, 0);
    }
  });
});

test('remote manifest bodies are bounded before JSON parsing and validation', async (t) => {
  const manifestUrl = 'https://models.example/research/bounded-manifest.resumable.json';

  await t.test('oversized unencoded Content-Length rejects before reading', async () => {
    let cancelled = false;
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        maxManifestBytes: 32,
        shardStore: new MemoryShardStore(),
        fetchFn: async () => new Response(new ReadableStream({
          pull(controller) {
            controller.enqueue(Buffer.from('{}'));
            controller.close();
          },
          cancel() {
            cancelled = true;
          },
        }), {
          headers: { 'content-length': '33' },
        }),
      }),
      /manifest exceeded configured limit 32 bytes/,
    );
    assert.equal(cancelled, true);
  });

  await t.test('oversized streamed body is cancelled before parsing', async () => {
    let cancelled = false;
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        maxManifestBytes: 32,
        shardStore: new MemoryShardStore(),
        fetchFn: async () => new Response(new ReadableStream({
          start(controller) {
            controller.enqueue(Buffer.alloc(33, 0x7b));
          },
          cancel() {
            cancelled = true;
          },
        })),
      }),
      /manifest exceeded configured limit 32 bytes/,
    );
    assert.equal(cancelled, true);
  });

  await t.test('content-encoded responses ignore encoded length but bound decoded bytes', async () => {
    const bytes = Buffer.of(0x3b);
    const manifest = singleShardManifest(bytes);
    const manifestBody = JSON.stringify(manifest);
    const maxManifestBytes = Buffer.byteLength(manifestBody);
    const store = new MemoryShardStore([{
      sha256: manifest.shards[0].sha256,
      bytes: bytes.buffer,
    }]);
    const result = await loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      maxManifestBytes,
      shardStore: store,
      fetchFn: async () => new Response(manifestBody, {
        headers: {
          'content-encoding': 'gzip',
          'content-length': String(maxManifestBytes + 1),
        },
      }),
    });
    assert.deepEqual(Buffer.from(result.model), bytes);

    let cancelled = false;
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        maxManifestBytes: maxManifestBytes - 1,
        shardStore: store,
        fetchFn: async () => new Response(new ReadableStream({
          start(controller) {
            controller.enqueue(Buffer.from(manifestBody));
          },
          cancel() {
            cancelled = true;
          },
        }), {
          headers: {
            'content-encoding': 'gzip',
            'content-length': '1',
          },
        }),
      }),
      new RegExp(`manifest exceeded configured limit ${maxManifestBytes - 1} bytes`),
    );
    assert.equal(cancelled, true);
  });
});

test('encoded shard responses ignore encoded Content-Length while bounding decoded bytes', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x3c);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/model.resumable.json';
  const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
  const fetched = [];
  const result = await loadResumableLc0ModelForOrt(manifestUrl, {
    researchOnly: true,
    corruptionRetries: 0,
    shardStore: new MemoryShardStore(),
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      fetched.push(url);
      if (url === manifestUrl) return new Response(JSON.stringify(manifest));
      if (url === shardUrl) {
        return new Response(bytes, {
          headers: {
            'content-encoding': 'gzip',
            'content-length': String(CHUNK_BYTES + 1),
          },
        });
      }
      return new Response(null, { status: 404 });
    },
  });
  assert.deepEqual(fetched, [manifestUrl, shardUrl]);
  assert.deepEqual(Buffer.from(result.model), bytes);
});

test('relative shard URLs resolve against the redirected manifest response URL', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x5d);
  const manifest = singleShardManifest(bytes);
  const requestedManifestUrl = 'https://models.example/latest/model.resumable.json';
  const redirectedManifestUrl = 'https://cdn.example/releases/v2/model.resumable.json';
  const redirectedShardUrl = new URL(manifest.shards[0].url, redirectedManifestUrl).href;
  const fetched = [];
  const result = await loadResumableLc0ModelForOrt(requestedManifestUrl, {
    researchOnly: true,
    corruptionRetries: 0,
    shardStore: new MemoryShardStore(),
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      fetched.push(url);
      if (url === requestedManifestUrl) {
        return responseWithUrl(JSON.stringify(manifest), undefined, redirectedManifestUrl);
      }
      if (url === redirectedShardUrl) return new Response(bytes);
      return new Response(null, { status: 404 });
    },
  });
  assert.deepEqual(fetched, [requestedManifestUrl, redirectedShardUrl]);
  assert.deepEqual(Buffer.from(result.model), bytes);
});

test('verified persistent cache hits refresh LRU metadata before cache-bound eviction', async () => {
  const first = Buffer.alloc(CHUNK_BYTES, 0x11);
  const second = Buffer.alloc(CHUNK_BYTES, 0x22);
  const firstHash = sha256(first);
  const secondHash = sha256(second);
  const manifest = {
    schema: 'lc0_browser.resumable_model_shards.v1',
    chunkBytes: CHUNK_BYTES,
    decoded: {
      bytes: first.byteLength + second.byteLength,
      sha256: sha256(Buffer.concat([first, second])),
    },
    shards: [
      { bytes: first.byteLength, sha256: firstHash, url: 'shards/first.bin' },
      { bytes: second.byteLength, sha256: secondHash, url: 'shards/second.bin' },
    ],
  };
  const oldUnprotected = Buffer.alloc(1, 0x33);
  const newUnprotected = Buffer.alloc(1, 0x44);
  const oldUnprotectedHash = sha256(oldUnprotected);
  const newUnprotectedHash = sha256(newUnprotected);
  const store = new MemoryShardStore([
    { sha256: firstHash, bytes: first.buffer, lastUsedAt: 1 },
    { sha256: secondHash, bytes: second.buffer, lastUsedAt: 2 },
    { sha256: oldUnprotectedHash, bytes: oldUnprotected.buffer, lastUsedAt: 3 },
    { sha256: newUnprotectedHash, bytes: newUnprotected.buffer, lastUsedAt: 4 },
  ]);
  const manifestUrl = 'https://models.example/research/model.resumable.json';
  const result = await loadResumableLc0ModelForOrt(manifestUrl, {
    researchOnly: true,
    concurrency: 1,
    maxCacheEntries: 3,
    maxCacheBytes: Infinity,
    shardStore: store,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === manifestUrl) return new Response(JSON.stringify(manifest));
      return new Response(null, { status: 500 });
    },
  });
  assert.equal(result.reusedShards, 2);
  assert.equal(result.downloadedShards, 0);
  assert.equal(store.touched.includes(firstHash), true);
  assert.equal(store.touched.includes(secondHash), true);
  assert.deepEqual(store.deleted, [oldUnprotectedHash]);
  assert.equal(store.entries.has(firstHash), true);
  assert.equal(store.entries.has(secondHash), true);
  assert.equal(store.entries.has(newUnprotectedHash), true);
});

test('quota eviction stops after the first deletion that creates sufficient reserve', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x6e);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/model.resumable.json';
  const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
  const oldFirst = Buffer.alloc(CHUNK_BYTES, 0x77);
  const oldSecond = Buffer.alloc(CHUNK_BYTES, 0x88);
  const oldFirstHash = sha256(oldFirst);
  const oldSecondHash = sha256(oldSecond);
  const store = new MemoryShardStore([
    { sha256: oldFirstHash, bytes: oldFirst.buffer, lastUsedAt: 1 },
    { sha256: oldSecondHash, bytes: oldSecond.buffer, lastUsedAt: 2 },
  ]);
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      storage: {
        async estimate() {
          const usage = [...store.entries.values()].reduce((sum, entry) => sum + entry.bytes.byteLength, 0);
          return { quota: 3 * CHUNK_BYTES, usage };
        },
      },
    },
  });
  try {
    const result = await loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      concurrency: 1,
      corruptionRetries: 0,
      maxCacheEntries: Infinity,
      maxCacheBytes: Infinity,
      minimumFreeBytesAfterCache: CHUNK_BYTES,
      shardStore: store,
      fetchFn: async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === manifestUrl) return new Response(JSON.stringify(manifest));
        if (url === shardUrl) return new Response(bytes);
        return new Response(null, { status: 404 });
      },
    });
    assert.equal(result.downloadedShards, 1);
    assert.deepEqual(store.deleted, [oldFirstHash]);
    assert.equal(store.entries.has(oldSecondHash), true);
    assert.equal(store.entries.has(manifest.shards[0].sha256), true);
  } finally {
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator);
    else delete globalThis.navigator;
  }
});

test('concurrent shard writes serialize quota reservation through persistence', async () => {
  const first = Buffer.alloc(CHUNK_BYTES, 0x41);
  const second = Buffer.alloc(CHUNK_BYTES, 0x42);
  const manifest = {
    schema: 'lc0_browser.resumable_model_shards.v1',
    chunkBytes: CHUNK_BYTES,
    decoded: {
      bytes: first.byteLength + second.byteLength,
      sha256: sha256(Buffer.concat([first, second])),
    },
    shards: [
      { bytes: first.byteLength, sha256: sha256(first), url: 'shards/first.bin' },
      { bytes: second.byteLength, sha256: sha256(second), url: 'shards/second.bin' },
    ],
  };
  const manifestUrl = 'https://models.example/research/model.resumable.json';
  const store = new MemoryShardStore();
  let activePuts = 0;
  let peakPuts = 0;
  store.put = async (hash, bytes) => {
    activePuts += 1;
    peakPuts = Math.max(peakPuts, activePuts);
    try {
      await new Promise((resolve) => setTimeout(resolve, 10));
      store.entries.set(hash, { bytes: bytes.slice(0), lastUsedAt: Date.now() });
    } finally {
      activePuts -= 1;
    }
  };
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      storage: {
        async estimate() {
          const usage = [...store.entries.values()].reduce((sum, entry) => sum + entry.bytes.byteLength, 0);
          return { quota: 2.5 * CHUNK_BYTES, usage };
        },
      },
    },
  });
  try {
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        concurrency: 2,
        corruptionRetries: 0,
        maxCacheEntries: Infinity,
        maxCacheBytes: Infinity,
        minimumFreeBytesAfterCache: CHUNK_BYTES,
        shardStore: store,
        fetchFn: async (input) => {
          const url = typeof input === 'string' ? input : input.url;
          if (url === manifestUrl) return new Response(JSON.stringify(manifest));
          if (url.endsWith('/first.bin')) return new Response(first);
          if (url.endsWith('/second.bin')) return new Response(second);
          return new Response(null, { status: 404 });
        },
      }),
      /Insufficient storage quota/,
    );
    assert.equal(peakPuts, 1);
    assert.equal(store.entries.size, 1);
  } finally {
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator);
    else delete globalThis.navigator;
  }
});

test('concurrent loads recheck persistence inside the serialized cache-write slot', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x4a);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/shared.resumable.json';
  const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
  const store = new MemoryShardStore();
  let shardFetches = 0;
  let puts = 0;
  let releaseFirstPut;
  let secondShardFetchStarted;
  const secondShardFetch = new Promise((resolve) => {
    secondShardFetchStarted = resolve;
  });
  const firstPutStarted = new Promise((resolve) => {
    store.put = async (hash, storedBytes) => {
      puts += 1;
      if (puts === 1) {
        await new Promise((release) => {
          releaseFirstPut = release;
          resolve();
        });
      }
      store.entries.set(hash, { bytes: storedBytes.slice(0), lastUsedAt: Date.now() });
    };
  });
  const fetchFn = async (input) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url === manifestUrl) return new Response(JSON.stringify(manifest));
    if (url === shardUrl) {
      shardFetches += 1;
      if (shardFetches === 2) secondShardFetchStarted();
      return new Response(bytes);
    }
    return new Response(null, { status: 404 });
  };
  const options = {
    researchOnly: true,
    cacheName: 'shared-write-recheck-test',
    concurrency: 1,
    corruptionRetries: 0,
    maxCacheEntries: Infinity,
    maxCacheBytes: Infinity,
    shardStore: store,
    fetchFn,
  };
  const first = loadResumableLc0ModelForOrt(manifestUrl, options);
  await firstPutStarted;
  const second = loadResumableLc0ModelForOrt(manifestUrl, options);
  await secondShardFetch;
  releaseFirstPut();
  const [firstResult, secondResult] = await Promise.all([first, second]);
  assert.equal(shardFetches, 2);
  assert.equal(puts, 1);
  assert.deepEqual(
    [firstResult.downloadedShards, secondResult.downloadedShards].sort((a, b) => a - b),
    [0, 1],
  );
  assert.deepEqual(
    [firstResult.reusedShards, secondResult.reusedShards].sort((a, b) => a - b),
    [0, 1],
  );
});

test('final eviction preserves shards protected by another active load', async () => {
  const protectedBytes = Buffer.alloc(CHUNK_BYTES, 0x4b);
  const incomingBytes = Buffer.alloc(CHUNK_BYTES, 0x4c);
  const protectedManifest = singleShardManifest(protectedBytes, 'shards/protected.bin');
  const incomingManifest = singleShardManifest(incomingBytes, 'shards/incoming.bin');
  const protectedManifestUrl = 'https://models.example/research/protected.resumable.json';
  const incomingManifestUrl = 'https://models.example/research/incoming.resumable.json';
  const protectedShardUrl = new URL(protectedManifest.shards[0].url, protectedManifestUrl).href;
  const incomingShardUrl = new URL(incomingManifest.shards[0].url, incomingManifestUrl).href;
  const protectedHash = protectedManifest.shards[0].sha256;
  const incomingHash = incomingManifest.shards[0].sha256;
  const store = new MemoryShardStore([
    { sha256: protectedHash, bytes: protectedBytes.buffer, lastUsedAt: 1 },
  ]);
  store.touch = async () => {};
  let protectedReads = 0;
  let releaseProtectedReconstruction;
  const reconstructionStarted = new Promise((resolve) => {
    const originalGet = store.get.bind(store);
    store.get = async (hash) => {
      if (hash === protectedHash) {
        protectedReads += 1;
        if (protectedReads === 2) {
          resolve();
          await new Promise((release) => {
            releaseProtectedReconstruction = release;
          });
        }
      }
      return originalGet(hash);
    };
  });
  const protectedLoad = loadResumableLc0ModelForOrt(protectedManifestUrl, {
    researchOnly: true,
    cacheName: 'shared-eviction-protection-test',
    concurrency: 1,
    maxCacheEntries: 1,
    maxCacheBytes: Infinity,
    shardStore: store,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === protectedManifestUrl) return new Response(JSON.stringify(protectedManifest));
      if (url === protectedShardUrl) return new Response(null, { status: 500 });
      return new Response(null, { status: 404 });
    },
  });
  await reconstructionStarted;
  const incomingLoad = await loadResumableLc0ModelForOrt(incomingManifestUrl, {
    researchOnly: true,
    cacheName: 'shared-eviction-protection-test',
    concurrency: 1,
    corruptionRetries: 0,
    maxCacheEntries: 1,
    maxCacheBytes: Infinity,
    shardStore: store,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === incomingManifestUrl) return new Response(JSON.stringify(incomingManifest));
      if (url === incomingShardUrl) return new Response(incomingBytes);
      return new Response(null, { status: 404 });
    },
  });
  assert.equal(incomingLoad.downloadedShards, 1);
  assert.equal(store.entries.has(protectedHash), true);
  assert.equal(store.entries.has(incomingHash), false);
  assert.equal(store.deleted.includes(protectedHash), false);
  releaseProtectedReconstruction();
  await protectedLoad;
});

test('active shard protection is isolated between different custom stores', async () => {
  const protectedBytes = Buffer.alloc(CHUNK_BYTES, 0x51);
  const incomingBytes = Buffer.alloc(CHUNK_BYTES, 0x52);
  const protectedManifest = singleShardManifest(protectedBytes, 'shards/protected.bin');
  const incomingManifest = singleShardManifest(incomingBytes, 'shards/incoming.bin');
  const protectedManifestUrl = 'https://models.example/research/custom-store-protected.resumable.json';
  const incomingManifestUrl = 'https://models.example/research/custom-store-incoming.resumable.json';
  const protectedHash = protectedManifest.shards[0].sha256;
  const incomingHash = incomingManifest.shards[0].sha256;
  const protectedStore = new MemoryShardStore([
    { sha256: protectedHash, bytes: protectedBytes.buffer, lastUsedAt: 1 },
  ]);
  const incomingStore = new MemoryShardStore([
    { sha256: protectedHash, bytes: protectedBytes.buffer, lastUsedAt: 1 },
  ]);
  protectedStore.touch = async () => {};
  let protectedReads = 0;
  let releaseProtectedReconstruction;
  const reconstructionStarted = new Promise((resolve) => {
    const originalGet = protectedStore.get.bind(protectedStore);
    protectedStore.get = async (hash) => {
      if (hash === protectedHash) {
        protectedReads += 1;
        if (protectedReads === 2) {
          resolve();
          await new Promise((release) => {
            releaseProtectedReconstruction = release;
          });
        }
      }
      return originalGet(hash);
    };
  });
  const protectedLoad = loadResumableLc0ModelForOrt(protectedManifestUrl, {
    researchOnly: true,
    cacheName: 'custom-store-isolation-test',
    concurrency: 1,
    maxCacheEntries: 1,
    maxCacheBytes: Infinity,
    shardStore: protectedStore,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === protectedManifestUrl) return new Response(JSON.stringify(protectedManifest));
      return new Response(null, { status: 404 });
    },
  });
  await reconstructionStarted;
  const incomingLoad = await loadResumableLc0ModelForOrt(incomingManifestUrl, {
    researchOnly: true,
    cacheName: 'custom-store-isolation-test',
    concurrency: 1,
    corruptionRetries: 0,
    maxCacheEntries: 1,
    maxCacheBytes: Infinity,
    shardStore: incomingStore,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === incomingManifestUrl) return new Response(JSON.stringify(incomingManifest));
      if (url.endsWith('/incoming.bin')) return new Response(incomingBytes);
      return new Response(null, { status: 404 });
    },
  });
  assert.equal(incomingLoad.downloadedShards, 1);
  assert.equal(incomingStore.deleted.includes(protectedHash), true);
  assert.equal(incomingStore.entries.has(protectedHash), false);
  assert.equal(incomingStore.entries.has(incomingHash), true);
  assert.equal(protectedStore.entries.has(protectedHash), true);
  releaseProtectedReconstruction();
  await protectedLoad;
});

test('reconstruction redownloads a shard evicted by another realm', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x4a);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/cross-realm.resumable.json';
  const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
  const hash = manifest.shards[0].sha256;
  const store = new MemoryShardStore([{ sha256: hash, bytes: bytes.buffer }]);
  const originalGet = store.get.bind(store);
  let reads = 0;
  store.get = async (requestedHash) => {
    if (requestedHash === hash) {
      reads += 1;
      if (reads === 2) store.entries.delete(hash);
    }
    return originalGet(requestedHash);
  };
  let shardFetches = 0;
  const result = await loadResumableLc0ModelForOrt(manifestUrl, {
    researchOnly: true,
    cacheName: 'cross-realm-reconstruction-recovery-test',
    concurrency: 1,
    corruptionRetries: 0,
    shardStore: store,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === manifestUrl) return new Response(JSON.stringify(manifest));
      if (url === shardUrl) {
        shardFetches += 1;
        return new Response(bytes);
      }
      return new Response(null, { status: 404 });
    },
  });
  assert.equal(shardFetches, 1);
  assert.equal(result.downloadedShards, 1);
  assert.equal(result.reusedShards, 0);
  assert.equal(result.downloadedBytes, CHUNK_BYTES);
  assert.deepEqual(Buffer.from(result.model), bytes);
});

test('failed loads release process-wide shard protection', async () => {
  const failedBytes = Buffer.alloc(CHUNK_BYTES, 0x4f);
  const replacementBytes = Buffer.alloc(CHUNK_BYTES, 0x50);
  const failedManifest = singleShardManifest(failedBytes, 'shards/failed.bin');
  const replacementManifest = singleShardManifest(replacementBytes, 'shards/replacement.bin');
  const failedManifestUrl = 'https://models.example/research/failed.resumable.json';
  const replacementManifestUrl = 'https://models.example/research/replacement.resumable.json';
  const replacementShardUrl = new URL(replacementManifest.shards[0].url, replacementManifestUrl).href;
  const failedHash = failedManifest.shards[0].sha256;
  const store = new MemoryShardStore([
    { sha256: failedHash, bytes: failedBytes.buffer, lastUsedAt: 1 },
  ]);
  store.touch = async () => {};
  const originalGet = store.get.bind(store);
  let failedReads = 0;
  store.get = async (hash) => {
    if (hash === failedHash) {
      failedReads += 1;
      if (failedReads === 2) throw new Error('forced reconstruction failure');
    }
    return originalGet(hash);
  };
  await assert.rejects(
    loadResumableLc0ModelForOrt(failedManifestUrl, {
      researchOnly: true,
      cacheName: 'failed-load-protection-test',
      concurrency: 1,
      maxCacheEntries: 1,
      maxCacheBytes: Infinity,
      shardStore: store,
      fetchFn: async () => new Response(JSON.stringify(failedManifest)),
    }),
    /forced reconstruction failure/,
  );
  const replacement = await loadResumableLc0ModelForOrt(replacementManifestUrl, {
    researchOnly: true,
    cacheName: 'failed-load-protection-test',
    concurrency: 1,
    corruptionRetries: 0,
    maxCacheEntries: 1,
    maxCacheBytes: Infinity,
    shardStore: store,
    fetchFn: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === replacementManifestUrl) return new Response(JSON.stringify(replacementManifest));
      if (url === replacementShardUrl) return new Response(replacementBytes);
      return new Response(null, { status: 404 });
    },
  });
  assert.equal(replacement.downloadedShards, 1);
  assert.equal(store.deleted.includes(failedHash), true);
  assert.equal(store.entries.has(failedHash), false);
});

test('cached shard hashing contributes to concurrent peak temporary bytes', async () => {
  const first = Buffer.alloc(CHUNK_BYTES, 0x4d);
  const second = Buffer.alloc(CHUNK_BYTES, 0x4e);
  const manifest = {
    schema: 'lc0_browser.resumable_model_shards.v1',
    chunkBytes: CHUNK_BYTES,
    decoded: {
      bytes: first.byteLength + second.byteLength,
      sha256: sha256(Buffer.concat([first, second])),
    },
    shards: [
      { bytes: first.byteLength, sha256: sha256(first), url: 'shards/first.bin' },
      { bytes: second.byteLength, sha256: sha256(second), url: 'shards/second.bin' },
    ],
  };
  const store = new MemoryShardStore([
    { sha256: manifest.shards[0].sha256, bytes: first.buffer },
    { sha256: manifest.shards[1].sha256, bytes: second.buffer },
  ]);
  const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  const subtle = globalThis.crypto.subtle;
  let initialDigests = 0;
  let releaseInitialDigests;
  const bothInitialDigestsStarted = new Promise((resolve) => {
    releaseInitialDigests = resolve;
  });
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        async digest(...args) {
          initialDigests += 1;
          if (initialDigests <= 2) {
            if (initialDigests === 2) releaseInitialDigests();
            await bothInitialDigestsStarted;
          }
          return subtle.digest(...args);
        },
      },
    },
  });
  try {
    const result = await loadResumableLc0ModelForOrt('https://models.example/research/memory.resumable.json', {
      researchOnly: true,
      cacheName: 'cached-memory-telemetry-test',
      concurrency: 2,
      maxCacheEntries: Infinity,
      maxCacheBytes: Infinity,
      shardStore: store,
      fetchFn: async () => new Response(JSON.stringify(manifest)),
    });
    assert.equal(result.reusedShards, 2);
    assert.equal(result.peakTemporaryBytes, 2 * CHUNK_BYTES);
  } finally {
    Object.defineProperty(globalThis, 'crypto', originalCrypto);
  }
});

test('first shard worker failure aborts siblings and waits for their settlement', async () => {
  const first = Buffer.alloc(CHUNK_BYTES, 0x51);
  const second = Buffer.alloc(CHUNK_BYTES, 0x52);
  const manifest = {
    schema: 'lc0_browser.resumable_model_shards.v1',
    chunkBytes: CHUNK_BYTES,
    decoded: {
      bytes: first.byteLength + second.byteLength,
      sha256: sha256(Buffer.concat([first, second])),
    },
    shards: [
      { bytes: first.byteLength, sha256: sha256(first), url: 'shards/fail.bin' },
      { bytes: second.byteLength, sha256: sha256(second), url: 'shards/slow.bin' },
    ],
  };
  const manifestUrl = 'https://models.example/research/model.resumable.json';
  let slowStarted;
  const slowStartedPromise = new Promise((resolve) => {
    slowStarted = resolve;
  });
  let siblingAborted = false;
  let siblingSettled = false;
  await assert.rejects(
    loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      concurrency: 2,
      corruptionRetries: 0,
      shardStore: new MemoryShardStore(),
      fetchFn: async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === manifestUrl) return new Response(JSON.stringify(manifest));
        if (url.endsWith('/fail.bin')) {
          await slowStartedPromise;
          return new Response(null, { status: 503 });
        }
        if (url.endsWith('/slow.bin')) {
          slowStarted();
          return new Promise((resolve, reject) => {
            init.signal.addEventListener('abort', () => {
              siblingAborted = true;
              setTimeout(() => {
                siblingSettled = true;
                reject(new DOMException('The operation was aborted', 'AbortError'));
              }, 10);
            }, { once: true });
          });
        }
        return new Response(null, { status: 404 });
      },
    }),
    /fetch failed.*503/,
  );
  assert.equal(siblingAborted, true);
  assert.equal(siblingSettled, true);
});

test('Cache Storage LRU touch writes metadata once without replaying the shard body', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x61);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/model.resumable.json';
  const hash = manifest.shards[0].sha256;
  const bodyUrl = `http://localhost/__lc0-resumable-model-shards__/sha256/${hash}`;
  const metadataUrl = `http://localhost/__lc0-resumable-model-shards__/metadata/sha256/${hash}`;
  const oldHash = sha256(Buffer.from('old shard'));
  const oldBodyUrl = `http://localhost/__lc0-resumable-model-shards__/sha256/${oldHash}`;
  const oldMetadataUrl = `http://localhost/__lc0-resumable-model-shards__/metadata/sha256/${oldHash}`;
  const cache = new FakeShardCache();
  cache.seed(bodyUrl, bytes, { 'x-lc0-shard-bytes': String(bytes.byteLength) });
  cache.seed(oldBodyUrl, Buffer.from('old shard'), { 'x-lc0-shard-bytes': String('old shard'.length) });
  cache.seed(oldMetadataUrl, JSON.stringify({ bytes: 'old shard'.length, lastUsedAt: 1 }), { 'content-type': 'application/json' });
  const originalCaches = globalThis.caches;
  const originalLocation = globalThis.location;
  globalThis.caches = new FakeShardCacheStorage(cache);
  globalThis.location = { href: 'http://localhost/' };
  try {
    const result = await loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      concurrency: 1,
      maxCacheEntries: 1,
      maxCacheBytes: Infinity,
      fetchFn: async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === manifestUrl) return new Response(JSON.stringify(manifest));
        return new Response(null, { status: 500 });
      },
    });
    assert.equal(result.reusedShards, 1);
    assert.equal(cache.bodyReads, 0, 'bounded cache reads must not call arrayBuffer()');
    assert(cache.bodyStreamReads > 0, 'verification and reconstruction stream the cached shard');
    assert.equal(cache.bodyPuts, 0, 'touch must not rewrite the shard body');
    assert.equal(cache.metadataPuts, 1, 'verified shard load refreshes LRU once');
    assert.equal(cache.entries.has(oldBodyUrl), false, 'eviction deletes the old shard body');
    assert.equal(cache.entries.has(oldMetadataUrl), false, 'eviction deletes matching metadata');
    assert.equal(cache.entries.has(metadataUrl), true);
    const cleared = await clearResumableLc0ModelShardCache({ researchOnly: true });
    assert.equal(cleared.removedEntries, 1, 'cleanup does not count metadata keys as shard bodies');
    assert.equal(cache.entries.size, 0);
  } finally {
    globalThis.caches = originalCaches;
    globalThis.location = originalLocation;
  }
});

test('Cache Storage metadata put failure does not invalidate a persisted shard body', async () => {
  const bytes = Buffer.of(0x64);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/metadata-put-failure.resumable.json';
  const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
  const hash = manifest.shards[0].sha256;
  const bodyUrl = `http://localhost/__lc0-resumable-model-shards__/sha256/${hash}`;
  const metadataUrl = `http://localhost/__lc0-resumable-model-shards__/metadata/sha256/${hash}`;
  const cache = new FakeShardCache();
  cache.failMetadataPuts = true;
  const originalCaches = globalThis.caches;
  const originalLocation = globalThis.location;
  globalThis.caches = new FakeShardCacheStorage(cache);
  globalThis.location = { href: 'http://localhost/' };
  try {
    const result = await loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      concurrency: 1,
      corruptionRetries: 0,
      maxCacheEntries: Infinity,
      maxCacheBytes: Infinity,
      fetchFn: async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === manifestUrl) return new Response(JSON.stringify(manifest));
        if (url === shardUrl) return new Response(bytes);
        return new Response(null, { status: 404 });
      },
    });
    assert.deepEqual(Buffer.from(result.model), bytes);
    assert.equal(result.downloadedShards, 1);
    assert.equal(cache.bodyPuts, 1);
    assert.equal(cache.metadataPuts, 1);
    assert.equal(cache.entries.has(bodyUrl), true);
    assert.equal(cache.entries.has(metadataUrl), false);
    assert.equal(cache.entries.get(bodyUrl).headers.get('x-lc0-shard-bytes'), String(bytes.byteLength));
  } finally {
    globalThis.caches = originalCaches;
    globalThis.location = originalLocation;
  }
});

test('oversized Cache Storage shard bodies are bounded, evicted, and redownloaded', async () => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x62);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/oversized-cache.resumable.json';
  const shardUrl = new URL(manifest.shards[0].url, manifestUrl).href;
  const hash = manifest.shards[0].sha256;
  const bodyUrl = `http://localhost/__lc0-resumable-model-shards__/sha256/${hash}`;
  const cache = new FakeShardCache();
  cache.seed(bodyUrl, Buffer.alloc(CHUNK_BYTES + 1, 0x63), {
    'x-lc0-shard-bytes': String(CHUNK_BYTES),
  });
  const originalCaches = globalThis.caches;
  const originalLocation = globalThis.location;
  globalThis.caches = new FakeShardCacheStorage(cache);
  globalThis.location = { href: 'http://localhost/' };
  let shardFetches = 0;
  try {
    const result = await loadResumableLc0ModelForOrt(manifestUrl, {
      researchOnly: true,
      concurrency: 1,
      corruptionRetries: 0,
      fetchFn: async (input) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url === manifestUrl) return new Response(JSON.stringify(manifest));
        if (url === shardUrl) {
          shardFetches += 1;
          return new Response(bytes);
        }
        return new Response(null, { status: 404 });
      },
    });
    assert.equal(cache.bodyReads, 0);
    assert.equal(cache.deleted.includes(bodyUrl), true);
    assert.equal(shardFetches, 1);
    assert.equal(result.corruptShardsEvicted, 1);
    assert.equal(result.downloadedShards, 1);
    assert.deepEqual(Buffer.from(result.model), bytes);
  } finally {
    globalThis.caches = originalCaches;
    globalThis.location = originalLocation;
  }
});

test('late aborts are honored after reconstruction, full-model hash, and cache cleanup', async (t) => {
  const bytes = Buffer.alloc(CHUNK_BYTES, 0x71);
  const manifest = singleShardManifest(bytes);
  const manifestUrl = 'https://models.example/research/model.resumable.json';

  await t.test('after reconstruction', async () => {
    const controller = new AbortController();
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        concurrency: 1,
        signal: controller.signal,
        shardStore: new MemoryShardStore([{ sha256: manifest.shards[0].sha256, bytes: bytes.buffer }]),
        fetchFn: async () => new Response(JSON.stringify(manifest)),
        onProgress(progress) {
          if (progress.phase === 'reconstruct' && progress.completedShards === progress.totalShards) controller.abort();
        },
      }),
      { name: 'AbortError' },
    );
  });

  await t.test('after full-model sha256', async () => {
    const controller = new AbortController();
    const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
    const subtle = globalThis.crypto.subtle;
    let digests = 0;
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: {
        subtle: {
          async digest(...args) {
            const digest = await subtle.digest(...args);
            digests += 1;
            if (digests === 3) controller.abort();
            return digest;
          },
        },
      },
    });
    try {
      await assert.rejects(
        loadResumableLc0ModelForOrt(manifestUrl, {
          researchOnly: true,
          concurrency: 1,
          signal: controller.signal,
          shardStore: new MemoryShardStore([{ sha256: manifest.shards[0].sha256, bytes: bytes.buffer }]),
          fetchFn: async () => new Response(JSON.stringify(manifest)),
        }),
        { name: 'AbortError' },
      );
      assert.equal(digests, 3);
    } finally {
      Object.defineProperty(globalThis, 'crypto', originalCrypto);
    }
  });

  await t.test('after cache cleanup', async () => {
    const controller = new AbortController();
    const store = new MemoryShardStore([{ sha256: manifest.shards[0].sha256, bytes: bytes.buffer }]);
    let lists = 0;
    store.list = async () => {
      lists += 1;
      if (lists === 2) controller.abort();
      return [...store.entries].map(([sha256Value, entry]) => ({
        sha256: sha256Value,
        bytes: entry.bytes.byteLength,
        lastUsedAt: entry.lastUsedAt,
      }));
    };
    await assert.rejects(
      loadResumableLc0ModelForOrt(manifestUrl, {
        researchOnly: true,
        concurrency: 1,
        signal: controller.signal,
        shardStore: store,
        fetchFn: async () => new Response(JSON.stringify(manifest)),
      }),
      { name: 'AbortError' },
    );
    assert.equal(lists, 2);
  });
});

test('abort around 40% persists completed shards and resume does not redownload them', async () => {
  const entry = await fixture();
  const controller = new AbortController();
  let abortCompletedShards = 0;
  await assert.rejects(
    loadResumableLc0ModelForOrt(pathToFileURL(entry.manifestPath).href, loaderOptions(entry, {
      concurrency: 1,
      signal: controller.signal,
      onProgress(progress) {
        if (progress.phase === 'download' && progress.completedBytes >= progress.totalBytes * 0.4) {
          abortCompletedShards = progress.completedShards;
          controller.abort();
        }
      },
    })),
    { name: 'AbortError' },
  );
  assert.equal(abortCompletedShards, 3);
  assert.equal((await readdir(entry.cacheDir)).length, 3);

  const fetched = [];
  const fileFetch = localFileFetch();
  const resumed = await loadResumableLc0ModelForOrt(pathToFileURL(entry.manifestPath).href, loaderOptions(entry, {
    concurrency: 2,
    fetchFn: async (input, init) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url.endsWith('.bin')) fetched.push(url);
      return fileFetch(input, init);
    },
  }));
  assert.equal(resumed.reusedShards, 3);
  assert.equal(resumed.downloadedShards, 3);
  assert.equal(fetched.length, 3);
  assert.equal(resumed.sha256, entry.expectedSha256);
});

test('cached corruption is evicted and only the corrupt shard is redownloaded', async () => {
  const entry = await fixture();
  const manifestUrl = pathToFileURL(entry.manifestPath).href;
  await loadResumableLc0ModelForOrt(manifestUrl, loaderOptions(entry));
  const corruptHash = entry.manifest.shards[2].sha256;
  await writeFile(join(entry.cacheDir, `${corruptHash}.bin`), Buffer.alloc(CHUNK_BYTES, 0x5a));

  let shardFetches = 0;
  const fileFetch = localFileFetch();
  const repaired = await loadResumableLc0ModelForOrt(manifestUrl, loaderOptions(entry, {
    fetchFn: async (input, init) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url.endsWith('.bin')) shardFetches += 1;
      return fileFetch(input, init);
    },
  }));
  assert.equal(repaired.corruptShardsEvicted, 1);
  assert.equal(repaired.downloadedShards, 1);
  assert.equal(repaired.reusedShards, 5);
  assert.equal(shardFetches, 1);
  assert.equal(repaired.sha256, entry.expectedSha256);
});

test('corrupt network shard is retried, persisted only when valid, and ordered reconstruction validates', async () => {
  const entry = await fixture();
  const manifestUrl = pathToFileURL(entry.manifestPath).href;
  const targetUrl = new URL(entry.manifest.shards[1].url, manifestUrl).href;
  const attempts = new Map();
  const fileFetch = localFileFetch();
  const result = await loadResumableLc0ModelForOrt(manifestUrl, loaderOptions(entry, {
    concurrency: 2,
    fetchFn: async (input, init) => {
      const url = typeof input === 'string' ? input : input.url;
      const count = (attempts.get(url) ?? 0) + 1;
      attempts.set(url, count);
      if (url === targetUrl && count === 1) return new Response(Buffer.alloc(CHUNK_BYTES, 0xff));
      return fileFetch(input, init);
    },
  }));
  assert.equal(attempts.get(targetUrl), 2);
  assert.equal(result.corruptionRetries, 1);
  assert.equal(result.downloadedBytes, MODEL_BYTES + CHUNK_BYTES);
  assert.equal(result.sha256, entry.expectedSha256);

  const reordered = structuredClone(entry.manifest);
  [reordered.shards[0], reordered.shards[1]] = [reordered.shards[1], reordered.shards[0]];
  const reversedPath = join(entry.root, 'published', 'reordered.resumable.json');
  await writeFile(reversedPath, JSON.stringify(reordered));
  await assert.rejects(
    loadResumableLc0ModelForOrt(pathToFileURL(reversedPath).href, loaderOptions(entry)),
    /decoded sha256 mismatch/,
  );
});

test('duplicate content hashes fetch once, reconstruct every ordered reference, and stay memory-bounded', async () => {
  const entry = await fixture();
  const duplicate = structuredClone(entry.manifest);
  duplicate.shards = [
    entry.manifest.shards[0],
    entry.manifest.shards[0],
    entry.manifest.shards.at(-1),
  ];
  const first = await readFile(new URL(entry.manifest.shards[0].url, pathToFileURL(entry.manifestPath)));
  const last = await readFile(new URL(entry.manifest.shards.at(-1).url, pathToFileURL(entry.manifestPath)));
  const decoded = Buffer.concat([first, first, last]);
  duplicate.decoded = {
    bytes: decoded.byteLength,
    sha256: createHash('sha256').update(decoded).digest('hex'),
  };
  const duplicatePath = join(entry.root, 'published', 'duplicate.resumable.json');
  await writeFile(duplicatePath, JSON.stringify(duplicate));
  const duplicateCache = join(entry.root, 'duplicate-cache');

  let activeFetches = 0;
  let peakFetches = 0;
  let shardFetches = 0;
  const progress = [];
  const fileFetch = localFileFetch();
  const result = await loadResumableLc0ModelForOrt(pathToFileURL(duplicatePath).href, {
    researchOnly: true,
    concurrency: 2,
    fetchFn: async (input, init) => {
      const url = typeof input === 'string' ? input : input.url;
      if (!url.endsWith('.bin')) return fileFetch(input, init);
      shardFetches += 1;
      activeFetches += 1;
      peakFetches = Math.max(peakFetches, activeFetches);
      try {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return await fileFetch(input, init);
      } finally {
        activeFetches -= 1;
      }
    },
    shardStore: new FileModelShardStore(duplicateCache),
    onProgress(update) {
      progress.push(update);
    },
  });
  assert.equal(result.uniqueShardCount, 2);
  assert.equal(result.deduplicatedReferences, 1);
  assert.equal(result.downloadedShards, 2);
  assert.equal(shardFetches, 2);
  assert(peakFetches <= 2);
  assert(result.peakTemporaryBytes <= 2 * CHUNK_BYTES);
  assert.deepEqual(Buffer.from(result.model), decoded);
  assert(progress.filter((update) => update.phase === 'download').every((update) => update.totalShards === 2));
  assert(progress.filter((update) => update.phase === 'reconstruct').every((update) => update.totalShards === 3));
});

test('persistent shard cache applies explicit bounds and supports research-only cleanup', async () => {
  const entry = await fixture();
  const store = new FileModelShardStore(entry.cacheDir);
  const result = await loadResumableLc0ModelForOrt(pathToFileURL(entry.manifestPath).href, loaderOptions(entry, {
    shardStore: store,
    maxCacheEntries: 2,
    maxCacheBytes: 2 * CHUNK_BYTES,
  }));
  assert.equal(result.sha256, entry.expectedSha256);
  const retained = await store.list();
  assert.equal(retained.length, 2);
  assert(retained.reduce((sum, shard) => sum + shard.bytes, 0) <= 2 * CHUNK_BYTES);
  const cleared = await clearResumableLc0ModelShardCache({ researchOnly: true, shardStore: store });
  assert.equal(cleared.removedEntries, 2);
  assert.equal((await store.list()).length, 0);
  await assert.rejects(
    clearResumableLc0ModelShardCache({ researchOnly: false, shardStore: store }),
    /researchOnly: true/,
  );
});

test('benchmark creates a fresh run directory when the work directory is reused', async () => {
  const workDir = await mkdtemp(join(tmpdir(), 'lc0-resumable-bench-'));
  temporaryRoots.add(workDir);
  const first = await createBenchmarkRunDirectory(workDir);
  const second = await createBenchmarkRunDirectory(workDir);
  assert.notEqual(first, second);
  assert.equal(first.startsWith(`${workDir}/run-`), true);
  assert.equal(second.startsWith(`${workDir}/run-`), true);
});
