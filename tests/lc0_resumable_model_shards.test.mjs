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
import { generateResumableModelShards } from '../scripts/generate_resumable_model_shards.mjs';
import { FileModelShardStore, localFileFetch } from '../scripts/reconstruct_resumable_model_shards.mjs';

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
  assert(result.peakTemporaryBytes <= 3 * CHUNK_BYTES);
  await assertMatchesFile(result.model, entry.modelPath);
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
