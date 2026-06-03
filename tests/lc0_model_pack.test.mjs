import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { loadLc0WebModelPack, validateLc0WebModelPackManifest } from '../src/lc0/modelPack.ts';

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function response(body, init = {}) {
  return new Response(body, { status: 200, ...init });
}

function makeTinyPack(shardBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])) {
  const tensorA = shardBytes.slice(0, 4);
  const tensorB = shardBytes.slice(4, 8);
  const manifest = {
    format: 'lc0web',
    version: 1,
    model: { name: 'tiny-pack', family: 'lc0', recommendedRuntime: 'custom-webgpu', layout: 'raw-f16' },
    graph: {
      inputs: [{ name: '/input/planes', dtype: 'f16', shape: [1, 112, 8, 8] }],
      outputs: [{ name: '/output/policy', dtype: 'f16', shape: [1, 1858] }],
      nodes: [],
    },
    weights: {
      totalTensorBytes: 8,
      tensorCount: 2,
      shards: [{ file: 'weights.000.bin', bytes: shardBytes.byteLength, sha256: sha256(shardBytes) }],
      tensors: [
        { name: 'a', dtype: 'f16', shape: [2], shard: 'weights.000.bin', byteOffset: 0, byteLength: 4, sha256: sha256(tensorA) },
        { name: 'b', dtype: 'f16', shape: [2], shard: 'weights.000.bin', byteOffset: 4, byteLength: 4, sha256: sha256(tensorB) },
      ],
    },
  };
  return { manifest, shardBytes };
}

function makeFetch(manifest, shardBytes) {
  return async (url) => {
    if (String(url) === 'https://example.test/pack/model.lc0web.json') {
      return response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } });
    }
    if (String(url) === 'https://example.test/pack/weights.000.bin') {
      return response(shardBytes);
    }
    return new Response('not found', { status: 404 });
  };
}

test('lc0web pack manifest validation catches bad tensor ranges', () => {
  const { manifest } = makeTinyPack();
  validateLc0WebModelPackManifest(manifest);
  const invalid = structuredClone(manifest);
  invalid.weights.tensors[0].byteOffset = 100;
  assert.throws(() => validateLc0WebModelPackManifest(invalid), /range exceeds shard/);
});

test('lc0web pack loader fetches and verifies shard-backed tensor views', async () => {
  const { manifest, shardBytes } = makeTinyPack();
  const pack = await loadLc0WebModelPack('https://example.test/pack/model.lc0web.json', {
    fetchFn: makeFetch(manifest, shardBytes),
  });
  assert.equal(pack.manifest.model.name, 'tiny-pack');
  assert.equal(pack.verifiedShards.length, 1);
  assert.deepEqual([...pack.tensors.get('a').bytes], [1, 2, 3, 4]);
  assert.deepEqual([...pack.tensors.get('b').bytes], [5, 6, 7, 8]);
});

test('lc0web pack loader can load only requested tensors', async () => {
  const { manifest, shardBytes } = makeTinyPack();
  const pack = await loadLc0WebModelPack('https://example.test/pack/model.lc0web.json', {
    fetchFn: makeFetch(manifest, shardBytes),
    tensorNames: ['b'],
  });
  assert.equal(pack.tensors.size, 1);
  assert.deepEqual([...pack.tensors.get('b').bytes], [5, 6, 7, 8]);
});

test('lc0web pack loader rejects corrupt shard bytes', async () => {
  const { manifest } = makeTinyPack();
  const corruptShard = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 9]);
  await assert.rejects(
    () => loadLc0WebModelPack('https://example.test/pack/model.lc0web.json', {
      fetchFn: makeFetch(manifest, corruptShard),
    }),
    /sha256 mismatch/,
  );
});

const GENERATED_PACK_DIR = '../models/lc0-bestnets/lc0web/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web';

test('generated batch-8 lc0web pack metadata and first shard verify', { skip: existsSync(GENERATED_PACK_DIR) ? false : `missing ${GENERATED_PACK_DIR}` }, async () => {
  const fetchFn = async (url) => {
    const name = String(url).split('/').pop();
    const path = `${GENERATED_PACK_DIR}/${name}`;
    if (!existsSync(path)) return new Response('not found', { status: 404 });
    return response(readFileSync(path));
  };
  const pack = await loadLc0WebModelPack('https://example.test/pack/model.lc0web.json', {
    fetchFn,
    tensorNames: ['/const/smolgen_w'],
  });
  assert.equal(pack.manifest.model.sourceSha256, '4a3d0b0ee3080c36d1f18c4a1acbdb0ef9751bb32fc33f28db032f4c230418eb');
  assert.equal(pack.manifest.weights.tensorCount, 375);
  assert.equal(pack.verifiedShards.length, 1);
  assert.equal(pack.tensors.get('/const/smolgen_w').bytes.byteLength, 256 * 4096 * 2);
});
