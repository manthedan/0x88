import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadLc0ModelForOrt, verifyLc0ModelBytes } from '../src/lc0/modelCache.ts';

const MODEL_URL = 'https://example.test/models/custom.onnx';
const SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

async function withEnvironment(run) {
  const previous = { fetch: globalThis.fetch, caches: globalThis.caches, navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator') };
  let writes = 0;
  let modelFetches = 0;
  globalThis.caches = { open: async () => ({ match: async () => undefined, put: async () => writes++, keys: async () => [] }) };
  globalThis.fetch = async (input) => {
    if ((input.url ?? input) === MODEL_URL) {
      modelFetches++;
      return new Response('abc');
    }
    return new Response(null, { status: 503 });
  };
  try {
    await run({ writes: () => writes, modelFetches: () => modelFetches });
  } finally {
    globalThis.fetch = previous.fetch;
    globalThis.caches = previous.caches;
    if (previous.navigator) Object.defineProperty(globalThis, 'navigator', previous.navigator);
    else delete globalThis.navigator;
  }
}

test('model verifier rejects missing integrity metadata and unavailable WebCrypto', async () => {
  const bytes = new TextEncoder().encode('abc').buffer;
  assert.equal((await verifyLc0ModelBytes(bytes, { expectedBytes: 3 })).ok, false);
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
  try {
    assert.equal((await verifyLc0ModelBytes(bytes, { expectedSha256: SHA256 })).ok, false);
  } finally {
    Object.defineProperty(globalThis, 'crypto', previous);
  }
});

for (const cache of [true, false]) {
  test(`missing expectations fail before model use or persistent cache writes (cache=${cache}) and can be retried`, async () => {
    await withEnvironment(async (counts) => {
      const options = { cache, manifestUrl: 'https://example.test/manifest.json', channelUrl: 'https://example.test/channels/stable.json' };
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, options), /integrity metadata.*retry/i);
      assert.equal(counts.writes(), 0);
      assert.equal(counts.modelFetches(), 0);
      const fetchBytes = globalThis.fetch;
      globalThis.fetch = async (input, init) =>
        (input.url ?? input) === options.manifestUrl
          ? Response.json({ models: [{ file: 'custom.onnx', url: MODEL_URL, bytes: 3, sha256: SHA256 }] })
          : fetchBytes(input, init);
      const result = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(result.sha256Valid, true);
      assert.deepEqual(new Uint8Array(result.model), new TextEncoder().encode('abc'));
    });
  });
}

for (const failure of ['network', 'invalid-json', 'invalid-schema', 'no-hash']) {
  test(`model metadata ${failure} failure cannot admit unverified bytes`, async () => {
    await withEnvironment(async (counts) => {
      const fetchBytes = globalThis.fetch;
      globalThis.fetch = async (input) => {
        if ((input.url ?? input) === MODEL_URL) return fetchBytes(input);
        if (failure === 'network') throw new TypeError('offline');
        if (failure === 'invalid-json') return new Response('{');
        return Response.json(failure === 'invalid-schema' ? { models: 'invalid' } : { models: [{ file: 'custom.onnx', url: MODEL_URL, bytes: 3 }] });
      };
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, { cache: true }), /integrity metadata.*retry/i);
      assert.equal(counts.writes(), 0);
      assert.equal(counts.modelFetches(), 0);
    });
  });
}

test('lab opt-out keeps arbitrary local-model loads usable without persisting unverified bytes', async () => {
  await withEnvironment(async (counts) => {
    const result = await loadLc0ModelForOrt(MODEL_URL, { cache: true, allowUnverified: true, onProgress: () => {} });
    assert.equal(result.mode, 'memory');
    assert.equal(result.sha256Valid, undefined);
    assert.deepEqual(new Uint8Array(result.model), new TextEncoder().encode('abc'));
    assert.equal(counts.writes(), 0);
    const direct = await loadLc0ModelForOrt(MODEL_URL, { cache: false, allowUnverified: true });
    assert.equal(direct.model, MODEL_URL);
  });
});

test('an opt-out flight cannot downgrade a concurrent product load', async () => {
  await withEnvironment(async (counts) => {
    const [lab, product] = await Promise.allSettled([
      loadLc0ModelForOrt(MODEL_URL, { cache: true, allowUnverified: true, onProgress: () => {} }),
      loadLc0ModelForOrt(MODEL_URL, { cache: true }),
    ]);
    assert.equal(lab.status, 'fulfilled');
    assert.equal(product.status, 'rejected');
    assert.match(product.reason.message, /integrity metadata.*retry/i);
    assert.equal(counts.writes(), 0);
  });
});

for (const fallback of ['cache-disabled', 'no-cache-api', 'quota', 'stale-quota']) {
  test(`product ${fallback} fallback verifies bytes and rejects corruption`, async () => {
    await withEnvironment(async (counts) => {
      if (fallback === 'no-cache-api') delete globalThis.caches;
      if (fallback.includes('quota')) {
        Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { storage: { estimate: async () => ({ usage: 1000, quota: 1000 }) } } });
      }
      if (fallback === 'stale-quota') {
        const open = globalThis.caches.open;
        globalThis.caches.open = async () => ({ ...(await open()), match: async () => new Response('bad'), delete: async () => true });
      }
      const options = { cache: fallback !== 'cache-disabled', manifestEntry: { file: 'custom.onnx', url: MODEL_URL, bytes: 3, sha256: SHA256 } };
      const result = await loadLc0ModelForOrt(MODEL_URL, options);
      assert.equal(result.mode, 'memory');
      assert.equal(result.sha256Valid, true);
      assert.equal(counts.writes(), 0);
      globalThis.fetch = async () => new Response('bad');
      await assert.rejects(loadLc0ModelForOrt(MODEL_URL, options), /sha256 mismatch/);
      assert.equal(counts.writes(), 0);
    });
  });
}

test('lab opt-out cannot persist unhashed bytes when WebCrypto is unavailable', async () => {
  await withEnvironment(async (counts) => {
    const previous = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
    try {
      await assert.rejects(
        loadLc0ModelForOrt(MODEL_URL, { cache: true, manifestEntry: { file: 'custom.onnx', url: MODEL_URL, sha256: SHA256 } }),
        /SHA-256 verification is unavailable.*retry/i,
      );
      assert.equal(counts.writes(), 0);
      const result = await loadLc0ModelForOrt(MODEL_URL, {
        cache: true,
        allowUnverified: true,
        onProgress: () => {},
        manifestEntry: { file: 'custom.onnx', url: MODEL_URL, sha256: SHA256 },
      });
      assert.equal(result.mode, 'memory');
      assert.equal(counts.writes(), 0);
    } finally {
      Object.defineProperty(globalThis, 'crypto', previous);
    }
  });
});
