import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { SquareformerTvmjsWebgpuEvaluator } from '../src/nn/squareformerTvmjsWebgpuEvaluator.ts';
import { CENTIPAWN_TVMJS_WEBGPU_V2, promotedTvmjsRuntimeForModel } from '../src/nn/runtimeRegistry.ts';

const prefix = '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/';
const base = {
  schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
  modelFamily: 'bt4-soap-rem-c19000-final',
  dtype: 'f32',
  target: 'webgpu',
  requiredFeatures: ['webgpu'],
  runtime: { tvmjsBundle: 'tvmjs.bundle.js' },
  models: [{ batch: 16, wasm: 'model.wasm' }],
  files: [{ path: 'tvmjs.bundle.js', sha256: '0'.repeat(64) }],
};
async function withGpu(
  {
    features = true,
    min = 32,
    max = 32,
    info,
    assetBase,
    manifest = { ...base, version: 'v3-subgroups', requiredFeatures: ['webgpu', 'subgroups'], subgroupSize: 32 },
    failPipelines = false,
  },
  fn,
) {
  const restores = [];
  function replace(name, value) {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    restores.push(() => (descriptor ? Object.defineProperty(globalThis, name, descriptor) : delete globalThis[name]));
  }
  const calls = { fetches: [], devices: [], destroyed: 0, disposed: 0 };
  replace('location', new URL('https://app.example/analysis'));
  if (assetBase) replace('LC0_BROWSER_ASSET_BASE_URL', assetBase);
  replace('fetch', async (url) => {
    calls.fetches.push(String(url));
    if (String(url).endsWith('/manifest.json'))
      return { ok: true, json: async () => (String(url).includes('/v3-subgroups/') ? manifest : { ...base, version: 'v2-shape-k16' }) };
    return { ok: true, arrayBuffer: async () => new ArrayBuffer(1) };
  });
  replace('navigator', {
    gpu: {
      requestAdapter: async () => ({
        features: { has: (name) => name === 'subgroups' && features },
        info: info === undefined ? { subgroupMinSize: min, subgroupMaxSize: max } : info,
        limits: {},
        requestDevice: async (descriptor) => {
          calls.devices.push(descriptor);
          return { destroy: () => calls.destroyed++ };
        },
      }),
    },
  });
  replace('__LC0_TVMJS_BUNDLE_SHA256__', '0'.repeat(64));
  replace('tvmjs', {
    createPolyfillWASI: () => ({}),
    instantiate: async () => ({
      initWebGPU() {},
      beginScope() {},
      endScope() {},
      systemLib() {},
      async asyncLoadWebGPUPipelines() {
        if (failPipelines) throw new Error('pipeline failure');
      },
      webgpu() {},
      createVirtualMachine: () => ({ getInternalModule: () => ({}) }),
      dispose: () => calls.disposed++,
    }),
  });
  try {
    await fn(calls);
  } finally {
    for (const restore of restores.reverse()) restore();
  }
}

test('subgroups requested only for v3 with a fixed size-32 adapter', async () => {
  await withGpu({}, async (calls) => {
    const evaluator = await SquareformerTvmjsWebgpuEvaluator.create(`${prefix}v3-subgroups/manifest.json`, {});
    assert.deepEqual(calls.devices[0].requiredFeatures, ['subgroups']);
    assert.match(evaluator.manifestUrl, /v3-subgroups\/manifest.json$/);
    evaluator.destroy();
    await Promise.resolve();
    assert.equal(calls.destroyed, 1);
  });
  await withGpu({}, async (calls) => {
    const evaluator = await SquareformerTvmjsWebgpuEvaluator.create(`${prefix}v2-shape-k16/manifest.json`, {});
    assert.deepEqual(calls.devices[0].requiredFeatures, []);
    evaluator.destroy();
    await Promise.resolve();
  });
});

for (const adapter of [{ features: false }, { min: 16, max: 32 }, { min: 32, max: 64 }, { info: {} }, { info: null }]) {
  test(`v3 falls back to v2 before model/device creation: ${JSON.stringify(adapter)}`, async () => {
    await withGpu(adapter, async (calls) => {
      const evaluator = await SquareformerTvmjsWebgpuEvaluator.create(`${prefix}v3-subgroups/manifest.json`, {});
      assert.equal(calls.devices.length, 1);
      assert.deepEqual(calls.devices[0].requiredFeatures, []);
      assert.match(evaluator.manifestUrl, /v2-shape-k16\/manifest.json$/);
      assert.ok(calls.fetches.some((url) => url.endsWith('/v2-shape-k16/manifest.json')));
      assert.ok(!calls.fetches.some((url) => url.endsWith('/v3-subgroups/model.wasm')));
      evaluator.destroy();
      await Promise.resolve();
    });
  });
}

test('v2 fallback preserves same-origin and configured CDN asset-base path prefixes', async () => {
  for (const baseUrl of ['https://app.example/preview', 'https://assets.example/preview']) {
    await withGpu({ features: false, assetBase: baseUrl }, async (calls) => {
      const evaluator = await SquareformerTvmjsWebgpuEvaluator.create(`${baseUrl}${prefix}v3-subgroups/manifest.json`, {});
      const selected = `${baseUrl}${prefix}v2-shape-k16/manifest.json`;
      assert.equal(evaluator.manifestUrl, selected);
      assert.ok(calls.fetches.includes(selected));
      assert.ok(calls.fetches.every((url) => url.startsWith(`${baseUrl}/`)));
      assert.deepEqual(calls.devices[0].requiredFeatures, []);
      evaluator.destroy();
      await Promise.resolve();
    });
  }
});

test('reject mismatched subgroup manifest contracts', async () => {
  for (const manifest of [
    { ...base, version: 'v3-subgroups' },
    { ...base, version: 'v2-shape-k16', requiredFeatures: ['webgpu', 'subgroups'] },
    { ...base, version: 'v3-subgroups', requiredFeatures: ['webgpu', 'subgroups'], subgroupSize: 16 },
  ])
    await withGpu({ manifest }, async (calls) => {
      await assert.rejects(SquareformerTvmjsWebgpuEvaluator.create(`${prefix}v3-subgroups/manifest.json`, {}), /Only the v3-subgroups artifact/);
      assert.equal(calls.devices.length, 0);
    });
});

test('subgroup pipeline initialization failure disposes runtime and device', async () => {
  await withGpu({ failPipelines: true }, async (calls) => {
    await assert.rejects(SquareformerTvmjsWebgpuEvaluator.create(`${prefix}v3-subgroups/manifest.json`, {}), /pipeline failure/);
    assert.equal(calls.destroyed, 1);
    assert.equal(calls.disposed, 1);
  });
});

test('v3 is experimental, all staged files match their manifest, default remains v2', () => {
  const dir = `public${prefix}v3-subgroups/`;
  const manifest = JSON.parse(readFileSync(`${dir}manifest.json`));
  assert.equal(manifest.status, 'experimental');
  assert.equal(manifest.subgroupSize, 32);
  assert.deepEqual(manifest.requiredFeatures, ['webgpu', 'subgroups']);
  assert.equal(promotedTvmjsRuntimeForModel(manifest.modelFamily), CENTIPAWN_TVMJS_WEBGPU_V2);
  for (const file of manifest.files) {
    const bytes = readFileSync(`${dir}${file.path}`);
    assert.equal(bytes.length, file.bytes);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), file.sha256);
  }
});
