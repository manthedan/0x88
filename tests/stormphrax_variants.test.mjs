import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  STORMPHRAX_EMSCRIPTEN_VARIANT,
  STORMPHRAX_RELAXED_VARIANT,
  STORMPHRAX_MAIN_NETWORK,
  checkStormphraxVariantAsset,
  defaultStormphraxVariantKey,
  hasExplicitStormphraxVariant,
  normalizeStormphraxVariant,
  resolveDefaultStormphraxVariantAssetFallback,
  stormphraxVariantByKey,
  stormphraxVariantFromParams,
} from '../src/lc0/stormphraxVariants.ts';
import { stormphraxSearchTimeoutMs } from '../src/lc0/stormphraxEngine.ts';
import { supportsWasmRelaxedSimd } from '../src/lc0/wasmFeatures.ts';

test('Stormphrax variant metadata pins the browser sidecars and undertown network', () => {
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.key, 'emscripten');
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.jsUrl, '/stormphrax/stormphrax-emscripten.js');
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.wasmUrl, '/stormphrax/stormphrax-emscripten.wasm');
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.dataUrl, '/stormphrax/stormphrax-emscripten.data');
  assert.equal(STORMPHRAX_MAIN_NETWORK, 'undertown.nnue');
  assert.equal(STORMPHRAX_RELAXED_VARIANT.key, 'emscripten-relaxed');
  assert.equal(STORMPHRAX_RELAXED_VARIANT.wasmUrl, '/stormphrax/stormphrax-emscripten-relaxed-simd128.wasm');
  assert.equal(defaultStormphraxVariantKey(), supportsWasmRelaxedSimd() ? 'emscripten-relaxed' : 'emscripten');
});

test('Stormphrax in-flight asset checks notify callbacks attached by a remount', async () => {
  const originalFetch = globalThis.fetch;
  const resolvers = [];
  globalThis.fetch = () => new Promise((resolve) => resolvers.push(resolve));
  const variant = {
    ...STORMPHRAX_EMSCRIPTEN_VARIANT,
    key: 'custom',
    jsUrl: '/stormphrax/remount-test.js',
    wasmUrl: '/stormphrax/remount-test.wasm',
    dataUrl: '/stormphrax/remount-test.data',
  };
  let firstNotifications = 0;
  let remountNotifications = 0;
  try {
    let remount;
    const first = checkStormphraxVariantAsset(variant, () => {
      firstNotifications += 1;
      if (firstNotifications === 1) remount = checkStormphraxVariantAsset(variant, () => { remountNotifications += 1; });
    });
    assert.equal(resolvers.length, 3, 're-entrant checking callback must reuse the registered probe');
    for (const resolve of resolvers) resolve({ ok: true });
    assert.equal(await first, 'present');
    assert.equal(await remount, 'present');
    assert.equal(firstNotifications, 2);
    assert.equal(remountNotifications, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Stormphrax missing default relaxed assets fall back to baseline SIMD', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false });
  try {
    assert.equal(await resolveDefaultStormphraxVariantAssetFallback(STORMPHRAX_RELAXED_VARIANT, false), STORMPHRAX_EMSCRIPTEN_VARIANT);
    assert.equal(await resolveDefaultStormphraxVariantAssetFallback(STORMPHRAX_RELAXED_VARIANT, true), STORMPHRAX_RELAXED_VARIANT);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Stormphrax search timeout leaves startup headroom above movetime', () => {
  assert.equal(stormphraxSearchTimeoutMs({ movetimeMs: 60_000 }), 75_000);
  assert.equal(stormphraxSearchTimeoutMs({ movetimeMs: 1_000 }), 60_000);
  assert.equal(stormphraxSearchTimeoutMs({ depth: 12 }), 120_000);
});

test('Stormphrax corresponding-source recipe rebuilds baseline and relaxed outputs from clean trees', () => {
  const source = readFileSync(new URL('../scripts/write_engine_source_archive.mjs', import.meta.url), 'utf8');
  assert.match(source, /stormphrax-baseline/);
  assert.match(source, /stormphrax-relaxed/);
  assert.match(source, /\$\{config\.envPrefix\}_WASM_RELAXED_SIMD=1/);
  assert.match(source, /stormphrax-emscripten-relaxed-simd128\.js/);
});

test('Stormphrax variant normalization and same-origin overrides are stable', () => {
  assert.equal(normalizeStormphraxVariant('browser worker'), 'emscripten');
  assert.equal(normalizeStormphraxVariant('relaxed SIMD'), 'emscripten-relaxed');
  assert.equal(normalizeStormphraxVariant('custom'), 'custom');
  assert.equal(stormphraxVariantByKey('unknown').label, 'Stormphrax 8');
  assert.equal(hasExplicitStormphraxVariant(new URLSearchParams('')), false);
  assert.equal(hasExplicitStormphraxVariant(new URLSearchParams('stormphrax=emscripten')), true);
  const custom = stormphraxVariantFromParams(new URLSearchParams('stormphraxJs=/stormphrax/custom.js&stormphraxWasm=/stormphrax/custom.wasm&stormphraxData=/stormphrax/custom.data'));
  assert.equal(custom.key, 'custom');
  assert.equal(custom.jsUrl, '/stormphrax/custom.js');
  assert.equal(custom.wasmUrl, '/stormphrax/custom.wasm');
  assert.equal(custom.dataUrl, '/stormphrax/custom.data');
  assert.equal(stormphraxVariantFromParams(new URLSearchParams('stormphraxJs=https://evil.example/x.js')).key, defaultStormphraxVariantKey());
});
