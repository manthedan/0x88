import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  BERSERK_DEFAULT_NNUE_URL,
  BERSERK_DEFAULT_VARIANT,
  BERSERK_EMSCRIPTEN_DATA_URL,
  BERSERK_EMSCRIPTEN_JS_URL,
  BERSERK_EMSCRIPTEN_VARIANT,
  BERSERK_EMSCRIPTEN_WASM_URL,
  BERSERK_MAIN_NETWORK,
  BERSERK_SIMD_VARIANT,
  BERSERK_SOURCE_NETWORK_URL,
  BERSERK_VARIANTS,
  berserkVariantAssetStatus,
  berserkVariantByKey,
  berserkVariantFromParams,
  checkBerserkVariantAsset,
  hasExplicitBerserkVariant,
  normalizeBerserkVariant,
  resolveDefaultBerserkVariantAssetFallback,
} from '../src/lc0/berserkVariants.ts';

test('Berserk variants pin Emscripten smoke and planned WASI metadata', () => {
  assert.equal(BERSERK_MAIN_NETWORK, 'berserk-9b84c340af7e.nn');
  assert.equal(BERSERK_DEFAULT_NNUE_URL, '/berserk/berserk-9b84c340af7e.nn');
  assert.equal(BERSERK_SOURCE_NETWORK_URL, 'https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn');
  assert.deepEqual(BERSERK_VARIANTS.map((variant) => variant.key), ['emscripten', 'default', 'simd']);
  assert.equal(BERSERK_EMSCRIPTEN_VARIANT.jsUrl, BERSERK_EMSCRIPTEN_JS_URL);
  assert.equal(BERSERK_EMSCRIPTEN_VARIANT.wasmUrl, BERSERK_EMSCRIPTEN_WASM_URL);
  assert.equal(BERSERK_EMSCRIPTEN_VARIANT.dataUrl, BERSERK_EMSCRIPTEN_DATA_URL);
  assert.equal(BERSERK_DEFAULT_VARIANT.wasmUrl, '/berserk/berserk.wasm');
  assert.equal(BERSERK_SIMD_VARIANT.wasmUrl, '/berserk/berserk-simd128.wasm');
  assert.equal(BERSERK_SIMD_VARIANT.nnueUrl, BERSERK_DEFAULT_NNUE_URL);
});

test('Berserk variant normalization and lookup are stable', () => {
  assert.equal(normalizeBerserkVariant('emscripten'), 'emscripten');
  assert.equal(normalizeBerserkVariant('browser worker'), 'emscripten');
  assert.equal(normalizeBerserkVariant('simd128'), 'simd');
  assert.equal(normalizeBerserkVariant('wasm simd'), 'simd');
  assert.equal(normalizeBerserkVariant('scalar'), 'default');
  assert.equal(normalizeBerserkVariant('full'), 'default');
  assert.equal(normalizeBerserkVariant('custom'), 'custom');
  assert.equal(normalizeBerserkVariant('unknown'), 'emscripten');
  assert.equal(berserkVariantByKey('emscripten').label, 'Berserk Emscripten experimental');
  assert.equal(berserkVariantByKey('simd').label, 'Berserk SIMD WASI planned');
  assert.equal(berserkVariantByKey('custom').key, 'custom');
});

test('Berserk URL params support explicit and custom variants', () => {
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('')), false);
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('berserk=simd')), true);
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('berserkJs=/berserk/custom.js')), true);
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('berserkNnue=/tmp/net.nn')), false);
  assert.equal(berserkVariantFromParams(new URLSearchParams('')).key, 'emscripten');
  assert.equal(berserkVariantFromParams(new URLSearchParams('berserk=simd')).key, 'simd');
  const builtInWithCustomNnue = berserkVariantFromParams(new URLSearchParams('berserk=simd&berserkNnue=/berserk/net.nn'));
  assert.equal(builtInWithCustomNnue.key, 'simd');
  assert.equal(builtInWithCustomNnue.wasmUrl, '/berserk/berserk-simd128.wasm');
  assert.equal(builtInWithCustomNnue.nnueUrl, '/berserk/net.nn');
  const defaultWithCustomNnue = berserkVariantFromParams(new URLSearchParams('berserk=default&berserkNnue=/berserk/default-net.nn'));
  assert.equal(defaultWithCustomNnue.nnueUrl, '/berserk/default-net.nn');
  const customJs = berserkVariantFromParams(new URLSearchParams('berserkJs=/berserk/custom.js&berserkWasm=/berserk/custom.wasm&berserkData=/berserk/custom.data'));
  assert.equal(customJs.key, 'custom');
  assert.equal(customJs.jsUrl, '/berserk/custom.js');
  assert.equal(customJs.wasmUrl, '/berserk/custom.wasm');
  assert.equal(customJs.dataUrl, '/berserk/custom.data');
  const customWasi = berserkVariantFromParams(new URLSearchParams('berserkWasm=/berserk/custom.wasm&berserkNnue=/berserk/net.nn'));
  assert.equal(customWasi.key, 'emscripten');
  const rejectedCustom = berserkVariantFromParams(new URLSearchParams('berserkJs=https://evil.example/berserk.js&berserkWasm=/local/berserk.wasm'));
  assert.equal(rejectedCustom.key, 'emscripten');
});

test('Berserk asset checks use Emscripten sidecars or WASI+NNUE assets', async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push([String(url), init?.method, init?.cache]);
    return { ok: !String(url).includes('missing') };
  };
  try {
    const present = { ...BERSERK_EMSCRIPTEN_VARIANT, jsUrl: '/ok/berserk.js', wasmUrl: '/ok/berserk.wasm', dataUrl: '/ok/berserk.data' };
    assert.equal(berserkVariantAssetStatus(present), 'unknown');
    assert.equal(await checkBerserkVariantAsset(present), 'present');
    assert.equal(berserkVariantAssetStatus(present), 'present');
    assert.deepEqual(calls.slice(0, 3), [['/ok/berserk.js', 'HEAD', 'no-store'], ['/ok/berserk.wasm', 'HEAD', 'no-store'], ['/ok/berserk.data', 'HEAD', 'no-store']]);

    const wasiPresent = { ...BERSERK_DEFAULT_VARIANT, wasmUrl: '/ok/berserk-wasi.wasm', nnueUrl: '/ok/net.nn' };
    assert.equal(await checkBerserkVariantAsset(wasiPresent), 'present');
    assert.deepEqual(calls.slice(3, 5), [['/ok/berserk-wasi.wasm', 'HEAD', 'no-store'], ['/ok/net.nn', 'HEAD', 'no-store']]);

    const missing = { ...BERSERK_SIMD_VARIANT, wasmUrl: '/missing/berserk.wasm' };
    assert.equal(await checkBerserkVariantAsset(missing), 'missing');
    assert.equal(await resolveDefaultBerserkVariantAssetFallback(missing, false), BERSERK_EMSCRIPTEN_VARIANT);
    assert.equal(await resolveDefaultBerserkVariantAssetFallback(missing, true), missing);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
