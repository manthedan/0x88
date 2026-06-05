import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  BERSERK_DEFAULT_NNUE_URL,
  BERSERK_DEFAULT_VARIANT,
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

test('Berserk variants pin planned URLs and network metadata', () => {
  assert.equal(BERSERK_MAIN_NETWORK, 'berserk-9b84c340af7e.nn');
  assert.equal(BERSERK_DEFAULT_NNUE_URL, '/berserk/berserk-9b84c340af7e.nn');
  assert.equal(BERSERK_SOURCE_NETWORK_URL, 'https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn');
  assert.deepEqual(BERSERK_VARIANTS.map((variant) => variant.key), ['default', 'simd']);
  assert.equal(BERSERK_DEFAULT_VARIANT.wasmUrl, '/berserk/berserk.wasm');
  assert.equal(BERSERK_SIMD_VARIANT.wasmUrl, '/berserk/berserk-simd128.wasm');
  assert.equal(BERSERK_SIMD_VARIANT.nnueUrl, BERSERK_DEFAULT_NNUE_URL);
});

test('Berserk variant normalization and lookup are stable', () => {
  assert.equal(normalizeBerserkVariant('simd128'), 'simd');
  assert.equal(normalizeBerserkVariant('wasm simd'), 'simd');
  assert.equal(normalizeBerserkVariant('scalar'), 'default');
  assert.equal(normalizeBerserkVariant('full'), 'default');
  assert.equal(normalizeBerserkVariant('custom'), 'custom');
  assert.equal(normalizeBerserkVariant('unknown'), 'default');
  assert.equal(berserkVariantByKey('simd').label, 'Berserk SIMD experimental');
  assert.equal(berserkVariantByKey('custom').key, 'custom');
});

test('Berserk URL params support explicit and custom variants', () => {
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('')), false);
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('berserk=simd')), true);
  assert.equal(hasExplicitBerserkVariant(new URLSearchParams('berserkNnue=/tmp/net.nn')), false);
  assert.equal(berserkVariantFromParams(new URLSearchParams('berserk=simd')).key, 'simd');
  const builtInWithCustomNnue = berserkVariantFromParams(new URLSearchParams('berserk=simd&berserkNnue=/local/net.nn'));
  assert.equal(builtInWithCustomNnue.key, 'simd');
  assert.equal(builtInWithCustomNnue.wasmUrl, '/berserk/berserk-simd128.wasm');
  assert.equal(builtInWithCustomNnue.nnueUrl, '/local/net.nn');
  const defaultWithCustomNnue = berserkVariantFromParams(new URLSearchParams('berserkNnue=/local/default-net.nn'));
  assert.equal(defaultWithCustomNnue.nnueUrl, '/local/default-net.nn');
  const custom = berserkVariantFromParams(new URLSearchParams('berserkWasm=/local/berserk.wasm&berserkNnue=/local/net.nn'));
  assert.equal(custom.key, 'custom');
  assert.equal(custom.wasmUrl, '/local/berserk.wasm');
  assert.equal(custom.nnueUrl, '/local/net.nn');
});

test('Berserk asset checks require wasm and NNUE assets', async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push([String(url), init?.method, init?.cache]);
    return { ok: !String(url).includes('missing') };
  };
  try {
    const present = { ...BERSERK_DEFAULT_VARIANT, wasmUrl: '/ok/berserk.wasm', nnueUrl: '/ok/net.nn' };
    assert.equal(berserkVariantAssetStatus(present), 'unknown');
    assert.equal(await checkBerserkVariantAsset(present), 'present');
    assert.equal(berserkVariantAssetStatus(present), 'present');
    assert.deepEqual(calls.slice(0, 2), [['/ok/berserk.wasm', 'HEAD', 'no-store'], ['/ok/net.nn', 'HEAD', 'no-store']]);

    const missing = { ...BERSERK_SIMD_VARIANT, wasmUrl: '/missing/berserk.wasm' };
    assert.equal(await checkBerserkVariantAsset(missing), 'missing');
    assert.equal(await resolveDefaultBerserkVariantAssetFallback(missing, false), BERSERK_DEFAULT_VARIANT);
    assert.equal(await resolveDefaultBerserkVariantAssetFallback(missing, true), missing);

    const missingWithCustomNnue = { ...BERSERK_SIMD_VARIANT, wasmUrl: '/missing-custom/berserk.wasm', nnueUrl: '/custom/net.nn' };
    const fallback = await resolveDefaultBerserkVariantAssetFallback(missingWithCustomNnue, false);
    assert.equal(fallback.key, 'default');
    assert.equal(fallback.wasmUrl, '/berserk/berserk.wasm');
    assert.equal(fallback.nnueUrl, '/custom/net.nn');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
