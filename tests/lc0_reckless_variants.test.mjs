import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  RECKLESS_FULL_VARIANT,
  RECKLESS_SIMD_VARIANT,
  defaultRecklessVariantKey,
  normalizeRecklessVariant,
  recklessVariantFromParams,
  resolveDefaultRecklessVariantAssetFallback,
  supportsWasmSimd,
} from '../src/lc0/recklessVariants.ts';

test('Reckless SIMD is the default when the browser supports WebAssembly SIMD', () => {
  const expected = supportsWasmSimd() ? 'simd' : 'full';
  assert.equal(defaultRecklessVariantKey(), expected);
  assert.equal(recklessVariantFromParams(new URLSearchParams()).key, expected);
});

test('Reckless variant aliases keep full SIMD explicit and scalar as fallback', () => {
  assert.equal(normalizeRecklessVariant('simd'), 'simd');
  assert.equal(normalizeRecklessVariant('simd128'), 'simd');
  assert.equal(normalizeRecklessVariant('full-simd'), 'simd');
  assert.equal(normalizeRecklessVariant('full'), 'full');
  assert.equal(recklessVariantFromParams(new URLSearchParams('recklessVariant=simd')).key, 'simd');
  assert.equal(recklessVariantFromParams(new URLSearchParams('recklessVariant=full')).key, 'full');
});

test('implicit SIMD default falls back to scalar when the SIMD asset is unavailable', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false });
  try {
    const resolved = await resolveDefaultRecklessVariantAssetFallback(RECKLESS_SIMD_VARIANT, false);
    assert.equal(resolved, RECKLESS_FULL_VARIANT);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('explicit SIMD does not silently downgrade during asset preflight', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false });
  try {
    const resolved = await resolveDefaultRecklessVariantAssetFallback(RECKLESS_SIMD_VARIANT, true);
    assert.equal(resolved, RECKLESS_SIMD_VARIANT);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
