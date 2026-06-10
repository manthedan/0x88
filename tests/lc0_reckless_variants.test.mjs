import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  RECKLESS_FULL_VARIANT,
  RECKLESS_RELAXED_SIMD_VARIANT,
  RECKLESS_SIMD_VARIANT,
  RECKLESS_VARIANTS,
  defaultRecklessVariantKey,
  normalizeRecklessVariant,
  recklessVariantFromParams,
  resolveDefaultRecklessVariantAssetFallback,
  supportsWasmRelaxedSimd,
  supportsWasmSimd,
} from '../src/lc0/recklessVariants.ts';

test('Reckless default follows the relaxed > simd > scalar speed ladder', () => {
  const expected = supportsWasmRelaxedSimd() ? 'relaxed-simd' : supportsWasmSimd() ? 'simd' : 'full';
  assert.equal(defaultRecklessVariantKey(), expected);
  assert.equal(recklessVariantFromParams(new URLSearchParams()).key, expected);
});

test('Reckless variant aliases keep full SIMD explicit and scalar as fallback', () => {
  assert.equal(normalizeRecklessVariant('relaxed'), 'relaxed-simd');
  assert.equal(normalizeRecklessVariant('relaxed-simd128'), 'relaxed-simd');
  assert.equal(normalizeRecklessVariant('simd'), 'simd');
  assert.equal(normalizeRecklessVariant('simd128'), 'simd');
  assert.equal(normalizeRecklessVariant('full-simd'), 'simd');
  assert.equal(normalizeRecklessVariant('full'), 'full');
  assert.equal(recklessVariantFromParams(new URLSearchParams('recklessVariant=relaxed-simd')).key, 'relaxed-simd');
  assert.equal(recklessVariantFromParams(new URLSearchParams('recklessVariant=simd')).key, 'simd');
  assert.equal(recklessVariantFromParams(new URLSearchParams('recklessVariant=full')).key, 'full');
});

test('Reckless relaxed SIMD is the feature-detected default where supported', () => {
  assert.equal(typeof supportsWasmRelaxedSimd(), 'boolean');
  assert.equal(recklessVariantFromParams(new URLSearchParams('recklessVariant=relaxed-simd')).wasmUrl, RECKLESS_RELAXED_SIMD_VARIANT.wasmUrl);
  if (supportsWasmRelaxedSimd()) assert.equal(defaultRecklessVariantKey(), 'relaxed-simd');
  assert.equal(RECKLESS_VARIANTS.some((variant) => variant.key === 'relaxed-simd'), true);
});

test('explicit relaxed SIMD falls back when the runtime cannot validate relaxed SIMD', async () => {
  const originalValidate = WebAssembly.validate;
  WebAssembly.validate = (bytes) => (bytes.byteLength === 49 ? false : originalValidate.call(WebAssembly, bytes));
  try {
    const resolved = await resolveDefaultRecklessVariantAssetFallback(RECKLESS_RELAXED_SIMD_VARIANT, true);
    assert.equal(resolved.key, supportsWasmSimd() ? 'simd' : 'full');
  } finally {
    WebAssembly.validate = originalValidate;
  }
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

test('implicit relaxed SIMD default falls through to scalar when both SIMD assets are unavailable', async () => {
  const originalFetch = globalThis.fetch;
  const originalValidate = WebAssembly.validate;
  globalThis.fetch = async () => ({ ok: false });
  WebAssembly.validate = () => true;
  try {
    const resolved = await resolveDefaultRecklessVariantAssetFallback(RECKLESS_RELAXED_SIMD_VARIANT, false);
    assert.equal(resolved, RECKLESS_FULL_VARIANT);
  } finally {
    globalThis.fetch = originalFetch;
    WebAssembly.validate = originalValidate;
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
