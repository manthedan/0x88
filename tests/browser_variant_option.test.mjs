import assert from 'node:assert/strict';
import test from 'node:test';
import { browserVariantOption } from '../src/lc0/browserVariantOption.ts';

test('browserVariantOption normalizes family asset statuses', () => {
  assert.deepEqual(browserVariantOption('simd', 'SIMD', { assetStatus: 'ok' }), {
    value: 'simd',
    label: 'SIMD',
    disabled: false,
  });
  assert.deepEqual(browserVariantOption('simd', 'SIMD', { assetStatus: 'missing' }), {
    value: 'simd',
    label: 'SIMD (asset missing)',
    disabled: true,
  });
});

test('browserVariantOption distinguishes optional probes from required generated assets', () => {
  assert.equal(browserVariantOption('base', 'Base', { assetStatus: 'unknown' }).disabled, false);
  assert.deepEqual(browserVariantOption('generated', 'Generated', { assetStatus: 'checking', requirePresent: true }), {
    value: 'generated',
    label: 'Generated (checking asset)',
    disabled: true,
  });
});

test('browserVariantOption gives unsupported reasons priority', () => {
  assert.deepEqual(
    browserVariantOption('relaxed', 'Relaxed', {
      assetStatus: 'missing',
      unsupportedReason: 'requires WebAssembly Relaxed SIMD',
      requirePresent: true,
    }),
    {
      value: 'relaxed',
      label: 'Relaxed (requires WebAssembly Relaxed SIMD)',
      disabled: true,
    },
  );
});
