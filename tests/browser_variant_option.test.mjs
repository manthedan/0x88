import assert from 'node:assert/strict';
import test from 'node:test';
import { browserVariantOption, createBrowserVariantSelector } from '../src/lc0/browserVariantOption.ts';

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

test('browser variant selectors include custom requests without duplicating built-ins', () => {
  let requested = { key: 'custom', label: 'Custom' };
  const selector = createBrowserVariantSelector({
    builtIns: [
      { key: 'base', label: 'Base' },
      { key: 'simd', label: 'SIMD' },
    ],
    requested: () => requested,
    normalize: (value) => value ?? 'base',
    byKey: (key) => ({ key, label: key }),
  });

  assert.deepEqual(
    selector.available().map((variant) => variant.key),
    ['base', 'simd', 'custom'],
  );
  assert.equal(selector.resolve('custom'), requested);

  requested = { key: 'simd', label: 'SIMD override' };
  assert.deepEqual(
    selector.available().map((variant) => variant.key),
    ['base', 'simd'],
  );
});

test('browser variant selectors reject unusable custom variants and fall back to a usable built-in', () => {
  const selector = createBrowserVariantSelector({
    builtIns: [
      { key: 'disabled', usable: false },
      { key: 'base', usable: true },
    ],
    requested: () => ({ key: 'custom', usable: false }),
    normalize: (value) => value ?? 'disabled',
    byKey: (key) => ({ key, usable: false }),
    usable: (variant) => variant.usable,
  });

  assert.deepEqual(selector.available(), [{ key: 'base', usable: true }]);
  assert.deepEqual(selector.resolve('disabled'), { key: 'base', usable: true });
});
