import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  VIRIDITHAS_RELAXED_SIMD_VARIANT,
  VIRIDITHAS_SIMD_VARIANT,
  checkViridithasVariantAsset,
  resolveDefaultViridithasVariantAssetFallback,
  viridithasVariantAssetStatus,
} from '../src/lc0/viridithasVariants.ts';

test('production probes deployed Viridithas relaxed asset', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push([String(url), init?.method]);
    return new Response(null, { status: 200 });
  };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '0x88.app' },
  });
  const variant = { ...VIRIDITHAS_RELAXED_SIMD_VARIANT };
  try {
    assert.equal(await checkViridithasVariantAsset(variant), 'ok');
    assert.equal(viridithasVariantAssetStatus(variant), 'ok');
    const r2Variant = { ...VIRIDITHAS_RELAXED_SIMD_VARIANT, wasmUrl: 'https://assets.0x88.app/viridithas/viridithas-relaxed-simd128.wasm' };
    assert.equal(await checkViridithasVariantAsset(r2Variant), 'ok');
    assert.deepEqual(calls, [
      ['/viridithas/viridithas-relaxed-simd128.wasm', 'HEAD'],
      ['https://assets.0x88.app/viridithas/viridithas-relaxed-simd128.wasm', 'HEAD'],
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

test('IPv6 loopback still probes local Viridithas generated assets', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response(null, { status: 404 }); };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '[::1]' },
  });
  const variant = { ...VIRIDITHAS_RELAXED_SIMD_VARIANT, wasmUrl: '/viridithas/local-relaxed.wasm' };
  try {
    assert.equal(await checkViridithasVariantAsset(variant), 'missing');
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

test('production still probes deployed Viridithas SIMD asset', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response(null, { status: 200 }); };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '0x88.app' },
  });
  const variant = { ...VIRIDITHAS_SIMD_VARIANT };
  try {
    assert.equal(await checkViridithasVariantAsset(variant), 'ok');
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});

test('Viridithas default fallback keeps deployed relaxed asset', async () => {
  const originalFetch = globalThis.fetch;
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response(null, { status: 200 }); };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { hostname: '0x88.app' },
  });
  const relaxed = { ...VIRIDITHAS_RELAXED_SIMD_VARIANT };
  try {
    const resolved = await resolveDefaultViridithasVariantAssetFallback(relaxed, false);
    assert.equal(resolved.key, 'relaxed-simd');
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
    else delete globalThis.location;
  }
});
