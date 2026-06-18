import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  PLENTYCHESS_EMSCRIPTEN_DATA_URL,
  PLENTYCHESS_EMSCRIPTEN_JS_URL,
  PLENTYCHESS_EMSCRIPTEN_VARIANT,
  PLENTYCHESS_EMSCRIPTEN_WASM_URL,
  PLENTYCHESS_MAIN_NETWORK,
  PLENTYCHESS_SOURCE_NETWORK_URL,
  PLENTYCHESS_VARIANTS,
  checkPlentyChessVariantAsset,
  defaultPlentyChessVariantKey,
  hasExplicitPlentyChessVariant,
  supportsWasmRelaxedSimd,
  supportsWasmSimd,
  normalizePlentyChessVariant,
  plentyChessVariantAssetStatus,
  plentyChessVariantByKey,
  plentyChessVariantFromParams,
  plentyChessVariantUnsupportedReason,
} from '../src/lc0/plentychessVariants.ts';

test('PlentyChess variant metadata pins the smoked Emscripten sidecars', () => {
  assert.equal(PLENTYCHESS_MAIN_NETWORK, '0134-2r24-s0.bin');
  assert.equal(PLENTYCHESS_SOURCE_NETWORK_URL, 'https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin');
  assert.deepEqual(PLENTYCHESS_VARIANTS.map((variant) => variant.key), ['emscripten', 'emscripten-sse41', 'emscripten-relaxed']);
  // Default follows the relaxed > sse41 ladder after explicitly validating
  // baseline wasm SIMD support; without SIMD, the base option stays selected
  // but disabled by the browser UI.
  assert.equal(defaultPlentyChessVariantKey(), !supportsWasmSimd() ? 'emscripten' : supportsWasmRelaxedSimd() ? 'emscripten-relaxed' : 'emscripten-sse41');
  assert.equal(PLENTYCHESS_EMSCRIPTEN_VARIANT.jsUrl, PLENTYCHESS_EMSCRIPTEN_JS_URL);
  assert.equal(PLENTYCHESS_EMSCRIPTEN_VARIANT.wasmUrl, PLENTYCHESS_EMSCRIPTEN_WASM_URL);
  assert.equal(PLENTYCHESS_EMSCRIPTEN_VARIANT.dataUrl, PLENTYCHESS_EMSCRIPTEN_DATA_URL);
});

test('PlentyChess variant normalization and lookup are stable', () => {
  assert.equal(normalizePlentyChessVariant('emscripten'), 'emscripten');
  assert.equal(normalizePlentyChessVariant('browser worker'), 'emscripten');
  assert.equal(normalizePlentyChessVariant('custom'), 'custom');
  assert.equal(normalizePlentyChessVariant('unknown'), 'emscripten');
  assert.equal(plentyChessVariantByKey('emscripten').label, 'PlentyChess Emscripten experimental');
  assert.equal(plentyChessVariantByKey('custom').key, 'custom');
  assert.equal(plentyChessVariantUnsupportedReason(PLENTYCHESS_EMSCRIPTEN_VARIANT), supportsWasmSimd() ? null : 'requires WebAssembly SIMD');
});

test('PlentyChess URL params support explicit and custom Emscripten sidecars', () => {
  assert.equal(hasExplicitPlentyChessVariant(new URLSearchParams('')), false);
  assert.equal(hasExplicitPlentyChessVariant(new URLSearchParams('plentychess=emscripten')), true);
  assert.equal(hasExplicitPlentyChessVariant(new URLSearchParams('plentyChessJs=/plentychess/custom.js')), true);
  assert.equal(plentyChessVariantFromParams(new URLSearchParams('')).key, defaultPlentyChessVariantKey());
  assert.equal(plentyChessVariantFromParams(new URLSearchParams('plentychess=custom')).key, 'custom');
  const custom = plentyChessVariantFromParams(new URLSearchParams('plentyChessJs=/plentychess/custom.js&plentyChessWasm=/plentychess/custom.wasm&plentyChessData=/plentychess/custom.data'));
  assert.equal(custom.key, 'custom');
  assert.equal(custom.jsUrl, '/plentychess/custom.js');
  assert.equal(custom.wasmUrl, '/plentychess/custom.wasm');
  assert.equal(custom.dataUrl, '/plentychess/custom.data');
  const rejectedCustom = plentyChessVariantFromParams(new URLSearchParams('plentyChessJs=https://evil.example/plenty.js&plentyChessWasm=/local/plenty.wasm'));
  assert.equal(rejectedCustom.key, defaultPlentyChessVariantKey());
});

test('PlentyChess asset checks require JS, WASM, and data sidecars', async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push([String(url), init?.method, init?.cache]);
    return { ok: !String(url).includes('missing') };
  };
  try {
    const present = { ...PLENTYCHESS_EMSCRIPTEN_VARIANT, jsUrl: '/ok/plenty.js', wasmUrl: '/ok/plenty.wasm', dataUrl: '/ok/plenty.data' };
    assert.equal(plentyChessVariantAssetStatus(present), 'unknown');
    assert.equal(await checkPlentyChessVariantAsset(present), 'present');
    assert.equal(plentyChessVariantAssetStatus(present), 'present');
    assert.deepEqual(calls.slice(0, 3), [['/ok/plenty.js', 'HEAD', 'no-store'], ['/ok/plenty.wasm', 'HEAD', 'no-store'], ['/ok/plenty.data', 'HEAD', 'no-store']]);

    const missing = { ...PLENTYCHESS_EMSCRIPTEN_VARIANT, jsUrl: '/missing/plenty.js', wasmUrl: '/ok/plenty.wasm', dataUrl: '/ok/plenty.data' };
    assert.equal(await checkPlentyChessVariantAsset(missing), 'missing');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
