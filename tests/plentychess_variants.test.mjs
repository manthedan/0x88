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
  normalizePlentyChessVariant,
  plentyChessVariantAssetStatus,
  plentyChessVariantByKey,
  plentyChessVariantFromParams,
} from '../src/lc0/plentychessVariants.ts';

test('PlentyChess variant metadata pins the smoked Emscripten sidecars', () => {
  assert.equal(PLENTYCHESS_MAIN_NETWORK, '0134-2r24-s0.bin');
  assert.equal(PLENTYCHESS_SOURCE_NETWORK_URL, 'https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin');
  assert.deepEqual(PLENTYCHESS_VARIANTS.map((variant) => variant.key), ['emscripten']);
  assert.equal(defaultPlentyChessVariantKey(), 'emscripten');
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
});

test('PlentyChess URL params support explicit and custom Emscripten sidecars', () => {
  assert.equal(hasExplicitPlentyChessVariant(new URLSearchParams('')), false);
  assert.equal(hasExplicitPlentyChessVariant(new URLSearchParams('plentychess=emscripten')), true);
  assert.equal(hasExplicitPlentyChessVariant(new URLSearchParams('plentyChessJs=/local/plenty.js')), true);
  assert.equal(plentyChessVariantFromParams(new URLSearchParams('')).key, 'emscripten');
  assert.equal(plentyChessVariantFromParams(new URLSearchParams('plentychess=custom')).key, 'custom');
  const custom = plentyChessVariantFromParams(new URLSearchParams('plentyChessJs=/local/plenty.js&plentyChessWasm=/local/plenty.wasm&plentyChessData=/local/plenty.data'));
  assert.equal(custom.key, 'custom');
  assert.equal(custom.jsUrl, '/local/plenty.js');
  assert.equal(custom.wasmUrl, '/local/plenty.wasm');
  assert.equal(custom.dataUrl, '/local/plenty.data');
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
