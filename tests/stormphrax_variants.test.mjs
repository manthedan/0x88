import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  STORMPHRAX_EMSCRIPTEN_VARIANT,
  STORMPHRAX_MAIN_NETWORK,
  defaultStormphraxVariantKey,
  hasExplicitStormphraxVariant,
  normalizeStormphraxVariant,
  stormphraxVariantByKey,
  stormphraxVariantFromParams,
} from '../src/lc0/stormphraxVariants.ts';
import { stormphraxSearchTimeoutMs } from '../src/lc0/stormphraxEngine.ts';

test('Stormphrax variant metadata pins the browser sidecars and undertown network', () => {
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.key, 'emscripten');
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.jsUrl, '/stormphrax/stormphrax-emscripten.js');
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.wasmUrl, '/stormphrax/stormphrax-emscripten.wasm');
  assert.equal(STORMPHRAX_EMSCRIPTEN_VARIANT.dataUrl, '/stormphrax/stormphrax-emscripten.data');
  assert.equal(STORMPHRAX_MAIN_NETWORK, 'undertown.nnue');
  assert.equal(defaultStormphraxVariantKey(), 'emscripten');
});

test('Stormphrax search timeout leaves startup headroom above movetime', () => {
  assert.equal(stormphraxSearchTimeoutMs({ movetimeMs: 60_000 }), 75_000);
  assert.equal(stormphraxSearchTimeoutMs({ movetimeMs: 1_000 }), 60_000);
  assert.equal(stormphraxSearchTimeoutMs({ depth: 12 }), 120_000);
});

test('Stormphrax variant normalization and same-origin overrides are stable', () => {
  assert.equal(normalizeStormphraxVariant('browser worker'), 'emscripten');
  assert.equal(normalizeStormphraxVariant('custom'), 'custom');
  assert.equal(stormphraxVariantByKey('unknown').label, 'Stormphrax 8');
  assert.equal(hasExplicitStormphraxVariant(new URLSearchParams('')), false);
  assert.equal(hasExplicitStormphraxVariant(new URLSearchParams('stormphrax=emscripten')), true);
  const custom = stormphraxVariantFromParams(new URLSearchParams('stormphraxJs=/stormphrax/custom.js&stormphraxWasm=/stormphrax/custom.wasm&stormphraxData=/stormphrax/custom.data'));
  assert.equal(custom.key, 'custom');
  assert.equal(custom.jsUrl, '/stormphrax/custom.js');
  assert.equal(custom.wasmUrl, '/stormphrax/custom.wasm');
  assert.equal(custom.dataUrl, '/stormphrax/custom.data');
  assert.equal(stormphraxVariantFromParams(new URLSearchParams('stormphraxJs=https://evil.example/x.js')).key, 'emscripten');
});
