import assert from 'node:assert/strict';
import test from 'node:test';
import { isTrustedExecutableAssetUrl, resolveAssetOverride } from '../src/lc0/assetUrls.ts';
import { berserkVariantFromParams } from '../src/lc0/berserkVariants.ts';
import { plentyChessVariantFromParams } from '../src/lc0/plentychessVariants.ts';
import { recklessVariantFromParams } from '../src/lc0/recklessVariants.ts';
import { stormphraxVariantFromParams } from '../src/lc0/stormphraxVariants.ts';
import { viridithasVariantFromParams } from '../src/lc0/viridithasVariants.ts';

test('asset overrides share base-override trust boundaries and reject non-HTTP schemes', (t) => {
  t.after(() => {
    delete globalThis.location;
    delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
  });
  globalThis.location = new URL('https://0x88.app/app/play/');
  globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
  assert.equal(resolveAssetOverride('/reckless/custom.wasm'), '/reckless/custom.wasm');
  assert.equal(resolveAssetOverride('https://assets.0x88.app/reckless/custom.wasm'), 'https://assets.0x88.app/reckless/custom.wasm');
  for (const url of [
    'https://evil.example/a.wasm',
    '//evil.example/a.wasm',
    'data:application/wasm;base64,AAAA',
    'javascript:alert(1)',
    'blob:https://0x88.app/abc',
  ])
    assert.equal(resolveAssetOverride(url), undefined);
  globalThis.location = new URL('http://localhost:5181/lab/');
  assert.equal(resolveAssetOverride('https://lab.example/a.wasm'), 'https://lab.example/a.wasm');
  assert.equal(isTrustedExecutableAssetUrl('https://lab.example/a.js'), true);
  assert.equal(resolveAssetOverride('data:text/javascript,alert(1)'), undefined);
});
for (const [name, fromParams, param, field] of [
  ['Reckless', recklessVariantFromParams, 'recklessWasm', 'wasmUrl'],
  ['Viridithas', viridithasVariantFromParams, 'viridithasWasm', 'wasmUrl'],
  ['Berserk', berserkVariantFromParams, 'berserkJs', 'jsUrl'],
  ['PlentyChess', plentyChessVariantFromParams, 'plentyChessWasm', 'wasmUrl'],
  ['Stormphrax', stormphraxVariantFromParams, 'stormphraxData', 'dataUrl'],
]) {
  test(`${name} rejects arbitrary production overrides and allows trusted hosts and local experiments`, (t) => {
    t.after(() => {
      delete globalThis.location;
      delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
    });
    globalThis.location = new URL('https://0x88.app/app/play/');
    globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
    const params = (url) => new URLSearchParams({ [param]: url });
    assert.notEqual(fromParams(params('https://evil.example/engine.wasm')).key, 'custom');
    assert.equal(fromParams(params('https://assets.0x88.app/engine.wasm'))[field], 'https://assets.0x88.app/engine.wasm');
    globalThis.location = new URL('http://[::1]:5181');
    assert.equal(fromParams(params('https://lab.example/engine.wasm'))[field], 'https://lab.example/engine.wasm');
  });
}
