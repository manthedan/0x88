import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { isTrustedExecutableAssetUrl, resolveAssetOverride, resolvePublicAssetUrl } from '../src/lc0/assetUrls.ts';

const MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';

function at(href, { configuredBase } = {}) {
  globalThis.location = new URL(href);
  if (configuredBase === undefined) delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
  else globalThis.LC0_BROWSER_ASSET_BASE_URL = configuredBase;
}

afterEach(() => {
  delete globalThis.location;
  delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
});

test('same-origin overrides preserve their origin when paths start with URL separators', () => {
  at('https://0x88.app/app/play/');
  for (const path of [
    '//evil.example/viridithas/viridithas.wasm',
    '/\\evil.example/viridithas/viridithas.wasm',
    '\\\\evil.example/viridithas/viridithas.wasm',
  ]) {
    const resolved = resolveAssetOverride(`https://0x88.app${path}`);
    assert.ok(resolved);
    assert.equal(new URL(resolved, location.href).origin, location.origin);
    assert.equal(isTrustedExecutableAssetUrl(resolved), true);
  }
});

test('the configured asset host resolves asset-host prefixes', () => {
  at('https://0x88.app/app/play/', { configuredBase: 'https://assets.0x88.app' });
  assert.equal(resolvePublicAssetUrl(MODEL), `https://assets.0x88.app${MODEL}`);
  assert.equal(resolvePublicAssetUrl('/stockfish/stockfish-19-lite.wasm'), 'https://assets.0x88.app/stockfish/stockfish-19-lite.wasm');
  assert.equal(resolvePublicAssetUrl('/openings/eco.json'), '/openings/eco.json');
});

test('a deployed page ignores ?assetBase= and ?modelBase= that name another origin', () => {
  for (const query of [
    '?assetBase=https://evil.example',
    '?modelBase=https://evil.example',
    '?assetBase=https://evil.example&modelBase=https://evil.example',
  ]) {
    at(`https://0x88.app/app/play/${query}`, { configuredBase: 'https://assets.0x88.app' });
    assert.equal(resolvePublicAssetUrl(MODEL), `https://assets.0x88.app${MODEL}`, query);
    assert.equal(resolvePublicAssetUrl('/reckless/reckless-simd128.wasm'), 'https://assets.0x88.app/reckless/reckless-simd128.wasm', query);
  }
});

test('a deployed page with no configured asset host stays same-origin under an override', () => {
  at('https://0x88.app/single-engine/?assetBase=https://evil.example');
  assert.equal(resolvePublicAssetUrl(MODEL), MODEL);
});

test('a deployed page honours an override naming the asset host or its own origin', () => {
  at('https://0x88.app/single-engine/?assetBase=https://assets.0x88.app/', { configuredBase: 'https://assets.0x88.app' });
  assert.equal(resolvePublicAssetUrl(MODEL), `https://assets.0x88.app${MODEL}`);
  at('https://0x88.app/single-engine/?assetBase=https://0x88.app', { configuredBase: 'https://assets.0x88.app' });
  assert.equal(resolvePublicAssetUrl(MODEL), `https://0x88.app${MODEL}`);
});

// docs/artifact_hosting_cache_strategy.md and local benchmark flows point a local page at
// another asset origin; that keeps working.
test('local pages honour any override', () => {
  at('http://127.0.0.1:5173/app/play/?assetBase=https://models.example.com', { configuredBase: 'https://assets.0x88.app' });
  assert.equal(resolvePublicAssetUrl(MODEL), `https://models.example.com${MODEL}`);
  at('http://localhost:5181/single-engine/?modelBase=http://127.0.0.1:8787/');
  assert.equal(resolvePublicAssetUrl(MODEL), `http://127.0.0.1:8787${MODEL}`);
  at('http://app.localhost:5181/single-engine/?assetBase=https://models.example.com');
  assert.equal(resolvePublicAssetUrl('/berserk/berserk-emscripten.wasm'), 'https://models.example.com/berserk/berserk-emscripten.wasm');
});

test('modelBase applies to model paths only', () => {
  at('http://localhost:5181/app/play/?modelBase=https://models.example.com');
  assert.equal(resolvePublicAssetUrl(MODEL), `https://models.example.com${MODEL}`);
  assert.equal(resolvePublicAssetUrl('/stockfish/stockfish-19-lite.wasm'), '/stockfish/stockfish-19-lite.wasm');
});

test('executable asset trust is limited to this origin and the configured asset host', () => {
  at('https://0x88.app/app/play/?assetBase=https://evil.example', { configuredBase: 'https://assets.0x88.app' });
  assert.equal(isTrustedExecutableAssetUrl('/stockfish/stockfish-19-lite.js'), true);
  assert.equal(isTrustedExecutableAssetUrl('https://assets.0x88.app/berserk/berserk-emscripten.js'), true);
  assert.equal(isTrustedExecutableAssetUrl('https://evil.example/berserk/berserk-emscripten.js'), false);
});

test('shared origin/path helpers handle loopback, blob workers and malformed paths', async () => {
  const { assetPathname, isLocalDevelopmentOrigin } = await import('../src/lc0/assetUrls.ts');
  for (const url of ['http://localhost:5181', 'http://dev.localhost:5181', 'http://127.0.0.1', 'http://[::1]']) {
    at(url);
    assert.equal(isLocalDevelopmentOrigin(), true, url);
  }
  globalThis.location = { href: 'blob:http://localhost:5181/worker', origin: 'http://localhost:5181', hostname: '' };
  assert.equal(isLocalDevelopmentOrigin(), true);
  at('https://0x88.app/app/play/');
  assert.equal(isLocalDevelopmentOrigin(), false);
  assert.equal(assetPathname('https://assets.0x88.app/reckless/engine.wasm?q=1'), '/reckless/engine.wasm');
  assert.equal(assetPathname('http://['), 'http://[');
});
