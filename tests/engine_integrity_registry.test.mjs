import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { enginePins } from '../scripts/write_engine_integrity.mjs';
import { BERSERK_VARIANTS } from '../src/lc0/berserkVariants.ts';
import { ENGINE_ARTIFACT_PINS } from '../src/lc0/engineArtifactPins.generated.ts';
import { engineArtifacts, resolveEngineArtifactUrl } from '../src/lc0/engineArtifacts.ts';
import { PLENTYCHESS_VARIANTS } from '../src/lc0/plentychessVariants.ts';
import { RECKLESS_VARIANTS } from '../src/lc0/recklessVariants.ts';
import { STORMPHRAX_VARIANTS } from '../src/lc0/stormphraxVariants.ts';
import { VIRIDITHAS_VARIANTS } from '../src/lc0/viridithasVariants.ts';

test('every published UCI variant has manifest pins for all executable/network sidecars', () => {
  for (const variant of [
    ...BERSERK_VARIANTS.filter((variant) => variant.jsUrl),
    ...PLENTYCHESS_VARIANTS,
    ...RECKLESS_VARIANTS,
    ...STORMPHRAX_VARIANTS,
    ...VIRIDITHAS_VARIANTS,
  ]) {
    if (variant.key === 'lite') continue; // unpromoted locally built Reckless lab experiment
    for (const field of ['jsUrl', 'wasmUrl', 'dataUrl', 'nnueUrl']) {
      if (variant[field]) assert.ok(engineArtifacts.pin(variant[field]), `${variant.key} ${field}`);
    }
  }
  assert.throws(() => resolveEngineArtifactUrl('/engine/missing.wasm'), /Missing engine artifact pin/);
});
test('hosted variants use manifest-derived v2 Brotli pins and local dist uses file paths', (t) => {
  t.after(() => {
    delete globalThis.location;
    delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
  });
  globalThis.location = new URL('https://0x88.app/app/play/');
  globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
  for (const path of [
    '/stormphrax/stormphrax-emscripten.wasm',
    '/stormphrax/stormphrax-emscripten-relaxed-simd128.wasm',
    '/stormphrax/stormphrax-emscripten.data',
  ]) {
    const pin = ENGINE_ARTIFACT_PINS[path];
    assert.equal(resolveEngineArtifactUrl(path), `https://assets.0x88.app/artifacts/sha256/${pin.sha256}/br/${pin.br}`);
  }
  delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
  globalThis.location = new URL('http://localhost:5181');
  assert.equal(resolveEngineArtifactUrl('/stormphrax/stormphrax-emscripten.wasm'), '/stormphrax/stormphrax-emscripten.wasm');
});
test('generation fails if a release representation no longer matches its manifest or a pin disappears', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'engine-pins-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const engine of ['berserk', 'plentychess', 'reckless', 'stockfish', 'stormphrax', 'viridithas']) {
    const index = JSON.parse(readFileSync('public/artifact-index.json', 'utf8'));
    const manifest = index.entries.find((entry) => entry.family === engine).manifestUrl;
    cpSync(`public${manifest}`, join(root, `public${manifest}`));
  }
  cpSync('scripts/engine_artifact_representations.json', join(root, 'scripts/engine_artifact_representations.json'));
  const path = join(root, 'public/stormphrax/stormphrax-emscripten-single-thread.manifest.json');
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  const original = manifest.artifacts[0].sha256;
  manifest.artifacts[0].sha256 = '0'.repeat(64);
  writeFileSync(path, JSON.stringify(manifest));
  assert.throws(() => enginePins(root), /Release\/engine manifest mismatch/);
  manifest.artifacts[0].sha256 = original;
  manifest.artifacts.shift();
  writeFileSync(path, JSON.stringify(manifest));
  assert.throws(() => enginePins(root), /no manifest pin/);
});

test('loopback asset bases retain their configured host and port', (t) => {
  const previousLocation = globalThis.location;
  const previousBase = globalThis.LC0_BROWSER_ASSET_BASE_URL;
  t.after(() => {
    globalThis.location = previousLocation;
    globalThis.LC0_BROWSER_ASSET_BASE_URL = previousBase;
  });
  globalThis.location = new URL('http://localhost:5173/app/play/');
  for (const base of ['http://127.0.0.1:8787', 'http://assets.localhost:8787', 'http://[::1]:8787']) {
    globalThis.LC0_BROWSER_ASSET_BASE_URL = base;
    assert.equal(resolveEngineArtifactUrl('/berserk/berserk-emscripten.wasm'), `${base}/berserk/berserk-emscripten.wasm`);
  }
});
