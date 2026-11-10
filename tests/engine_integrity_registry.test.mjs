import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
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

for (const profile of ['default', 'v0']) {
  test(`all variant and adapter selections have generated pins in the ${profile} deployment profile`, async (t) => {
    const previousLocation = globalThis.location;
    const previousBase = globalThis.LC0_BROWSER_ASSET_BASE_URL;
    globalThis.location = new URL('https://0x88.app/app/play/');
    delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
    t.after(() => {
      globalThis.location = previousLocation;
      globalThis.LC0_BROWSER_ASSET_BASE_URL = previousBase;
    });
    const modulePath = (name) => JSON.stringify(fileURLToPath(new URL(`../src/lc0/${name}.ts`, import.meta.url)));
    const entry = `
      import { BERSERK_VARIANTS } from ${modulePath('berserkVariants')};
      import { PLENTYCHESS_VARIANTS } from ${modulePath('plentychessVariants')};
      import { RECKLESS_VARIANTS } from ${modulePath('recklessVariants')};
      import { STORMPHRAX_VARIANTS } from ${modulePath('stormphraxVariants')};
      import { VIRIDITHAS_VARIANTS } from ${modulePath('viridithasVariants')};
      import * as sf from ${modulePath('stockfishEngine')};
      import { DEFAULT_RECKLESS_WASM_URL } from ${modulePath('recklessEngine')};
      import { DEFAULT_VIRIDITHAS_WASM_URL } from ${modulePath('viridithasEngine')};
      import { DEFAULT_BERSERK_EMSCRIPTEN_JS_URL } from ${modulePath('berserkEngine')};
      import { DEFAULT_PLENTYCHESS_EMSCRIPTEN_JS_URL } from ${modulePath('plentychessEngine')};
      import { DEFAULT_STORMPHRAX_EMSCRIPTEN_JS_URL } from ${modulePath('stormphraxEngine')};
      const variants = [...BERSERK_VARIANTS.filter(v => v.jsUrl), ...PLENTYCHESS_VARIANTS,
        ...RECKLESS_VARIANTS.filter(v => v.key !== 'lite'), ...STORMPHRAX_VARIANTS, ...VIRIDITHAS_VARIANTS];
      const glue = [DEFAULT_BERSERK_EMSCRIPTEN_JS_URL, DEFAULT_PLENTYCHESS_EMSCRIPTEN_JS_URL, DEFAULT_STORMPHRAX_EMSCRIPTEN_JS_URL];
      const stockfish = [sf.DEFAULT_STOCKFISH_URL, sf.STOCKFISH_LITE_SINGLE_URL, sf.STOCKFISH_LITE_SINGLE_RELAXED_URL,
        ...['lite-single','single','lite-threaded','threaded'].map(sf.stockfishFlavorUrl)];
      export const selections = [...variants.flatMap(v => ['jsUrl','wasmUrl','dataUrl','nnueUrl'].map(k => v[k]).filter(Boolean)),
        DEFAULT_RECKLESS_WASM_URL, DEFAULT_VIRIDITHAS_WASM_URL,
        ...glue.flatMap(u => [u, u.replace(/\\.js$/, '.wasm'), u.replace(/\\.js$/, '.data')]),
        ...stockfish.flatMap(u => [u, u.replace(/\\.js$/, '.wasm')])];
    `;
    const result = await build({
      configFile: false,
      envDir: false,
      publicDir: false,
      logLevel: 'silent',
      define: {
        'import.meta.env': JSON.stringify({
          DEV: false,
          VITE_BROWSER_CHESS_DEPLOY_PROFILE: profile,
          VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://assets.0x88.app',
        }),
      },
      plugins: [
        {
          name: 'integrity-profile',
          resolveId: (id) => (id === 'integrity-profile' ? '\0integrity-profile' : undefined),
          load: (id) => (id === '\0integrity-profile' ? entry : undefined),
        },
      ],
      build: { write: false, minify: false, rollupOptions: { input: 'integrity-profile', preserveEntrySignatures: 'strict' } },
    });
    const output = result.output.find((item) => item.type === 'chunk' && item.isEntry);
    const { selections } = await import(`data:text/javascript;base64,${Buffer.from(output.code).toString('base64')}`);
    const assertPinned = (url) => {
      const path = new URL(url, location.href).pathname;
      const match = path.match(/^\/artifacts\/sha256\/([a-f0-9]{64})\/(identity|br\/([a-f0-9]{64}))$/);
      const pin = match ? Object.values(ENGINE_ARTIFACT_PINS).find((pin) => pin.sha256 === match[1]) : ENGINE_ARTIFACT_PINS[path];
      assert.ok(pin, `Unpinned ${profile} selection: ${url}`);
      if (match?.[3]) assert.equal(match[3], pin.br, `Unpinned Brotli representation: ${url}`);
    };
    assert.ok(selections.length > 50);
    for (const url of selections) assertPinned(url);
    assert.throws(() => assertPinned('/stockfish/unregistered.wasm'), /Unpinned/);
    assert.throws(() => assertPinned(`https://assets.0x88.app/artifacts/sha256/${'0'.repeat(64)}/identity`), /Unpinned/);
  });
}
