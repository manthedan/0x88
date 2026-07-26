import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { resolveEmscriptenAssetUrl } from '../src/lc0/emscriptenLocateFile.ts';

const JS = 'https://assets.example/plentychess/plentychess-emscripten-relaxed-simd128.js';
const WASM = 'https://assets.example/plentychess/plentychess-emscripten-relaxed-simd128.wasm';
const DATA = 'https://assets.example/plentychess/plentychess-emscripten.data';

// The regression this guards: only the canonical <engine>-emscripten.data is
// published, but each glue file asks for the .data named after the build that
// produced it. Emscripten routes that request through locateFile, so the
// redirect below is the only thing keeping a non-default SIMD tier from 404ing.
test('a variant glue requesting its own .data is redirected to the canonical one', () => {
  const requested = 'plentychess-emscripten-relaxed-simd128.data';
  assert.equal(resolveEmscriptenAssetUrl(requested, { jsUrl: JS, wasmUrl: WASM, dataUrl: DATA }), DATA);
});

test('the wasm sidecar stays per-variant', () => {
  assert.equal(resolveEmscriptenAssetUrl('plentychess-emscripten-relaxed-simd128.wasm', { jsUrl: JS, wasmUrl: WASM, dataUrl: DATA }), WASM);
});

test('unpinned assets resolve relative to the glue URL', () => {
  assert.equal(
    resolveEmscriptenAssetUrl('something.else', { jsUrl: JS, wasmUrl: WASM, dataUrl: DATA }),
    'https://assets.example/plentychess/something.else',
  );
  // No dataUrl pinned: fall back to sibling resolution rather than returning undefined.
  assert.equal(
    resolveEmscriptenAssetUrl('x.data', { jsUrl: JS, wasmUrl: null, dataUrl: null }),
    'https://assets.example/plentychess/x.data',
  );
});

test('redirection is extension-based, never name-based', () => {
  // Honouring the requested basename is exactly the bug: every tier asks for a
  // different .data name and they must all land on the same canonical file.
  for (const name of [
    'berserk-emscripten.data',
    'berserk-emscripten-simd128.data',
    'berserk-emscripten-relaxed-simd128.data',
  ]) {
    assert.equal(resolveEmscriptenAssetUrl(name, { jsUrl: JS, dataUrl: DATA }), DATA, name);
  }
});

test('every Emscripten engine adapter routes locateFile through the shared resolver', () => {
  // A worker that hand-rolls locateFile again would silently reintroduce the
  // per-variant .data assumption, and no unit test above would notice.
  for (const family of ['berserk', 'plentychess', 'stormphrax']) {
    const source = readFileSync(new URL(`../src/lc0/${family}Engine.ts`, import.meta.url), 'utf8');
    assert.match(source, /locateFile\(file\)\s*\{\s*return resolveEmscriptenAssetUrl\(/, `${family} must delegate locateFile`);
    assert.doesNotMatch(source, /endsWith\('\.data'\)/, `${family} must not re-implement .data routing`);
  }
});

// The resolver above is only half the contract. It is the *generated* glue that
// decides whether to consult Module.locateFile at all: if a future Emscripten
// version or build flag emitted a package fetch that bypassed the hook, every
// non-default SIMD tier would request a .data that is no longer published and
// the tests above would still pass. These assert against the real committed
// artifacts, so a regenerated glue that dropped the hook fails here.
// Berserk is listed even though its artifacts are never committed: the skip
// below keeps a clean checkout green, while anyone who has run
// `npm run berserk:build-emscripten` gets the same assertion applied to their
// locally generated glue, which is the only place it can be checked at all.
const GENERATED_GLUE = [
  ['plentychess', 'plentychess-emscripten.js'],
  ['plentychess', 'plentychess-emscripten-sse41.js'],
  ['plentychess', 'plentychess-emscripten-relaxed-simd128.js'],
  ['stormphrax', 'stormphrax-emscripten.js'],
  ['stormphrax', 'stormphrax-emscripten-relaxed-simd128.js'],
  ['berserk', 'berserk-emscripten.js'],
  ['berserk', 'berserk-emscripten-simd128.js'],
  ['berserk', 'berserk-emscripten-relaxed-simd128.js'],
];

for (const [family, file] of GENERATED_GLUE) {
  test(`generated glue ${file} resolves its preload package through Module.locateFile`, (t) => {
    const path = new URL(`../public/${family}/${file}`, import.meta.url);
    let source;
    try {
      source = readFileSync(path, 'utf8');
    } catch (error) {
      // Artifacts are build outputs; some checkouts legitimately lack them.
      // Skip rather than fail so this does not become a false red for someone
      // who has not built or pulled LFS.
      // Loud on purpose. Berserk's glue is never committed (its network has no
      // resolved license), so this assertion CANNOT run in clean CI and a quiet
      // skip would read as coverage it does not provide. The CI-runnable half
      // of this guarantee is the --preload-file assertion below.
      t.skip(`${file} absent — generated-glue assertion NOT exercised in this checkout (${error.code})`);
      return;
    }
    assert.match(
      source,
      /Module\["locateFile"\]\(REMOTE_PACKAGE_BASE/,
      `${file} must route its preload package through Module.locateFile, or the shared .data redirect cannot work`,
    );
  });
}

// The CI-runnable half of the Berserk guarantee.
//
// Emscripten only emits the package loader that consults Module.locateFile when
// the build packages a preload file; drop `--preload-file` and the generated
// glue stops routing the .data through the hook entirely, which is the exact
// regression the assertions above exist to catch. Berserk's glue is never
// committed, so for that family this is the only check that can run without a
// local toolchain — it verifies the cause rather than the effect.
test('every Emscripten build script still packages its preload file', () => {
  for (const family of ['berserk', 'plentychess', 'stormphrax']) {
    const script = readFileSync(new URL(`../scripts/build_${family}_emscripten.mjs`, import.meta.url), 'utf8');
    assert.match(script, /'--preload-file'/, `${family} build must package a preload file`);
  }
});
