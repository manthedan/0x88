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
