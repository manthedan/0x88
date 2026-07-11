import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BROWSER_ENGINE_ASSET_GROUPS,
  EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES,
  PRECOMPRESS_ARTIFACT_DIRECTORIES,
  isExternalArtifactName,
} from '../scripts/engine_artifact_registry.mjs';

const EXPECTED_BROWSER_FAMILIES = ['lc0', 'stockfish', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax'];

test('browser artifact registry has unique family ids and deployment directories', () => {
  const families = BROWSER_ENGINE_ASSET_GROUPS.map((group) => group.family);
  assert.deepEqual(families, EXPECTED_BROWSER_FAMILIES);
  assert.equal(new Set(families).size, families.length);
  assert.equal(new Set(EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES).size, EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES.length);

  for (const family of families.filter((entry) => !['lc0'].includes(entry))) {
    assert.ok(EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES.includes(family), `${family} deployment directory`);
  }
  for (const directory of EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES.filter((entry) => entry !== 'runtimes')) {
    assert.ok(PRECOMPRESS_ARTIFACT_DIRECTORIES.includes(directory), `${directory} precompression directory`);
  }
});

test('browser artifact groups expose absolute URL paths and preparation metadata', () => {
  for (const group of BROWSER_ENGINE_ASSET_GROUPS) {
    assert.ok(group.assets.length > 0, `${group.family} required assets`);
    assert.ok(group.command.startsWith('npm '), `${group.family} preparation command`);
    assert.ok(group.docs.includes('#'), `${group.family} documentation anchor`);
    for (const asset of [...group.assets, ...(group.optionalAssets ?? [])]) {
      assert.ok(asset.startsWith('/'), `${group.family} absolute asset path`);
    }
  }
});

test('Stormphrax registry requires baseline and relaxed SIMD sidecars', () => {
  const group = BROWSER_ENGINE_ASSET_GROUPS.find((entry) => entry.family === 'stormphrax');
  assert.ok(group);
  assert.match(group.command, /stormphrax:build-relaxed-simd-emscripten/);
  assert.deepEqual(group.assets.filter((asset) => asset.includes('relaxed-simd128')), [
    '/stormphrax/stormphrax-emscripten-relaxed-simd128.js',
    '/stormphrax/stormphrax-emscripten-relaxed-simd128.wasm',
    '/stormphrax/stormphrax-emscripten-relaxed-simd128.data',
  ]);
});

test('external artifact classifier covers deployable binaries and keeps manifests', () => {
  for (const name of ['engine.wasm', 'engine.data', 'net.nnue', 'model.onnx', 'source.tar.gz', 'engine.js.br']) {
    assert.equal(isExternalArtifactName(name), true, name);
  }
  for (const name of ['README.md', 'engine.manifest.json', 'NOTICE.txt']) {
    assert.equal(isExternalArtifactName(name), false, name);
  }
});
