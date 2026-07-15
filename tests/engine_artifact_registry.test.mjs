import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BROWSER_ENGINE_ASSET_GROUPS,
  EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES,
  PRECOMPRESS_ARTIFACT_DIRECTORIES,
  buildArtifactReleaseCatalog,
  isExternalArtifactName,
  releaseCatalogEntries,
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

test('Reckless registry includes the external SIMD WASI artifact', () => {
  const group = BROWSER_ENGINE_ASSET_GROUPS.find((entry) => entry.family === 'reckless');
  assert.ok(group);
  assert.equal(group.command, 'npm run reckless:build-release');
  assert.ok(group.optionalAssets.includes('/reckless/reckless-simd128-external.wasm'));
});

test('external artifact classifier covers deployable binaries and keeps manifests', () => {
  for (const name of ['engine.wasm', 'engine.data', 'net.nnue', 'model.onnx', 'source.tar.gz', 'engine.js.br']) {
    assert.equal(isExternalArtifactName(name), true, name);
  }
  for (const name of ['README.md', 'engine.manifest.json', 'NOTICE.txt']) {
    assert.equal(isExternalArtifactName(name), false, name);
  }
});

test('shared release catalog reads v1 and v2 representation keys and deduplicates aliases', () => {
  const rawSha = 'a'.repeat(64);
  const brSha = 'b'.repeat(64);
  const v1Key = `artifacts/sha256/${'c'.repeat(64)}/legacy.wasm`;
  const identityKey = `artifacts/sha256/${rawSha}/identity`;
  const brKey = `artifacts/sha256/${rawSha}/br/${brSha}`;
  const migratedV1Key = identityKey.replace(/\/identity$/, '/model-a.onnx');
  const releases = [
    {
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'v1',
      artifacts: [{ logicalUrl: '/legacy.wasm', artifactUrl: `/${v1Key}` }],
    },
    {
      schema: 'lc0_browser.artifact_release_manifest.v2',
      releaseId: 'v2',
      artifacts: [
        {
          logicalUrl: '/model-a.onnx',
          kind: 'model',
          carriedForwardFrom: 'v1',
          migrationSource: {
            schema: 'lc0_browser.artifact_migration_source.v1',
            releaseId: 'v1',
            key: migratedV1Key,
            url: `/${migratedV1Key}`,
          },
          raw: { sha256: rawSha, bytes: 3 },
          representations: [
            { encoding: 'identity', url: `/${identityKey}`, sha256: rawSha, bytes: 3 },
            { encoding: 'br', url: `/${brKey}`, sha256: brSha, bytes: 2 },
          ],
        },
        {
          logicalUrl: '/model-b.onnx',
          kind: 'source',
          raw: { sha256: rawSha, bytes: 3 },
          representations: [{ encoding: 'identity', url: `/${identityKey}`, sha256: rawSha, bytes: 3 }],
        },
      ],
    },
  ];

  assert.equal(releaseCatalogEntries(releases[1]).length, 3);
  const catalog = buildArtifactReleaseCatalog(releases);
  assert.deepEqual([...catalog.keys()].sort(), [brKey, identityKey, v1Key].sort());
  assert.deepEqual(catalog.get(identityKey).logicalUrls, ['/model-a.onnx', '/model-b.onnx']);
  assert.deepEqual(catalog.get(identityKey).releases, ['v2']);
  assert.deepEqual(catalog.get(identityKey).kinds, ['model', 'source']);
  assert.equal(catalog.has(migratedV1Key), false);
});

test('shared release catalog fails closed on corrupt v2 representation keys', () => {
  const rawSha = 'a'.repeat(64);
  const wrongSha = 'b'.repeat(64);
  assert.throws(() => buildArtifactReleaseCatalog([{
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'corrupt',
    artifacts: [{
      logicalUrl: '/model.onnx',
      raw: { sha256: rawSha, bytes: 3 },
      representations: [{
        encoding: 'identity',
        url: `/artifacts/sha256/${wrongSha}/identity`,
        sha256: rawSha,
        bytes: 3,
      }],
    }],
  }]), /Invalid identity representation/);
});

test('shared release catalog requires exactly one identity representation per v2 artifact', () => {
  const rawSha = 'a'.repeat(64);
  const brSha = 'b'.repeat(64);
  const artifact = {
    logicalUrl: '/model.onnx',
    raw: { sha256: rawSha, bytes: 3 },
  };
  const release = (representations) => ({
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'identity-contract',
    artifacts: [{ ...artifact, representations }],
  });
  const identity = {
    encoding: 'identity',
    url: `/artifacts/sha256/${rawSha}/identity`,
    sha256: rawSha,
    bytes: 3,
  };
  const br = {
    encoding: 'br',
    url: `/artifacts/sha256/${rawSha}/br/${brSha}`,
    sha256: brSha,
    bytes: 2,
  };

  assert.throws(
    () => buildArtifactReleaseCatalog([release([br])]),
    /must have exactly one identity representation.*found 0/,
  );
  assert.throws(
    () => buildArtifactReleaseCatalog([release([identity, { ...identity }])]),
    /must have exactly one identity representation.*found 2/,
  );
  assert.throws(
    () => buildArtifactReleaseCatalog([{
      ...release([]),
      artifacts: [{ ...artifact, artifactUrl: `/artifacts/sha256/${rawSha}/legacy.onnx` }],
    }]),
    /V2 artifact has no representations/,
  );
  assert.throws(
    () => buildArtifactReleaseCatalog([{
      ...release([]),
      artifacts: [{
        ...artifact,
        artifactUrl: `/artifacts/sha256/${rawSha}/legacy.onnx`,
        representations: [identity],
      }],
    }]),
    /V2 artifact contains legacy artifactUrl/,
  );
});
