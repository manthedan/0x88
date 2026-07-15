import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
const DEFAULT_RELEASE_OUTPUT = '.local-dev-artifacts/artifact-releases';

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

function runGenerator(root) {
  return spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
}

function identity(rawSha256 = ABC_SHA256, bytes = 3) {
  return {
    encoding: 'identity',
    url: `/artifacts/sha256/${rawSha256}/identity`,
    sha256: rawSha256,
    bytes,
  };
}

function baseArtifact(logicalUrl, overrides = {}) {
  return {
    logicalUrl,
    raw: { sha256: ABC_SHA256, bytes: 3 },
    representations: [identity()],
    ...overrides,
  };
}

test('release generator rejects conflicting deduplicated raw alias metadata before manifest writes', async () => {
  const cases = [
    {
      name: 'bytes',
      artifacts: [
        baseArtifact('/models/a.onnx'),
        baseArtifact('/models/b.onnx', {
          raw: { sha256: ABC_SHA256, bytes: 4 },
          representations: [identity(ABC_SHA256, 4)],
        }),
      ],
      expected: /Conflicting raw byte lengths/,
    },
    {
      name: 'hash',
      artifacts: [
        baseArtifact('/models/a.onnx'),
        baseArtifact('/models/b.onnx', { sha256: 'f'.repeat(64), bytes: 3 }),
      ],
      expected: /Conflicting raw SHA-256 metadata/,
    },
    {
      name: 'migration',
      artifacts: [
        baseArtifact('/models/a.onnx', {
          carriedForwardFrom: 'legacy-v1',
          migrationSource: {
            schema: 'lc0_browser.artifact_migration_source.v1',
            releaseId: 'legacy-v1',
            key: `artifacts/sha256/${ABC_SHA256}/model.onnx`,
            url: `/artifacts/sha256/${ABC_SHA256}/model.onnx`,
          },
        }),
        baseArtifact('/models/b.onnx', {
          carriedForwardFrom: 'legacy-v1',
          migrationSource: {
            schema: 'lc0_browser.artifact_migration_source.v1',
            releaseId: 'legacy-v1',
            key: `artifacts/sha256/${ABC_SHA256}/model.onnx`,
            url: `https://assets.example/artifacts/sha256/${ABC_SHA256}/model.onnx`,
          },
        }),
      ],
      expected: /Incompatible migration provenance/,
    },
    {
      name: 'content-type',
      artifacts: [
        baseArtifact('/models/a.onnx', { contentType: 'application/octet-stream' }),
        baseArtifact('/engines/a.wasm', { contentType: 'application/wasm' }),
      ],
      expected: /Incompatible contentType metadata/,
    },
  ];

  for (const scenario of cases) {
    const root = await mkdtemp(join(tmpdir(), `lc0-release-alias-${scenario.name}-`));
    await writeJson(join(root, 'base.json'), {
      schema: 'lc0_browser.artifact_release_manifest.v2',
      releaseId: 'base',
      artifacts: scenario.artifacts,
    });

    const result = runGenerator(root);

    assert.notEqual(result.status, 0, scenario.name);
    assert.match(result.stderr, scenario.expected);
    assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json')), false);
    assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json')), false);
  }
});

test('release generator accepts compatible raw aliases and keeps one shared identity', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-compatible-aliases-'));
  const migrationSource = (file) => ({
    schema: 'lc0_browser.artifact_migration_source.v1',
    releaseId: 'legacy-v1',
    key: `artifacts/sha256/${ABC_SHA256}/${file}`,
    url: `/artifacts/sha256/${ABC_SHA256}/${file}`,
  });
  await writeJson(join(root, 'base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'base',
    artifacts: [
      {
        ...baseArtifact('/models/a.onnx'),
        contentType: 'application/octet-stream',
        carriedForwardFrom: 'legacy-v1',
        migrationSource: migrationSource('a.onnx'),
      },
      {
        ...baseArtifact('/models/b.onnx'),
        contentType: 'application/octet-stream',
        carriedForwardFrom: 'legacy-v1',
        migrationSource: migrationSource('b.onnx'),
      },
    ],
  });

  const result = runGenerator(root);

  assert.equal(result.status, 0, result.stderr);
  const release = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json'), 'utf8'));
  assert.equal(release.artifacts.length, 2);
  assert.equal(release.artifacts[0].representations[0].url, release.artifacts[1].representations[0].url);
  assert.deepEqual(release.artifacts.map((artifact) => artifact.migrationSource.key).sort(), [
    `artifacts/sha256/${ABC_SHA256}/a.onnx`,
    `artifacts/sha256/${ABC_SHA256}/b.onnx`,
  ]);
});

test('release generator rejects incompatible local alias content types before materializing representations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-local-content-type-aliases-'));
  await mkdir(join(root, 'public/models'), { recursive: true });
  await mkdir(join(root, 'public/engines'), { recursive: true });
  await writeFile(join(root, 'public/models/a.onnx'), 'abc');
  await writeFile(join(root, 'public/engines/a.wasm'), 'abc');
  await writeJson(join(root, 'public/models/manifest.json'), {
    models: [{ file: 'a.onnx', url: '/models/a.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
  await writeJson(join(root, 'public/engines/manifest.json'), {
    artifacts: [{ path: 'public/engines/a.wasm', bytes: 3, sha256: ABC_SHA256 }],
  });

  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--manifest', 'public/models/manifest.json',
    '--manifest', 'public/engines/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Incompatible contentType metadata/);
  assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'artifacts/sha256', ABC_SHA256, 'identity')), false);
  assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json')), false);
  assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json')), false);
});

test('release generator materializes compatible local alias content types once', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-local-compatible-aliases-'));
  await mkdir(join(root, 'public/models'), { recursive: true });
  await writeFile(join(root, 'public/models/a.onnx'), 'abc');
  await writeFile(join(root, 'public/models/b.onnx'), 'abc');
  await writeJson(join(root, 'public/models/manifest.json'), {
    models: [
      { file: 'a.onnx', url: '/models/a.onnx', bytes: 3, sha256: ABC_SHA256 },
      { file: 'b.onnx', url: '/models/b.onnx', bytes: 3, sha256: ABC_SHA256 },
    ],
  });

  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--manifest', 'public/models/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
  const release = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json'), 'utf8'));
  const aliases = release.artifacts.filter((artifact) => artifact.logicalUrl === '/models/a.onnx' || artifact.logicalUrl === '/models/b.onnx');
  assert.equal(aliases.length, 2);
  assert.deepEqual(aliases.map((artifact) => artifact.contentType), ['application/octet-stream', 'application/octet-stream']);
  assert.equal(aliases[0].representations[0].url, aliases[1].representations[0].url);
  assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'artifacts/sha256', ABC_SHA256, 'identity')), true);
});
