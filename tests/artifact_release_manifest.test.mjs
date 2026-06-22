import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

test('write_artifact_release_manifests creates channel and content-addressed release manifests', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-manifest-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await mkdir(join(root, 'public/stockfish'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  await writeFile(join(root, 'public/stockfish/engine.wasm'), 'abc');
  await writeJson(join(root, 'public/models/lc0/manifest.json'), {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: ABC_SHA256, mode: 'symlink' }],
  });
  await writeJson(join(root, 'public/stockfish/stockfish.manifest.json'), {
    artifacts: [{ path: 'public/stockfish/engine.wasm', bytes: 3, sha256: ABC_SHA256 }],
  });

  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'test-release',
    '--channel', 'stable',
    '--generated-at', '2026-06-22T00:00:00.000Z',
    '--manifest', 'public/models/lc0/manifest.json',
    '--manifest', 'public/stockfish/stockfish.manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);

  const channel = JSON.parse(await readFile(join(root, 'public/channels/stable.json'), 'utf8'));
  assert.equal(channel.schema, 'lc0_browser.artifact_channel_manifest.v1');
  assert.equal(channel.releaseManifestUrl, '/releases/test-release.json');

  const release = JSON.parse(await readFile(join(root, 'public/releases/test-release.json'), 'utf8'));
  assert.equal(release.schema, 'lc0_browser.artifact_release_manifest.v1');
  assert.equal(release.releaseId, 'test-release');
  assert.equal(release.artifacts.length, 2);
  assert.deepEqual(release.artifacts.map((artifact) => artifact.logicalUrl).sort(), ['/models/lc0/test.onnx', '/stockfish/engine.wasm']);
  for (const artifact of release.artifacts) {
    assert.match(artifact.artifactUrl, new RegExp(`/artifacts/sha256/${ABC_SHA256}/`));
    assert.equal(artifact.sha256, ABC_SHA256);
    assert.equal(artifact.bytes, 3);
  }

  const check = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'test-release',
    '--channel', 'stable',
    '--generated-at', '2026-06-22T00:00:00.000Z',
    '--manifest', 'public/models/lc0/manifest.json',
    '--manifest', 'public/stockfish/stockfish.manifest.json',
    '--check',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(check.status, 0, check.stderr);
});

test('publish_hashed_artifacts_to_r2 rejects artifactUrl hash mismatches', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const wrongSha = '0'.repeat(64);
  const releasePath = join(root, 'release.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'bad-key',
    artifacts: [{
      logicalUrl: '/models/lc0/test.onnx',
      artifactUrl: `/artifacts/sha256/${wrongSha}/test.onnx`,
      sha256: ABC_SHA256,
      bytes: 3,
      file: 'test.onnx',
      kind: 'model',
      sourceManifest: 'test',
      localPath: 'public/models/lc0/test.onnx',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Content-addressed key mismatch/);
});
