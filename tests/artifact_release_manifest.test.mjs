import assert from 'node:assert/strict';
import { chmod, mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { test } from 'node:test';

const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

async function listen(server) {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return server.address().port;
}

async function runNode(args, options = {}) {
  return await new Promise((resolve) => {
    const child = spawn(process.execPath, args, { cwd: process.cwd(), ...options, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
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

test('publish_hashed_artifacts_to_r2 plans release and channel manifest uploads', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-manifests-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const releasePath = join(root, 'public/releases/test-release.json');
  const channelPath = join(root, 'public/channels/stable.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'test-release',
    artifacts: [{
      logicalUrl: '/models/lc0/test.onnx',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/test.onnx`,
      sha256: ABC_SHA256,
      bytes: 3,
      file: 'test.onnx',
      kind: 'model',
      sourceManifest: 'test',
      localPath: 'public/models/lc0/test.onnx',
    }],
  });
  await writeJson(channelPath, {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    channel: 'stable',
    releaseId: 'test-release',
    releaseManifestUrl: '/releases/test-release.json',
  });
  const result = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--channel-manifest', channelPath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const parsed = JSON.parse(result.stdout);
  assert.deepEqual(parsed.manifests.map((item) => [item.type, item.key, item.cacheControl]), [
    ['release-manifest', 'releases/test-release.json', 'public, max-age=300, stale-while-revalidate=86400'],
    ['channel-manifest', 'channels/stable.json', 'no-cache'],
  ]);
});

test('publish_hashed_artifacts_to_r2 skips existing validated artifact uploads', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-existing-artifact-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const server = createServer((req, res) => {
    if (req.url === `/artifacts/sha256/${ABC_SHA256}/test.onnx`) {
      const headers = {
        'X-Artifact-Content-Length': '3',
        'Cache-Control': 'public, max-age=31536000, immutable',
      };
      if (req.method === 'HEAD') res.writeHead(200, headers).end();
      else res.writeHead(200, { ...headers, 'Content-Length': '3' }).end('abc');
      return;
    }
    res.writeHead(404).end();
  });
  const port = await listen(server);
  try {
    const releasePath = join(root, 'public/releases/test-release.json');
    await writeJson(releasePath, {
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'test-release',
      artifacts: [{
        logicalUrl: '/models/lc0/test.onnx',
        artifactUrl: `http://127.0.0.1:${port}/artifacts/sha256/${ABC_SHA256}/test.onnx`,
        sha256: ABC_SHA256,
        bytes: 3,
        file: 'test.onnx',
        kind: 'model',
        sourceManifest: 'test',
        localPath: 'public/models/lc0/test.onnx',
      }],
    });
    const logPath = join(root, 'wrangler.log');
    const wrangler = join(root, 'fake-wrangler.sh');
    await writeFile(wrangler, '#!/bin/sh\nprintf "%s\\n" "$*" >> "$LOG"\nif [ "$1 $2 $3" = "r2 object get" ]; then exit 1; fi\nexit 0\n');
    await chmod(wrangler, 0o755);
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--execute',
      '--wrangler-bin', wrangler,
    ], { env: { ...process.env, LOG: logPath } });
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.planned[0].remoteState, 'existing');
    assert.equal(parsed.planned[0].uploadAction, 'skip-existing');
    const log = await readFile(logPath, 'utf8');
    assert.doesNotMatch(log, /artifacts\/sha256/);
    assert.match(log, /r2 object put test-bucket\/releases\/test-release\.json/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects existing artifact size mismatches', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-bad-existing-artifact-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const server = createServer((req, res) => {
    if (req.method === 'HEAD') {
      res.writeHead(200, {
        'X-Artifact-Content-Length': '2',
        'Cache-Control': 'public, max-age=31536000, immutable',
      }).end();
      return;
    }
    res.writeHead(200, { 'Content-Length': '3' }).end('abc');
  });
  const port = await listen(server);
  try {
    const releasePath = join(root, 'public/releases/test-release.json');
    await writeJson(releasePath, {
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'test-release',
      artifacts: [{
        logicalUrl: '/models/lc0/test.onnx',
        artifactUrl: `http://127.0.0.1:${port}/artifacts/sha256/${ABC_SHA256}/test.onnx`,
        sha256: ABC_SHA256,
        bytes: 3,
        file: 'test.onnx',
        kind: 'model',
        sourceManifest: 'test',
        localPath: 'public/models/lc0/test.onnx',
      }],
    });
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--probe-existing',
    ]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Remote artifact size mismatch/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects existing artifact hash mismatches', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-bad-existing-artifact-hash-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const server = createServer((req, res) => {
    const headers = {
      'X-Artifact-Content-Length': '3',
      'Cache-Control': 'public, max-age=31536000, immutable',
    };
    if (req.method === 'HEAD') res.writeHead(200, headers).end();
    else res.writeHead(200, { ...headers, 'Content-Length': '3' }).end('abd');
  });
  const port = await listen(server);
  try {
    const releasePath = join(root, 'public/releases/test-release.json');
    await writeJson(releasePath, {
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'test-release',
      artifacts: [{
        logicalUrl: '/models/lc0/test.onnx',
        artifactUrl: `http://127.0.0.1:${port}/artifacts/sha256/${ABC_SHA256}/test.onnx`,
        sha256: ABC_SHA256,
        bytes: 3,
        file: 'test.onnx',
        kind: 'model',
        sourceManifest: 'test',
        localPath: 'public/models/lc0/test.onnx',
      }],
    });
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--probe-existing',
    ]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Remote artifact SHA-256 mismatch/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects a stale channel manifest', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-stale-channel-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const releasePath = join(root, 'public/releases/test-release.json');
  const channelPath = join(root, 'public/channels/stable.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'test-release',
    artifacts: [{
      logicalUrl: '/models/lc0/test.onnx',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/test.onnx`,
      sha256: ABC_SHA256,
      bytes: 3,
      file: 'test.onnx',
      kind: 'model',
      sourceManifest: 'test',
      localPath: 'public/models/lc0/test.onnx',
    }],
  });
  await writeJson(channelPath, {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    channel: 'stable',
    releaseId: 'old-release',
    releaseManifestUrl: '/releases/old-release.json',
  });
  const result = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--channel-manifest', channelPath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /does not match release test-release/);
});

test('publish_hashed_artifacts_to_r2 refuses to overwrite release manifests', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-existing-release-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  const releasePath = join(root, 'public/releases/test-release.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'test-release',
    artifacts: [{
      logicalUrl: '/models/lc0/test.onnx',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/test.onnx`,
      sha256: ABC_SHA256,
      bytes: 3,
      file: 'test.onnx',
      kind: 'model',
      sourceManifest: 'test',
      localPath: 'public/models/lc0/test.onnx',
    }],
  });
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const wrangler = join(root, 'fake-wrangler.sh');
    await writeFile(wrangler, '#!/bin/sh\nif [ "$1 $2 $3" = "r2 object get" ]; then exit 0; fi\nexit 0\n');
    await chmod(wrangler, 0o755);
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
    ]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Refusing to overwrite immutable release manifest/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 refuses execute when artifacts are skipped', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-skipped-'));
  const releasePath = join(root, 'public/releases/test-release.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'test-release',
    artifacts: [{
      logicalUrl: '/models/lc0/missing.onnx',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/missing.onnx`,
      sha256: ABC_SHA256,
      bytes: 3,
      file: 'missing.onnx',
      kind: 'model',
      sourceManifest: 'test',
      localPath: 'public/models/lc0/missing.onnx',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--bucket', 'test-bucket',
    '--allow-missing',
    '--execute',
    '--wrangler-bin', process.execPath,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Refusing to publish release\/channel manifests when artifacts were skipped/);
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
