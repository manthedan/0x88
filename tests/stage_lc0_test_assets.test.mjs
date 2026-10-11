import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { releaseCatalogEntries } from '../scripts/engine_artifact_registry.mjs';
import { REQUIRED_ASSETS } from '../scripts/stage_lc0_test_assets.mjs';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const manifest = JSON.parse(readFileSync(new URL('../public/models/lc0/manifest.json', import.meta.url), 'utf8'));

function sha256(body) {
  return createHash('sha256').update(body).digest('hex');
}

function stage(args) {
  return spawnSync(process.execPath, ['scripts/stage_lc0_test_assets.mjs', ...args], { cwd: repoRoot, encoding: 'utf8' });
}

// spawnSync would block the test process event loop and deadlock an in-process fixture server.
function stageAsync(args) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['scripts/stage_lc0_test_assets.mjs', ...args], { cwd: repoRoot });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.on('error', (error) => resolve({ status: 1, stdout: Buffer.concat(stdout).toString('utf8'), stderr: error.message, pid: child.pid }));
    child.on('close', (status) =>
      resolve({ status: status ?? 1, stdout: Buffer.concat(stdout).toString('utf8'), stderr: Buffer.concat(stderr).toString('utf8'), pid: child.pid }),
    );
  });
}

test('LC0 test asset staging rejects a stable release that differs from committed integrity metadata', () => {
  const pinned = manifest.models.find((model) => model.url.endsWith('.batch1.f32.onnx'));
  assert.ok(pinned);
  const changedSha256 = '0'.repeat(64);
  assert.notEqual(changedSha256, pinned.sha256);
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'mutated-stable-channel',
    artifacts: [
      {
        logicalUrl: pinned.url,
        artifactUrl: `https://assets.example/artifacts/sha256/${changedSha256}/identity`,
        bytes: pinned.bytes,
        sha256: changedSha256,
      },
    ],
  };
  const releaseUrl = `data:application/json,${encodeURIComponent(JSON.stringify(release))}`;
  const channel = {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    releaseUrl,
  };
  const channelUrl = `data:application/json,${encodeURIComponent(JSON.stringify(channel))}`;
  const result = stage(['--channel-url', channelUrl, '--root', join(tmpdir(), 'lc0-stage-pinned-test'), '--check']);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Stable artifact release integrity does not match the committed LC0 manifest/);
});

test('the committed release catalog matches the committed LC0 manifest for every required fixture', () => {
  const channel = JSON.parse(readFileSync(new URL('../public/channels/stable.json', import.meta.url), 'utf8'));
  assert.match(channel.releaseManifestUrl, /^\/releases\//);
  const release = JSON.parse(readFileSync(new URL(`../public${channel.releaseManifestUrl}`, import.meta.url), 'utf8'));
  const entries = new Map(
    releaseCatalogEntries(release)
      .filter((entry) => entry.encoding === 'identity')
      .map((entry) => [entry.logicalUrl, entry]),
  );
  const pinned = new Map();
  for (const model of manifest.models ?? []) pinned.set(model.url, { bytes: model.bytes, sha256: model.sha256.toLowerCase() });
  for (const pack of manifest.packs ?? []) {
    pinned.set(pack.url, { bytes: pack.metadataBytes, sha256: pack.metadataSha256.toLowerCase() });
    const prefix = pack.url.slice(0, pack.url.lastIndexOf('/') + 1);
    for (const shard of pack.shards ?? []) pinned.set(`${prefix}${shard.file}`, { bytes: shard.bytes, sha256: shard.sha256.toLowerCase() });
  }
  for (const logicalUrl of REQUIRED_ASSETS) {
    const entry = entries.get(logicalUrl);
    assert.ok(entry, `commit-pinned release is missing ${logicalUrl}`);
    const published = {
      bytes: entry.artifact.raw?.bytes ?? entry.artifact.bytes,
      sha256: (entry.artifact.raw?.sha256 ?? entry.artifact.sha256).toLowerCase(),
    };
    assert.deepEqual(published, pinned.get(logicalUrl), logicalUrl);
    assert.match(entry.artifact.artifactUrl, /^https:\/\/assets\.0x88\.app\/artifacts\/sha256\//);
  }
});

test('a complete local cache verifies by hash and does not touch the network', async (t) => {
  const body = Buffer.from('pinned-lc0-fixture');
  const digest = sha256(body);
  const logicalUrl = '/models/lc0/tiny.onnx';
  const root = await mkdtemp(join(tmpdir(), 'lc0-stage-warm-'));
  const manifestDir = await mkdtemp(join(tmpdir(), 'lc0-stage-manifest-'));
  t.after(() => Promise.all([rm(root, { recursive: true, force: true }), rm(manifestDir, { recursive: true, force: true })]));
  const manifestPath = join(manifestDir, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify({ models: [{ url: logicalUrl, bytes: body.length, sha256: digest }] }));
  const assetPath = join(root, 'onnx', 'tiny.onnx');
  await mkdir(join(root, 'onnx'), { recursive: true });
  await writeFile(assetPath, body);
  const result = stage([
    '--root',
    root,
    '--manifest',
    manifestPath,
    '--required',
    logicalUrl,
    '--offline',
    '--channel-url',
    'http://127.0.0.1:1/channels/stable.json',
  ]);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.network, false);
  assert.equal(report.staged[0].status, 'verified');
  assert.equal(report.staged[0].sha256, digest);
  await writeFile(assetPath, Buffer.alloc(body.length, 1));
  const corrupt = stage(['--root', root, '--manifest', manifestPath, '--required', logicalUrl, '--offline']);
  assert.equal(corrupt.status, 1);
  assert.match(corrupt.stderr, /missing or corrupt/);
  assert.equal(sha256(await readFile(assetPath)), sha256(Buffer.alloc(body.length, 1)));
});

test('offline mode fails a missing asset and does not create a file', async (t) => {
  const body = Buffer.from('missing-lc0-fixture');
  const logicalUrl = '/models/lc0/tiny.onnx';
  const root = await mkdtemp(join(tmpdir(), 'lc0-stage-missing-'));
  const manifestDir = await mkdtemp(join(tmpdir(), 'lc0-stage-missing-manifest-'));
  t.after(() => Promise.all([rm(root, { recursive: true, force: true }), rm(manifestDir, { recursive: true, force: true })]));
  const manifestPath = join(manifestDir, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify({ models: [{ url: logicalUrl, bytes: body.length, sha256: sha256(body) }] }));
  const result = stage([
    '--root',
    root,
    '--manifest',
    manifestPath,
    '--required',
    logicalUrl,
    '--offline',
    '--channel-url',
    'http://127.0.0.1:1/channels/stable.json',
  ]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /missing or corrupt/);
  assert.match(result.stderr, /offline; refusing to fetch/);
  await assert.rejects(readFile(join(root, 'onnx', 'tiny.onnx')));
});

test('a missing asset is fetched from the pinned catalog and rejected when the bytes do not match', async (t) => {
  const body = Buffer.from('downloaded-lc0-fixture');
  const digest = sha256(body);
  const logicalUrl = '/models/lc0/tiny.onnx';
  const root = await mkdtemp(join(tmpdir(), 'lc0-stage-fetch-'));
  const manifestDir = await mkdtemp(join(tmpdir(), 'lc0-stage-fetch-manifest-'));
  let hits = 0;
  const server = createServer((request, response) => {
    hits += 1;
    const payload = request.url?.includes('bad=1') ? Buffer.from('not-the-pinned-bytes!!') : body;
    response.end(payload);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  t.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    await Promise.all([rm(root, { recursive: true, force: true }), rm(manifestDir, { recursive: true, force: true })]);
  });
  const manifestPath = join(manifestDir, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify({ models: [{ url: logicalUrl, bytes: body.length, sha256: digest }] }));
  const releasePath = join(manifestDir, 'release.json');
  const goodUrl = `http://127.0.0.1:${port}/artifacts/sha256/${digest}/tiny.onnx`;
  await writeFile(
    releasePath,
    JSON.stringify({
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'test-pin',
      artifacts: [{ logicalUrl, artifactUrl: goodUrl, bytes: body.length, sha256: digest }],
    }),
  );
  const checked = await stageAsync(['--root', root, '--manifest', manifestPath, '--required', logicalUrl, '--release', releasePath, '--check']);
  assert.equal(checked.status, 1);
  assert.match(checked.stderr, /missing or corrupt/);
  assert.equal(hits, 0);
  const downloaded = await stageAsync(['--root', root, '--manifest', manifestPath, '--required', logicalUrl, '--release', releasePath]);
  assert.equal(downloaded.status, 0, downloaded.stderr);
  assert.equal(hits, 1);
  const report = JSON.parse(downloaded.stdout);
  assert.equal(report.network, true);
  assert.equal(report.releaseId, 'test-pin');
  assert.equal(sha256(await readFile(join(root, 'onnx', 'tiny.onnx'))), digest);

  await rm(join(root, 'onnx', 'tiny.onnx'));
  const badUrl = `http://127.0.0.1:${port}/artifacts/sha256/${digest}/tiny.onnx?bad=1`;
  await writeFile(
    releasePath,
    JSON.stringify({
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'test-pin-bad-bytes',
      artifacts: [{ logicalUrl, artifactUrl: badUrl, bytes: body.length, sha256: digest }],
    }),
  );
  const rejected = await stageAsync(['--root', root, '--manifest', manifestPath, '--required', logicalUrl, '--release', releasePath]);
  assert.equal(rejected.status, 1);
  assert.match(rejected.stderr, /Artifact integrity mismatch/);
  await assert.rejects(readFile(join(root, 'onnx', 'tiny.onnx')));
  await assert.rejects(readFile(join(root, 'onnx', `tiny.onnx.partial-${rejected.pid}`)));
});
