import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { chmod, mkdtemp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { brotliCompressSync, brotliDecompressSync } from 'node:zlib';

const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
const DEFAULT_RELEASE_OUTPUT = '.local-dev-artifacts/artifact-releases';

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

function mockR2ObjectPath(root, target) {
  return join(root, createHash('sha256').update(target).digest('hex'));
}

function mockR2MetadataPath(root, target) {
  return `${mockR2ObjectPath(root, target)}.metadata.json`;
}

async function writeMockR2Object(root, target, body, metadata) {
  await mkdir(root, { recursive: true });
  await writeFile(mockR2ObjectPath(root, target), body);
  await writeJson(mockR2MetadataPath(root, target), metadata);
}

async function writeStatefulWrangler(root) {
  const path = join(root, 'fake-wrangler.mjs');
  await writeFile(path, `#!/usr/bin/env node
import { appendFileSync, copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';

const args = process.argv.slice(2);
if (process.env.LOG) appendFileSync(process.env.LOG, \`\${args.join(' ')}\\n\`);
const target = args[3];
const objectPath = join(process.env.MOCK_R2_DIR, createHash('sha256').update(target).digest('hex'));
if (args[0] === 'r2' && args[1] === 'object' && args[2] === 'get') {
  if (!existsSync(objectPath)) {
    console.error('The specified key does not exist.');
    process.exit(1);
  }
  const file = args[args.indexOf('--file') + 1];
  mkdirSync(dirname(file), { recursive: true });
  copyFileSync(objectPath, file);
  process.exit(0);
}
if (args[0] === 'r2' && args[1] === 'object' && args[2] === 'put') {
  const file = args[args.indexOf('--file') + 1];
  mkdirSync(dirname(objectPath), { recursive: true });
  copyFileSync(file, objectPath);
  process.exit(0);
}
process.exit(1);
`);
  await chmod(path, 0o755);
  return path;
}

async function writeAtomicAws(root) {
  const path = join(root, 'fake-aws.mjs');
  await writeFile(path, `#!/usr/bin/env node
import { appendFileSync, existsSync, linkSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';

const args = process.argv.slice(2);
if (process.env.LOG) appendFileSync(process.env.LOG, \`\${args.join(' ')}\\n\`);
const bucket = args[args.indexOf('--bucket') + 1];
const key = args[args.indexOf('--key') + 1];
const target = \`\${bucket}/\${key}\`;
const objectPath = join(process.env.MOCK_R2_DIR, createHash('sha256').update(target).digest('hex'));
const metadataPath = \`\${objectPath}.metadata.json\`;
if (args[0] === 's3api' && args[1] === 'head-object') {
  if (!existsSync(objectPath) || !existsSync(metadataPath)) {
    console.error('NoSuchKey: status code: 404');
    process.exit(1);
  }
  process.stdout.write(readFileSync(metadataPath, 'utf8'));
  process.exit(0);
}
if (
  args[0] !== 's3api'
  || args[1] !== 'put-object'
  || args[args.indexOf('--if-none-match') + 1] !== '*'
  || args[args.indexOf('--region') + 1] !== 'auto'
) process.exit(2);
const file = args[args.indexOf('--body') + 1];
mkdirSync(dirname(objectPath), { recursive: true });
try {
  linkSync(file, objectPath);
} catch (error) {
  if (error.code === 'EEXIST') {
    console.error('PreconditionFailed: status code: 412');
    process.exit(1);
  }
  throw error;
}
const contentEncodingIndex = args.indexOf('--content-encoding');
const customMetadataIndex = args.indexOf('--metadata');
writeFileSync(metadataPath, JSON.stringify({
  ContentLength: statSync(file).size,
  ContentType: args[args.indexOf('--content-type') + 1],
  CacheControl: args[args.indexOf('--cache-control') + 1],
  ...(contentEncodingIndex >= 0 ? { ContentEncoding: args[contentEncodingIndex + 1] } : {}),
  ...(customMetadataIndex >= 0 ? { Metadata: JSON.parse(args[customMetadataIndex + 1]) } : {}),
}));
`);
  await chmod(path, 0o755);
  return path;
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

  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const channel = JSON.parse(await readFile(join(outputRoot, 'channels/stable.json'), 'utf8'));
  assert.equal(channel.schema, 'lc0_browser.artifact_channel_manifest.v2');
  assert.equal(channel.releaseManifestUrl, '/releases/test-release.json');
  assert.equal(channel.releaseUrl, '/releases/test-release.json');

  const releasePath = join(outputRoot, 'releases/test-release.json');
  const release = JSON.parse(await readFile(releasePath, 'utf8'));
  assert.equal(release.schema, 'lc0_browser.artifact_release_manifest.v2');
  assert.equal(release.releaseId, 'test-release');
  assert.equal(release.artifacts.length, 4);
  assert.deepEqual(release.artifacts.map((artifact) => artifact.logicalUrl).sort(), [
    '/models/lc0/manifest.json',
    '/models/lc0/test.onnx',
    '/stockfish/engine.wasm',
    '/stockfish/stockfish.manifest.json',
  ]);
  for (const artifact of release.artifacts.filter((entry) => entry.kind !== 'manifest')) {
    assert.equal(artifact.raw.sha256, ABC_SHA256);
    assert.equal(artifact.raw.bytes, 3);
    assert.equal(artifact.sha256, ABC_SHA256);
    assert.equal(artifact.bytes, 3);
    assert.deepEqual(artifact.representations.map((entry) => entry.encoding), ['identity', 'br']);
    assert.match(artifact.representations[0].url, new RegExp(`/artifacts/sha256/${ABC_SHA256}/identity$`));
    assert.doesNotMatch(artifact.representations[0].url, /test\.onnx|engine\.wasm/);
    assert.match(artifact.representations[1].url, new RegExp(`/artifacts/sha256/${ABC_SHA256}/br/[a-f0-9]{64}$`));
  }
  assert.equal(release.artifacts.filter((entry) => entry.kind === 'manifest').length, 2);
  const equalBodies = release.artifacts.filter((entry) => entry.raw?.sha256 === ABC_SHA256);
  assert.equal(equalBodies.length, 2);
  assert.equal(equalBodies[0].representations[0].url, equalBodies[1].representations[0].url);
  assert.equal(equalBodies[0].representations[1].url, equalBodies[1].representations[1].url);
  const identityPath = join(outputRoot, new URL(equalBodies[0].representations[0].url).pathname);
  const brPath = join(outputRoot, new URL(equalBodies[0].representations[1].url).pathname);
  assert.equal((await readFile(identityPath)).toString(), 'abc');
  assert.equal(brotliDecompressSync(await readFile(brPath)).toString(), 'abc');
  assert.equal(existsSync(join(root, 'public/artifacts')), false);
  assert.equal(existsSync(join(root, 'public/releases')), false);
  assert.equal(existsSync(join(root, 'public/channels')), false);

  const publishPlan = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--channel-manifest', join(outputRoot, 'channels/stable.json'),
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(publishPlan.status, 0, publishPlan.stderr);
  const planned = JSON.parse(publishPlan.stdout).planned;
  assert.ok(planned.every((entry) => entry.localPath.startsWith(outputRoot)));
  assert.ok(planned.every((entry) => existsSync(entry.localPath)));

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

test('write_artifact_release_manifests check mode works with a read-only staging tree', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-manifest-read-only-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  await writeJson(join(root, 'public/models/lc0/manifest.json'), {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
  const args = [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'read-only-check',
    '--generated-at', '2026-07-14T00:00:00.000Z',
    '--manifest', 'public/models/lc0/manifest.json',
  ];
  const generated = spawnSync(process.execPath, args, { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(generated.status, 0, generated.stderr);

  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const releasePath = join(outputRoot, 'releases/read-only-check.json');
  const channelPath = join(outputRoot, 'channels/stable.json');
  const originalRelease = await readFile(releasePath, 'utf8');
  const originalChannel = await readFile(channelPath, 'utf8');
  await chmod(outputRoot, 0o555);
  try {
    const checked = spawnSync(process.execPath, [...args, '--check'], { cwd: process.cwd(), encoding: 'utf8' });
    assert.equal(checked.status, 0, checked.stderr);
  } finally {
    await chmod(outputRoot, 0o755);
  }
  assert.equal(await readFile(releasePath, 'utf8'), originalRelease);
  assert.equal(await readFile(channelPath, 'utf8'), originalChannel);
});

test('write_artifact_release_manifests includes TVMJS runtime files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'centipawn-tvmjs-release-manifest-'));
  const runtimeDir = join(root, 'public/runtimes/centipawn-tvmjs-webgpu/model/f32/v2');
  await mkdir(runtimeDir, { recursive: true });
  for (const file of ['tvmjs.bundle.js', 'tvmjs_runtime.wasm', 'model.tvmjs.wasm']) {
    await writeFile(join(runtimeDir, file), 'abc');
  }
  await writeJson(join(runtimeDir, 'manifest.json'), {
    schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
    files: [
      { path: 'tvmjs.bundle.js', bytes: 3, sha256: ABC_SHA256 },
      { path: 'tvmjs_runtime.wasm', bytes: 3, sha256: ABC_SHA256 },
      { path: 'model.tvmjs.wasm', bytes: 3, sha256: ABC_SHA256 },
    ],
  });

  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'centipawn-tvmjs',
    '--manifest', 'public/runtimes/centipawn-tvmjs-webgpu/model/f32/v2/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);

  const release = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/centipawn-tvmjs.json'), 'utf8'));
  assert.equal(release.schema, 'lc0_browser.artifact_release_manifest.v2');
  assert.deepEqual(release.artifacts.map((artifact) => [artifact.logicalUrl, artifact.kind, artifact.contentType]), [
    ['/runtimes/centipawn-tvmjs-webgpu/model/f32/v2/manifest.json', 'manifest', 'application/json'],
    ['/runtimes/centipawn-tvmjs-webgpu/model/f32/v2/model.tvmjs.wasm', 'runtime', 'application/wasm'],
    ['/runtimes/centipawn-tvmjs-webgpu/model/f32/v2/tvmjs_runtime.wasm', 'runtime', 'application/wasm'],
    ['/runtimes/centipawn-tvmjs-webgpu/model/f32/v2/tvmjs.bundle.js', 'runtime', 'text/javascript; charset=utf-8'],
  ]);
});

test('write_artifact_release_manifests carries forward an immutable v2 base release without local artifact files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-carry-forward-'));
  await mkdir(join(root, 'public/stormphrax'), { recursive: true });
  await writeFile(join(root, 'public/stormphrax/engine.wasm'), 'abc');
  await writeJson(join(root, 'public/stormphrax/manifest.json'), {
    artifacts: [{ path: 'public/stormphrax/engine.wasm', bytes: 3, sha256: ABC_SHA256 }],
  });
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'base',
    sourceManifests: ['public/legacy/manifest.json'],
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      raw: { sha256: ABC_SHA256, bytes: 3 },
      representations: [{
        encoding: 'identity',
        url: `https://assets.example/artifacts/sha256/${ABC_SHA256}/identity`,
        sha256: ABC_SHA256,
        bytes: 3,
      }],
      file: 'legacy.wasm',
      kind: 'engine',
      contentType: 'application/wasm',
      sourceManifest: 'public/legacy/manifest.json',
      localPath: 'public/legacy/engine.wasm',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'public/stormphrax/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const release = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json'), 'utf8'));
  assert.equal(release.schema, 'lc0_browser.artifact_release_manifest.v2');
  assert.equal(release.baseReleaseId, 'base');
  assert.deepEqual(release.artifacts.map((artifact) => artifact.logicalUrl), [
    '/legacy/engine.wasm',
    '/stormphrax/engine.wasm',
    '/stormphrax/manifest.json',
  ]);
  assert.equal(release.artifacts[0].carriedForwardFrom, 'base');
  assert.equal(release.artifacts[0].representations[0].url, `https://assets.example/artifacts/sha256/${ABC_SHA256}/identity`);
  assert.equal(release.artifacts[1].carriedForwardFrom, undefined);
  assert.equal(release.artifacts[1].representations[0].url.endsWith(`/${ABC_SHA256}/identity`), true);
  assert.equal(release.artifacts.filter((artifact) => artifact.kind === 'manifest').length, 1);
});

test('write_artifact_release_manifests migrates an actual v1-shaped base release into deduplicated v2 representations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-v1-carry-forward-'));
  await mkdir(join(root, 'public/legacy'), { recursive: true });
  await writeFile(join(root, 'public/legacy/engine.wasm'), 'abc');
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    generatedAt: '2026-07-14T00:00:00.000Z',
    channel: 'stable',
    sourceManifests: ['public/legacy/manifest.json'],
    artifacts: [
      {
        logicalUrl: '/legacy/engine.wasm',
        artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
        sha256: ABC_SHA256,
        bytes: 3,
        file: 'engine.wasm',
        kind: 'engine',
        contentType: 'application/wasm',
        sourceManifest: 'public/legacy/manifest.json',
        localPath: 'public/legacy/engine.wasm',
      },
      {
        logicalUrl: '/legacy/engine-copy.wasm',
        artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine-copy.wasm`,
        sha256: ABC_SHA256,
        bytes: 3,
        file: 'engine-copy.wasm',
        kind: 'engine',
        contentType: 'application/wasm',
        sourceManifest: 'public/legacy/manifest.json',
        localPath: 'public/legacy/missing-copy.wasm',
      },
    ],
  });
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const releasePath = join(outputRoot, 'releases/next.json');
  const release = JSON.parse(await readFile(releasePath, 'utf8'));
  assert.equal(release.schema, 'lc0_browser.artifact_release_manifest.v2');
  assert.equal(release.baseReleaseId, 'base');
  assert.deepEqual(release.sourceManifests, ['public/legacy/manifest.json']);
  assert.equal(release.artifacts.length, 2);
  assert.ok(release.artifacts.every((artifact) => artifact.carriedForwardFrom === 'base'));
  assert.ok(release.artifacts.every((artifact) => artifact.artifactUrl === undefined));
  assert.ok(release.artifacts.every((artifact) => artifact.representations.filter((entry) => entry.encoding === 'identity').length === 1));
  assert.equal(release.artifacts[0].representations[0].url, release.artifacts[1].representations[0].url);
  assert.equal(release.artifacts[0].representations[1].url, release.artifacts[1].representations[1].url);
  assert.deepEqual(release.artifacts.map((artifact) => artifact.migrationSource.url).sort(), [
    `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine-copy.wasm`,
    `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
  ]);

  const publish = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(publish.status, 0, publish.stderr);
  const plan = JSON.parse(publish.stdout);
  assert.equal(plan.catalogObjectCount, 2);
  assert.equal(plan.plannedCount, 2);
  assert.ok(plan.planned.some((entry) => entry.key === `artifacts/sha256/${ABC_SHA256}/identity`
    && entry.localPath === join(outputRoot, 'artifacts/sha256', ABC_SHA256, 'identity')));
});

test('write_artifact_release_manifests preserves verified v1 provenance when a body is unavailable locally', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-v1-missing-body-'));
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
      localPath: 'public/legacy/engine.wasm',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const release = JSON.parse(await readFile(join(outputRoot, 'releases/next.json'), 'utf8'));
  assert.equal(release.artifacts.length, 1);
  const [artifact] = release.artifacts;
  assert.equal(artifact.artifactUrl, undefined);
  assert.equal(artifact.localPath, undefined);
  assert.deepEqual(artifact.migrationSource, {
    schema: 'lc0_browser.artifact_migration_source.v1',
    releaseId: 'base',
    key: `artifacts/sha256/${ABC_SHA256}/engine.wasm`,
    url: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
  });
  assert.deepEqual(artifact.representations, [{
    encoding: 'identity',
    url: `https://assets.0x88.app/artifacts/sha256/${ABC_SHA256}/identity`,
    sha256: ABC_SHA256,
    bytes: 3,
  }]);

  const dryRun = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', join(outputRoot, 'releases/next.json'),
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(dryRun.status, 0, dryRun.stderr);
  const plan = JSON.parse(dryRun.stdout);
  assert.equal(plan.plannedCount, 1);
  assert.equal(plan.planned[0].key, `artifacts/sha256/${ABC_SHA256}/identity`);
  assert.equal(plan.planned[0].uploadAction, 'materialize-from-v1-and-conditional-create');
  assert.deepEqual(plan.planned[0].migrationSources, [{
    releaseId: 'base',
    key: `artifacts/sha256/${ABC_SHA256}/engine.wasm`,
    url: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
  }]);
  assert.equal(existsSync(join(outputRoot, 'artifacts/sha256', ABC_SHA256, 'identity')), false);
});

test('write_artifact_release_manifests ignores a stale v1 localPath after a later candidate verifies', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-v1-stale-local-path-'));
  await mkdir(join(root, 'public/legacy'), { recursive: true });
  await mkdir(join(root, 'stale'), { recursive: true });
  await writeFile(join(root, 'stale/engine.wasm'), 'abd');
  await writeFile(join(root, 'public/legacy/engine.wasm'), 'abc');
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
      localPath: 'stale/engine.wasm',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const release = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json'), 'utf8'));
  assert.equal(release.artifacts[0].localPath, 'public/legacy/engine.wasm');
});

test('write_artifact_release_manifests accepts a verified equal local body after a stale v1 localPath', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-v1-equal-local-body-'));
  await mkdir(join(root, 'public/current'), { recursive: true });
  await mkdir(join(root, 'public/legacy'), { recursive: true });
  await writeFile(join(root, 'public/current/model.onnx'), 'abc');
  await writeFile(join(root, 'public/legacy/engine.wasm'), 'abd');
  await writeJson(join(root, 'public/current/manifest.json'), {
    models: [{
      file: 'model.onnx',
      url: '/current/model.onnx',
      bytes: 3,
      sha256: ABC_SHA256,
    }],
  });
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
      localPath: 'public/legacy/engine.wasm',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'public/current/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const release = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json'), 'utf8'));
  const migrated = release.artifacts.find((artifact) => artifact.logicalUrl === '/legacy/engine.wasm');
  assert.equal(migrated.localPath, 'public/current/model.onnx');
});

test('write_artifact_release_manifests rejects v1 migration bodies when every available candidate is corrupt', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-v1-corrupt-body-'));
  await mkdir(join(root, 'public/legacy'), { recursive: true });
  await mkdir(join(root, 'stale'), { recursive: true });
  await writeFile(join(root, 'stale/engine.wasm'), 'abe');
  await writeFile(join(root, 'public/legacy/engine.wasm'), 'abd');
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
      localPath: 'stale/engine.wasm',
    }],
  });
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Corrupt local v1 migration source/);
  assert.match(result.stderr, /stale\/engine\.wasm has 3\//);
  assert.match(result.stderr, /public\/legacy\/engine\.wasm has 3\//);
  assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'releases/next.json')), false);
  assert.equal(existsSync(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json')), false);
});

test('publish_hashed_artifacts_to_r2 materializes a missing SHA-only identity from authoritative v1 R2 content', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-v1-migration-'));
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
      contentType: 'application/wasm',
    }],
  });
  const generated = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(generated.status, 0, generated.stderr);

  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const r2Root = join(root, 'mock-r2');
  const wrangler = await writeStatefulWrangler(root);
  const aws = await writeAtomicAws(root);
  const legacyTarget = `test-bucket/artifacts/sha256/${ABC_SHA256}/engine.wasm`;
  const identityTarget = `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`;
  await mkdir(r2Root, { recursive: true });
  await writeFile(mockR2ObjectPath(r2Root, legacyTarget), 'abc');
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', join(outputRoot, 'releases/next.json'),
      '--channel-manifest', join(outputRoot, 'channels/stable.json'),
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, MOCK_R2_DIR: r2Root } });
    assert.equal(result.status, 0, result.stderr);
    const plan = JSON.parse(result.stdout);
    assert.equal(plan.plannedCount, 1);
    assert.equal(plan.planned[0].uploadAction, 'migrated-and-uploaded');
    assert.equal(plan.planned[0].migrationSourceUsed.key, `artifacts/sha256/${ABC_SHA256}/engine.wasm`);
    assert.equal((await readFile(mockR2ObjectPath(r2Root, identityTarget))).toString(), 'abc');
    assert.equal((await readFile(mockR2ObjectPath(r2Root, legacyTarget))).toString(), 'abc');
    assert.equal(existsSync(mockR2ObjectPath(r2Root, 'test-bucket/releases/next.json')), true);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, 'test-bucket/channels/stable.json')), true);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 blocks migration when the v1 body is unavailable locally and remotely', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-v1-migration-missing-'));
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
    }],
  });
  const generated = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(generated.status, 0, generated.stderr);

  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const r2Root = join(root, 'mock-r2');
  const wrangler = await writeStatefulWrangler(root);
  const aws = await writeAtomicAws(root);
  await mkdir(r2Root, { recursive: true });
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', join(outputRoot, 'releases/next.json'),
      '--channel-manifest', join(outputRoot, 'channels/stable.json'),
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, MOCK_R2_DIR: r2Root } });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /V1 migration body is unavailable at R2 key/);
    assert.match(result.stderr, /Restore the legacy object or provide matching local decoded bytes/);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`)), false);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, 'test-bucket/releases/next.json')), false);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, 'test-bucket/channels/stable.json')), false);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects corrupt v1 migration content before identity, release, or channel mutation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-v1-migration-corrupt-'));
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/engine.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
    }],
  });
  const generated = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'next',
    '--base-release', 'public/releases/base.json',
    '--manifest', 'missing.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(generated.status, 0, generated.stderr);

  const outputRoot = join(root, DEFAULT_RELEASE_OUTPUT);
  const r2Root = join(root, 'mock-r2');
  const wrangler = await writeStatefulWrangler(root);
  const aws = await writeAtomicAws(root);
  const legacyTarget = `test-bucket/artifacts/sha256/${ABC_SHA256}/engine.wasm`;
  await mkdir(r2Root, { recursive: true });
  await writeFile(mockR2ObjectPath(r2Root, legacyTarget), 'abd');
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', join(outputRoot, 'releases/next.json'),
      '--channel-manifest', join(outputRoot, 'channels/stable.json'),
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, MOCK_R2_DIR: r2Root } });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Corrupt v1 migration source/);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`)), false);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, 'test-bucket/releases/next.json')), false);
    assert.equal(existsSync(mockR2ObjectPath(r2Root, 'test-bucket/channels/stable.json')), false);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('write_artifact_release_manifests keeps releases write-once while channels remain mutable for rollback', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-write-once-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  await writeJson(join(root, 'public/models/lc0/manifest.json'), {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
  const generate = (releaseId, generatedAt) => spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', releaseId,
    '--generated-at', generatedAt,
    '--manifest', 'public/models/lc0/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });

  assert.equal(generate('one', '2026-07-14T00:00:00.000Z').status, 0);
  assert.equal(generate('two', '2026-07-14T00:01:00.000Z').status, 0);
  let channel = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json'), 'utf8'));
  assert.equal(channel.releaseId, 'two');

  const rollback = generate('one', '2026-07-14T00:00:00.000Z');
  assert.equal(rollback.status, 0, rollback.stderr);
  channel = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json'), 'utf8'));
  assert.equal(channel.releaseId, 'one');

  const overwrite = generate('one', '2026-07-14T00:02:00.000Z');
  assert.notEqual(overwrite.status, 0);
  assert.match(overwrite.stderr, /Refusing to overwrite immutable release manifest/);
  channel = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json'), 'utf8'));
  assert.equal(channel.releaseId, 'one');
});

test('write_artifact_release_manifests reuses an existing release timestamp for default retries', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-default-retry-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  await writeJson(join(root, 'public/models/lc0/manifest.json'), {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
  const generate = () => spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--manifest', 'public/models/lc0/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });

  const first = generate();
  assert.equal(first.status, 0, first.stderr);
  const releaseDirectory = join(root, DEFAULT_RELEASE_OUTPUT, 'releases');
  const [releaseFilename] = await readdir(releaseDirectory);
  const defaultReleaseId = releaseFilename.replace(/\.json$/, '');
  const releasePath = join(releaseDirectory, releaseFilename);
  const original = await readFile(releasePath, 'utf8');
  const originalRelease = JSON.parse(original);
  assert.equal(originalRelease.releaseId, defaultReleaseId);
  assert.match(defaultReleaseId, /^\d{4}-\d{2}-\d{2}\.nogit$/);

  await new Promise((resolve) => setTimeout(resolve, 10));
  const retry = generate();
  assert.equal(retry.status, 0, retry.stderr);
  assert.equal(await readFile(releasePath, 'utf8'), original);
  const channel = JSON.parse(await readFile(join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json'), 'utf8'));
  assert.equal(channel.generatedAt, originalRelease.generatedAt);
});

test('write_artifact_release_manifests rejects mismatched default retries without moving the channel', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-default-retry-mismatch-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  const assetPath = join(root, 'public/models/lc0/test.onnx');
  const manifestPath = join(root, 'public/models/lc0/manifest.json');
  await writeFile(assetPath, 'abc');
  await writeJson(manifestPath, {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
  const generate = (releaseId) => spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', releaseId,
    '--manifest', 'public/models/lc0/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });

  assert.equal(generate('accepted').status, 0);
  assert.equal(generate('conflict').status, 0);
  const channelPath = join(root, DEFAULT_RELEASE_OUTPUT, 'channels/stable.json');
  const acceptedChannel = JSON.parse(await readFile(channelPath, 'utf8'));
  assert.equal(acceptedChannel.releaseId, 'conflict');

  const differentSha = createHash('sha256').update('abd').digest('hex');
  await writeFile(assetPath, 'abd');
  await writeJson(manifestPath, {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: differentSha }],
  });
  const mismatch = generate('conflict');
  assert.notEqual(mismatch.status, 0);
  assert.match(mismatch.stderr, /Refusing to overwrite immutable release manifest/);
  assert.equal(JSON.parse(await readFile(channelPath, 'utf8')).releaseId, 'conflict');

  const releasePath = join(root, DEFAULT_RELEASE_OUTPUT, 'releases/conflict.json');
  const tampered = JSON.parse(await readFile(releasePath, 'utf8'));
  tampered.channel = 'canary';
  await writeJson(releasePath, tampered);
  const identityMismatch = generate('conflict');
  assert.notEqual(identityMismatch.status, 0);
  assert.match(identityMismatch.stderr, /Existing immutable release identity does not match requested release/);
  assert.equal(JSON.parse(await readFile(channelPath, 'utf8')).releaseId, 'conflict');
});

test('write_artifact_release_manifests rejects unsafe release and channel names', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-path-safety-'));
  const run = (...args) => spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--manifest', 'missing.json',
    ...args,
  ], { cwd: process.cwd(), encoding: 'utf8' });

  const release = run('--release-id', '../outside');
  assert.notEqual(release.status, 0);
  assert.match(release.stderr, /--release-id must contain only/);

  const channel = run('--release-id', 'safe', '--channel', '../outside');
  assert.notEqual(channel.status, 0);
  assert.match(channel.stderr, /--channel must contain only/);
});

test('write_artifact_release_manifests rejects corrupt existing SHA-only bodies', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-corrupt-body-'));
  await mkdir(join(root, 'public/models/lc0'), { recursive: true });
  await writeFile(join(root, 'public/models/lc0/test.onnx'), 'abc');
  await writeJson(join(root, 'public/models/lc0/manifest.json'), {
    models: [{ file: 'test.onnx', url: '/models/lc0/test.onnx', bytes: 3, sha256: ABC_SHA256 }],
  });
  const identity = join(root, DEFAULT_RELEASE_OUTPUT, 'artifacts/sha256', ABC_SHA256, 'identity');
  await mkdir(dirname(identity), { recursive: true });
  await writeFile(identity, 'abd');
  const result = spawnSync(process.execPath, [
    'scripts/write_artifact_release_manifests.mjs',
    '--root', root,
    '--release-id', 'corrupt',
    '--manifest', 'public/models/lc0/manifest.json',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Content-addressed key collision or corruption/);
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
    ['release-manifest', 'releases/test-release.json', 'public, max-age=31536000, immutable'],
    ['channel-manifest', 'channels/stable.json', 'no-cache'],
  ]);
});

test('publish_hashed_artifacts_to_r2 rejects a channel manifest whose filename disagrees with channel.channel', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-channel-name-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'test-release.json');
  const channelPath = join(root, 'canary.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'test-release',
    artifacts: [{
      logicalUrl: '/model.onnx',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/model.onnx`,
      sha256: ABC_SHA256,
      bytes: 3,
      localPath: 'model.onnx',
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

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /filename canary\.json does not match channel\.channel stable/);
});

test('publish_hashed_artifacts_to_r2 rejects v2 artifacts without exactly one identity representation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-v2-identity-contract-'));
  const releasePath = join(root, 'release.json');
  const identity = {
    encoding: 'identity',
    url: `/artifacts/sha256/${ABC_SHA256}/identity`,
    sha256: ABC_SHA256,
    bytes: 3,
  };
  const run = () => spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });

  for (const [artifact, expected] of [
    [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `/artifacts/sha256/${ABC_SHA256}/legacy.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
    }, /V2 artifact has no representations/],
    [{
      logicalUrl: '/models/model.onnx',
      raw: { sha256: ABC_SHA256, bytes: 3 },
      representations: [],
    }, /V2 artifact has no representations/],
    [{
      logicalUrl: '/models/model.onnx',
      raw: { sha256: ABC_SHA256, bytes: 3 },
      representations: [identity, { ...identity }],
    }, /must have exactly one identity representation.*found 2/],
  ]) {
    await writeJson(releasePath, {
      schema: 'lc0_browser.artifact_release_manifest.v2',
      releaseId: 'invalid-identity',
      artifacts: [artifact],
    });
    const result = run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, expected);
  }
});

test('publish_hashed_artifacts_to_r2 plans deduplicated v2 identity and Brotli representations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-v2-'));
  const sourceA = join(root, 'a.onnx');
  const sourceB = join(root, 'b.onnx');
  await writeFile(sourceA, 'abc');
  await writeFile(sourceB, 'abc');
  const materialize = spawnSync(process.execPath, [
    'scripts/publish_content_addressed_release.mjs',
    '--root', root,
    '--release-id', 'v2-release',
    '--channel', 'stable',
    '--asset', `model-a=${sourceA}`,
    '--asset', `model-b=${sourceB}`,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(materialize.status, 0, materialize.stderr);

  const releasePath = join(root, 'releases/v2-release.json');
  const channelPath = join(root, 'channels/stable.json');
  const result = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--channel-manifest', channelPath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const parsed = JSON.parse(result.stdout);
  assert.equal(parsed.releaseSchema, 'lc0-webgpu.artifact-release.v2');
  assert.equal(parsed.plannedCount, 2, 'equal logical assets share identity and Brotli objects');
  const identity = parsed.planned.find((item) => item.contentEncoding === undefined);
  const br = parsed.planned.find((item) => item.contentEncoding === 'br');
  assert.equal(identity.key, `artifacts/sha256/${ABC_SHA256}/identity`);
  assert.deepEqual(identity.logicalUrls.sort(), ['model-a', 'model-b']);
  assert.match(br.key, new RegExp(`^artifacts/sha256/${ABC_SHA256}/br/[a-f0-9]{64}$`));
  assert.deepEqual(br.logicalUrls.sort(), ['model-a', 'model-b']);
  assert.equal(parsed.manifests[0].cacheControl, 'public, max-age=31536000, immutable');
});

test('publish_hashed_artifacts_to_r2 uses any local duplicate when an inherited v2 alias appears first', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-v2-inherited-dedupe-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
  const identity = { encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 };
  await writeJson(releasePath, {
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'v2-inherited-dedupe',
    artifacts: [
      { name: 'inherited', carriedForwardFrom: 'base', raw: { bytes: 3, sha256: ABC_SHA256 }, representations: [identity] },
      { name: 'local', localPath: 'model.onnx', raw: { bytes: 3, sha256: ABC_SHA256 }, representations: [identity] },
      { name: 'missing-later-alias', localPath: 'missing.onnx', raw: { bytes: 3, sha256: ABC_SHA256 }, representations: [identity] },
    ],
  });
  const result = spawnSync(process.execPath, [
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--bucket', 'test-bucket',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const parsed = JSON.parse(result.stdout);
  assert.equal(parsed.plannedCount, 1);
  assert.deepEqual(parsed.planned[0].logicalUrls, ['inherited', 'local', 'missing-later-alias']);
  assert.equal(parsed.planned[0].remoteState, 'not-probed');
  assert.equal(parsed.planned[0].localPath, join(root, 'model.onnx'));
});

test('publish_hashed_artifacts_to_r2 rejects same-length corrupt v2 remote bodies', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-v2-corrupt-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
  await writeJson(releasePath, {
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'v2-corrupt',
    artifacts: [{
      name: 'model',
      localPath: 'model.onnx',
      raw: { bytes: 3, sha256: ABC_SHA256 },
      representations: [{ encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 }],
    }],
  });
  const server = createServer((req, res) => {
    const headers = {
      'Content-Length': '3',
      'X-Artifact-Content-Length': '3',
      'Cache-Control': 'public, max-age=31536000, immutable',
    };
    if (req.method === 'HEAD') res.writeHead(200, headers).end();
    else res.writeHead(200, headers).end('abd');
  });
  const port = await listen(server);
  try {
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--probe-existing',
    ]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Remote artifact SHA-256 mismatch/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 trusts encoded-length metadata when identity and Brotli HEAD Content-Length is normalized to zero', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-v2-normalized-head-'));
  const identityBody = Buffer.from('abc');
  const brBody = brotliCompressSync(identityBody);
  const brSha256 = createHash('sha256').update(brBody).digest('hex');
  await writeFile(join(root, 'model.onnx'), identityBody);
  await writeFile(join(root, 'model.onnx.br'), brBody);
  const releasePath = join(root, 'release.json');
  await writeJson(releasePath, {
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'v2-normalized-head',
    artifacts: [{
      name: 'model',
      raw: { bytes: identityBody.byteLength, sha256: ABC_SHA256 },
      representations: [
        {
          encoding: 'identity',
          url: `/artifacts/sha256/${ABC_SHA256}/identity`,
          localPath: 'model.onnx',
          bytes: identityBody.byteLength,
          sha256: ABC_SHA256,
        },
        {
          encoding: 'br',
          url: `/artifacts/sha256/${ABC_SHA256}/br/${brSha256}`,
          localPath: 'model.onnx.br',
          bytes: brBody.byteLength,
          sha256: brSha256,
        },
      ],
    }],
  });
  const server = createServer((req, res) => {
    const isBr = req.url?.endsWith(`/br/${brSha256}`);
    const body = isBr ? brBody : identityBody;
    const headers = {
      'Content-Length': req.method === 'HEAD' ? '0' : String(body.byteLength),
      'X-Artifact-Content-Length': String(identityBody.byteLength),
      'X-Artifact-Encoded-Length': String(body.byteLength),
      'Cache-Control': 'public, max-age=31536000, immutable',
      ...(isBr ? { 'Content-Encoding': 'br' } : {}),
    };
    res.writeHead(200, headers);
    res.end(req.method === 'HEAD' ? undefined : body);
  });
  const port = await listen(server);
  try {
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--probe-existing',
    ]);
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.plannedCount, 2);
    assert.ok(parsed.planned.every((entry) => entry.remoteState === 'existing'));
    assert.equal(parsed.planned.find((entry) => !entry.contentEncoding).remoteProbe.bytes, identityBody.byteLength);
    assert.equal(parsed.planned.find((entry) => entry.contentEncoding === 'br').remoteProbe.bytes, brBody.byteLength);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 uploads v2 Brotli bodies with Content-Encoding', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-v2-execute-'));
  const source = join(root, 'model.onnx');
  await writeFile(source, 'abc');
  const materialize = spawnSync(process.execPath, [
    'scripts/publish_content_addressed_release.mjs',
    '--root', root,
    '--release-id', 'v2-execute',
    '--asset', `model=${source}`,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(materialize.status, 0, materialize.stderr);

  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const logPath = join(root, 'wrangler.log');
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', join(root, 'releases/v2-execute.json'),
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });
    assert.equal(result.status, 0, result.stderr);
    const log = await readFile(logPath, 'utf8');
    assert.match(log, new RegExp(`s3api put-object --bucket test-bucket --key artifacts/sha256/${ABC_SHA256}/identity --body .* --content-type application/octet-stream --cache-control public, max-age=31536000, immutable --if-none-match \\* --endpoint-url https://r2\\.invalid --region auto`));
    assert.match(log, new RegExp(`s3api put-object --bucket test-bucket --key artifacts/sha256/${ABC_SHA256}/br/[a-f0-9]{64} --body .* --content-type application/octet-stream --cache-control public, max-age=31536000, immutable --if-none-match \\* --endpoint-url https://r2\\.invalid --region auto --content-encoding br`));
    assert.match(log, /s3api put-object --bucket test-bucket --key releases\/v2-execute\.json .*--content-type application\/json; charset=utf-8 --cache-control public, max-age=31536000, immutable --if-none-match \* .*--endpoint-url https:\/\/r2\.invalid --region auto/);
    assert.doesNotMatch(log, /r2 object put test-bucket\/artifacts\//);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 refuses a warm valid CDN object when authoritative R2 content differs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-warm-cdn-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
  await writeJson(releasePath, {
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'stale-cdn',
    artifacts: [{
      name: 'model',
      localPath: 'model.onnx',
      raw: { bytes: 3, sha256: ABC_SHA256 },
      representations: [{ encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 }],
    }],
  });
  const server = createServer((req, res) => {
    const headers = {
      'Content-Length': '3',
      'X-Artifact-Content-Length': '3',
      'Cache-Control': 'public, max-age=31536000, immutable',
    };
    if (req.method === 'HEAD') res.writeHead(200, headers).end();
    else res.writeHead(200, headers).end('abc');
  });
  const port = await listen(server);
  try {
    const logPath = join(root, 'wrangler.log');
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const target = `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`;
    await writeMockR2Object(r2Root, target, 'abd', {
      ContentLength: 3,
      ContentType: 'application/octet-stream',
      CacheControl: 'public, max-age=31536000, immutable',
    });

    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Refusing to overwrite immutable object .*remote content differs/);
    const log = await readFile(logPath, 'utf8');
    assert.match(log, new RegExp(`r2 object get ${target}`));
    assert.doesNotMatch(log, new RegExp(`r2 object put ${target}`));
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects a differing artifact after an atomic create conflict and never updates the channel', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-create-race-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
  const channelPath = join(root, 'stable.json');
  await writeJson(releasePath, {
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'create-race',
    artifacts: [{
      name: 'model',
      localPath: 'model.onnx',
      raw: { bytes: 3, sha256: ABC_SHA256 },
      representations: [{ encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 }],
    }],
  });
  await writeJson(channelPath, {
    schema: 'lc0_browser.artifact_channel_manifest.v2',
    channel: 'stable',
    releaseId: 'create-race',
    releaseManifestUrl: '/releases/create-race.json',
    releaseUrl: '/releases/create-race.json',
  });
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const logPath = join(root, 'wrangler.log');
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const target = `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`;
    await writeMockR2Object(r2Root, target, 'abd', {
      ContentLength: 3,
      ContentType: 'application/octet-stream',
      CacheControl: 'public, max-age=31536000, immutable',
    });

    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--channel-manifest', channelPath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Refusing to overwrite immutable object .*remote content differs/);
    const log = await readFile(logPath, 'utf8');
    assert.match(log, new RegExp(`s3api put-object --bucket test-bucket --key artifacts/sha256/${ABC_SHA256}/identity .*--if-none-match \\*`));
    assert.match(log, /r2 object get .*identity/);
    assert.doesNotMatch(log, /r2 object put .*identity/);
    assert.doesNotMatch(log, /releases\/create-race\.json/);
    assert.doesNotMatch(log, /r2 object put test-bucket\/channels\/stable\.json/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects Brotli conflicts with missing or wrong authoritative Content-Encoding', async () => {
  for (const contentEncoding of [undefined, 'gzip']) {
    const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-br-metadata-race-'));
    const source = join(root, 'model.onnx');
    await writeFile(source, 'abc');
    const materialize = spawnSync(process.execPath, [
      'scripts/publish_content_addressed_release.mjs',
      '--root', root,
      '--release-id', 'br-metadata-race',
      '--channel', 'stable',
      '--asset', `model=${source}`,
    ], { cwd: process.cwd(), encoding: 'utf8' });
    assert.equal(materialize.status, 0, materialize.stderr);
    const releasePath = join(root, 'releases/br-metadata-race.json');
    const channelPath = join(root, 'channels/stable.json');
    const release = JSON.parse(await readFile(releasePath, 'utf8'));
    const br = release.artifacts[0].representations.find((entry) => entry.encoding === 'br');
    const brKey = new URL(br.url, 'https://assets.invalid').pathname.replace(/^\/+/, '');

    const server = createServer((_req, res) => res.writeHead(404).end());
    const port = await listen(server);
    try {
      const logPath = join(root, 'publish.log');
      const r2Root = join(root, 'mock-r2');
      const wrangler = await writeStatefulWrangler(root);
      const aws = await writeAtomicAws(root);
      await writeMockR2Object(r2Root, `test-bucket/${brKey}`, await readFile(join(root, brKey)), {
        ContentLength: br.bytes,
        ContentType: 'application/octet-stream',
        CacheControl: 'public, max-age=31536000, immutable',
        ...(contentEncoding ? { ContentEncoding: contentEncoding } : {}),
      });

      const result = await runNode([
        'scripts/publish_hashed_artifacts_to_r2.mjs',
        '--root', root,
        '--release', releasePath,
        '--channel-manifest', channelPath,
        '--bucket', 'test-bucket',
        '--artifact-base', `http://127.0.0.1:${port}`,
        '--execute',
        '--wrangler-bin', wrangler,
        '--aws-bin', aws,
        '--r2-endpoint', 'https://r2.invalid',
      ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });

      assert.notEqual(result.status, 0);
      assert.match(result.stderr, new RegExp(`Content-Encoding is ${contentEncoding ?? 'missing'}, expected br`));
      const log = await readFile(logPath, 'utf8');
      assert.match(log, new RegExp(`s3api put-object --bucket test-bucket --key ${brKey.replaceAll('/', '\\/')} .*--if-none-match \\*`));
      assert.match(log, new RegExp(`s3api head-object --bucket test-bucket --key ${brKey.replaceAll('/', '\\/')}`));
      assert.doesNotMatch(log, /releases\/br-metadata-race\.json/);
      assert.doesNotMatch(log, /r2 object put test-bucket\/channels\/stable\.json/);
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }
  }
});

test('publish_hashed_artifacts_to_r2 rejects an identity conflict with wrong authoritative Content-Type', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-identity-metadata-race-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
  const channelPath = join(root, 'stable.json');
  await writeJson(releasePath, {
    schema: 'lc0-webgpu.artifact-release.v2',
    releaseId: 'identity-metadata-race',
    artifacts: [{
      name: 'model',
      localPath: 'model.onnx',
      contentType: 'application/octet-stream',
      raw: { bytes: 3, sha256: ABC_SHA256 },
      representations: [{ encoding: 'identity', url: `/artifacts/sha256/${ABC_SHA256}/identity`, bytes: 3, sha256: ABC_SHA256 }],
    }],
  });
  await writeJson(channelPath, {
    schema: 'lc0_browser.artifact_channel_manifest.v2',
    channel: 'stable',
    releaseId: 'identity-metadata-race',
    releaseManifestUrl: '/releases/identity-metadata-race.json',
    releaseUrl: '/releases/identity-metadata-race.json',
  });
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const logPath = join(root, 'publish.log');
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const target = `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`;
    await writeMockR2Object(r2Root, target, 'abc', {
      ContentLength: 3,
      ContentType: 'text/plain',
      CacheControl: 'public, max-age=31536000, immutable',
    });

    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--channel-manifest', channelPath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Content-Type is text\/plain, expected application\/octet-stream/);
    const log = await readFile(logPath, 'utf8');
    assert.doesNotMatch(log, /releases\/identity-metadata-race\.json/);
    assert.doesNotMatch(log, /r2 object put test-bucket\/channels\/stable\.json/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 rejects an identical release conflict with wrong authoritative metadata', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-release-metadata-race-'));
  const releasePath = join(root, 'release.json');
  const channelPath = join(root, 'stable.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'release-metadata-race',
    artifacts: [],
  });
  await writeJson(channelPath, {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    channel: 'stable',
    releaseId: 'release-metadata-race',
    releaseManifestUrl: '/releases/release-metadata-race.json',
  });
  const r2Root = join(root, 'mock-r2');
  const logPath = join(root, 'publish.log');
  const wrangler = await writeStatefulWrangler(root);
  const aws = await writeAtomicAws(root);
  const releaseBody = await readFile(releasePath);
  await writeMockR2Object(r2Root, 'test-bucket/releases/release-metadata-race.json', releaseBody, {
    ContentLength: releaseBody.byteLength,
    ContentType: 'application/json; charset=utf-8',
    CacheControl: 'no-cache',
  });

  const result = await runNode([
    'scripts/publish_hashed_artifacts_to_r2.mjs',
    '--root', root,
    '--release', releasePath,
    '--channel-manifest', channelPath,
    '--bucket', 'test-bucket',
    '--execute',
    '--wrangler-bin', wrangler,
    '--aws-bin', aws,
    '--r2-endpoint', 'https://r2.invalid',
  ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Cache-Control is no-cache, expected public, max-age=31536000, immutable/);
  const log = await readFile(logPath, 'utf8');
  assert.match(log, /s3api put-object --bucket test-bucket --key releases\/release-metadata-race\.json .*--if-none-match \*/);
  assert.match(log, /s3api head-object --bucket test-bucket --key releases\/release-metadata-race\.json/);
  assert.doesNotMatch(log, /r2 object put test-bucket\/channels\/stable\.json/);
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
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const target = `test-bucket/artifacts/sha256/${ABC_SHA256}/test.onnx`;
    await writeMockR2Object(r2Root, target, 'abc', {
      ContentLength: 3,
      ContentType: 'application/octet-stream',
      CacheControl: 'public, max-age=31536000, immutable',
    });
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.planned[0].remoteState, 'identical-r2');
    assert.equal(parsed.planned[0].uploadAction, 'skip-identical-r2');
    const log = await readFile(logPath, 'utf8');
    assert.match(log, new RegExp(`r2 object get ${target}`));
    assert.doesNotMatch(log, new RegExp(`r2 object put ${target}`));
    assert.match(log, /s3api put-object --bucket test-bucket --key releases\/test-release\.json .*--if-none-match \*/);
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
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const releaseTarget = 'test-bucket/releases/test-release.json';
    const differingReleaseBody = '{"different":true}\n';
    await writeMockR2Object(r2Root, releaseTarget, differingReleaseBody, {
      ContentLength: Buffer.byteLength(differingReleaseBody),
      ContentType: 'application/json; charset=utf-8',
      CacheControl: 'public, max-age=31536000, immutable',
    });
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, MOCK_R2_DIR: r2Root } });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Refusing to accept immutable object .*releases\/test-release\.json/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 fails closed when artifact existence cannot be checked authoritatively', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-release-check-failure-'));
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
      localPath: 'public/models/lc0/test.onnx',
    }],
  });
  const server = createServer((req, res) => {
    const headers = {
      'X-Artifact-Content-Length': '3',
      'Cache-Control': 'public, max-age=31536000, immutable',
    };
    if (req.method === 'HEAD') res.writeHead(200, headers).end();
    else res.writeHead(200, { ...headers, 'Content-Length': '3' }).end('abc');
  });
  const port = await listen(server);
  try {
    const wrangler = join(root, 'fake-wrangler.sh');
    const aws = join(root, 'fake-aws.sh');
    await writeFile(wrangler, '#!/bin/sh\nif [ "$1 $2 $3" = "r2 object get" ]; then echo "authentication failed" >&2; exit 1; fi\nexit 0\n');
    await writeFile(aws, '#!/bin/sh\necho "PreconditionFailed: status code: 412" >&2\nexit 1\n');
    await chmod(wrangler, 0o755);
    await chmod(aws, 0o755);
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, new RegExp(`Unable to verify immutable object .*artifacts/sha256/${ABC_SHA256}/test\\.onnx`));
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 treats an identical remote release manifest as idempotent', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-identical-release-'));
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
      localPath: 'public/models/lc0/test.onnx',
    }],
  });
  await writeJson(channelPath, {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    channel: 'stable',
    releaseId: 'test-release',
    releaseManifestUrl: '/releases/test-release.json',
  });
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const logPath = join(root, 'wrangler.log');
    const r2Root = join(root, 'mock-r2');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const releaseTarget = 'test-bucket/releases/test-release.json';
    const releaseBody = await readFile(releasePath);
    await writeMockR2Object(r2Root, releaseTarget, releaseBody, {
      ContentLength: releaseBody.byteLength,
      ContentType: 'application/json; charset=utf-8',
      CacheControl: 'public, max-age=31536000, immutable',
    });
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--channel-manifest', channelPath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    const releaseItem = parsed.manifests.find((item) => item.type === 'release-manifest');
    assert.equal(releaseItem.remoteState, 'identical');
    assert.equal(releaseItem.uploadAction, 'skip-identical');
    const log = await readFile(logPath, 'utf8');
    assert.doesNotMatch(log, /r2 object put test-bucket\/releases\/test-release\.json/);
    assert.match(log, /s3api put-object --bucket test-bucket --key releases\/test-release\.json .*--if-none-match \*/);
    assert.match(log, /r2 object put test-bucket\/channels\/stable\.json/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('publish_hashed_artifacts_to_r2 atomically admits only one competing release body and only its channel', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-atomic-race-'));
  const r2Root = join(root, 'mock-r2');
  const logPath = join(root, 'publish.log');
  const wrangler = await writeStatefulWrangler(root);
  const aws = await writeAtomicAws(root);
  const publishers = ['alpha', 'beta'];

  const runs = await Promise.all(publishers.map(async (publisher) => {
    const releasePath = join(root, publisher, `${publisher}.json`);
    const channelPath = join(root, publisher, 'stable.json');
    await writeJson(releasePath, {
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'test-release',
      publisher,
      artifacts: [],
    });
    await writeJson(channelPath, {
      schema: 'lc0_browser.artifact_channel_manifest.v1',
      channel: 'stable',
      releaseId: 'test-release',
      releaseManifestUrl: '/releases/test-release.json',
      publisher,
    });
    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--channel-manifest', channelPath,
      '--bucket', 'test-bucket',
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ], { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } });
    return { publisher, result };
  }));

  const succeeded = runs.filter(({ result }) => result.status === 0);
  const rejected = runs.filter(({ result }) => result.status !== 0);
  assert.equal(succeeded.length, 1, runs.map(({ publisher, result }) => `${publisher}: ${result.stderr}`).join('\n'));
  assert.equal(rejected.length, 1);
  assert.match(rejected[0].result.stderr, /Refusing to accept immutable object .*releases\/test-release\.json: Content-Length is/);

  const releaseTarget = 'test-bucket/releases/test-release.json';
  const channelTarget = 'test-bucket/channels/stable.json';
  const storedRelease = JSON.parse(await readFile(mockR2ObjectPath(r2Root, releaseTarget), 'utf8'));
  const storedChannel = JSON.parse(await readFile(mockR2ObjectPath(r2Root, channelTarget), 'utf8'));
  assert.equal(storedRelease.publisher, succeeded[0].publisher);
  assert.equal(storedChannel.publisher, succeeded[0].publisher);

  const log = await readFile(logPath, 'utf8');
  assert.equal(log.match(/s3api put-object --bucket test-bucket --key releases\/test-release\.json/g)?.length, 2);
  assert.equal(log.match(/r2 object put test-bucket\/channels\/stable\.json/g)?.length, 1);
});

test('publish_hashed_artifacts_to_r2 safely handles concurrent writers for the same immutable artifact representations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-artifact-race-'));
  const source = join(root, 'model.onnx');
  await writeFile(source, 'abc');
  const materialize = spawnSync(process.execPath, [
    'scripts/publish_content_addressed_release.mjs',
    '--root', root,
    '--release-id', 'artifact-race',
    '--asset', `model=${source}`,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(materialize.status, 0, materialize.stderr);

  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const r2Root = join(root, 'mock-r2');
    const logPath = join(root, 'publish.log');
    const wrangler = await writeStatefulWrangler(root);
    const aws = await writeAtomicAws(root);
    const args = [
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', join(root, 'releases/artifact-race.json'),
      '--channel-manifest', join(root, 'channels/stable.json'),
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
      '--aws-bin', aws,
      '--r2-endpoint', 'https://r2.invalid',
    ];
    const runs = await Promise.all([
      runNode(args, { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } }),
      runNode(args, { env: { ...process.env, LOG: logPath, MOCK_R2_DIR: r2Root } }),
    ]);
    assert.ok(runs.every((result) => result.status === 0), runs.map((result) => result.stderr).join('\n'));
    const plans = runs.map((result) => JSON.parse(result.stdout));
    for (const plan of plans) {
      assert.ok(plan.planned.every((item) => ['created-r2', 'identical-r2'].includes(item.remoteState)));
      assert.ok(plan.planned.every((item) => ['uploaded', 'skip-identical-r2'].includes(item.uploadAction)));
    }

    const release = JSON.parse(await readFile(join(root, 'releases/artifact-race.json'), 'utf8'));
    for (const representation of release.artifacts[0].representations) {
      const key = new URL(representation.url, 'https://assets.invalid').pathname.replace(/^\/+/, '');
      const stored = await readFile(mockR2ObjectPath(r2Root, `test-bucket/${key}`));
      assert.equal(createHash('sha256').update(stored).digest('hex'), representation.sha256);
    }
    const log = await readFile(logPath, 'utf8');
    assert.equal(log.match(new RegExp(`s3api put-object --bucket test-bucket --key artifacts/sha256/${ABC_SHA256}/identity`, 'g'))?.length, 2);
    assert.equal(log.match(new RegExp(`s3api put-object --bucket test-bucket --key artifacts/sha256/${ABC_SHA256}/br/[a-f0-9]{64}`, 'g'))?.length, 2);
    assert.doesNotMatch(log, /r2 object put test-bucket\/artifacts\//);
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
