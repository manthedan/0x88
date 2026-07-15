import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { chmod, mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { brotliDecompressSync } from 'node:zlib';

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
import { appendFileSync, constants, copyFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';

const args = process.argv.slice(2);
if (process.env.LOG) appendFileSync(process.env.LOG, \`\${args.join(' ')}\\n\`);
if (
  args[0] !== 's3api'
  || args[1] !== 'put-object'
  || args[args.indexOf('--if-none-match') + 1] !== '*'
  || args[args.indexOf('--region') + 1] !== 'auto'
) process.exit(2);
const bucket = args[args.indexOf('--bucket') + 1];
const key = args[args.indexOf('--key') + 1];
const file = args[args.indexOf('--body') + 1];
const target = \`\${bucket}/\${key}\`;
const objectPath = join(process.env.MOCK_R2_DIR, createHash('sha256').update(target).digest('hex'));
mkdirSync(dirname(objectPath), { recursive: true });
try {
  copyFileSync(file, objectPath, constants.COPYFILE_EXCL);
} catch (error) {
  if (error.code === 'EEXIST') {
    console.error('PreconditionFailed: status code: 412');
    process.exit(1);
  }
  throw error;
}
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

test('write_artifact_release_manifests carries forward an immutable base release without local legacy files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-release-carry-forward-'));
  await mkdir(join(root, 'public/stormphrax'), { recursive: true });
  await writeFile(join(root, 'public/stormphrax/engine.wasm'), 'abc');
  await writeJson(join(root, 'public/stormphrax/manifest.json'), {
    artifacts: [{ path: 'public/stormphrax/engine.wasm', bytes: 3, sha256: ABC_SHA256 }],
  });
  await writeJson(join(root, 'public/releases/base.json'), {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'base',
    sourceManifests: ['public/legacy/manifest.json'],
    artifacts: [{
      logicalUrl: '/legacy/engine.wasm',
      artifactUrl: `https://assets.example/artifacts/sha256/${ABC_SHA256}/legacy.wasm`,
      sha256: ABC_SHA256,
      bytes: 3,
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
  assert.equal(release.artifacts[0].artifactUrl, `https://assets.example/artifacts/sha256/${ABC_SHA256}/legacy.wasm`);
  assert.equal(release.artifacts[0].representations, undefined, 'v1 body key remains readable during migration');
  assert.equal(release.artifacts[1].carriedForwardFrom, undefined);
  assert.equal(release.artifacts[1].representations[0].url.endsWith(`/${ABC_SHA256}/identity`), true);
  assert.equal(release.artifacts.filter((artifact) => artifact.kind === 'manifest').length, 1);
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

test('publish_hashed_artifacts_to_r2 retains mixed v1 entries inside a migrated v2 release', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-mixed-release-'));
  await writeFile(join(root, 'legacy.wasm'), 'abc');
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
  await writeJson(releasePath, {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'mixed',
    artifacts: [
      {
        logicalUrl: '/legacy/engine.wasm',
        artifactUrl: `/artifacts/sha256/${ABC_SHA256}/legacy.wasm`,
        sha256: ABC_SHA256,
        bytes: 3,
        localPath: 'legacy.wasm',
      },
      {
        logicalUrl: '/models/model.onnx',
        raw: { sha256: ABC_SHA256, bytes: 3 },
        representations: [{
          encoding: 'identity',
          url: `/artifacts/sha256/${ABC_SHA256}/identity`,
          sha256: ABC_SHA256,
          bytes: 3,
          localPath: 'model.onnx',
        }],
      },
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
  assert.equal(parsed.catalogObjectCount, 2);
  assert.deepEqual(parsed.planned.map((entry) => entry.key).sort(), [
    `artifacts/sha256/${ABC_SHA256}/identity`,
    `artifacts/sha256/${ABC_SHA256}/legacy.wasm`,
  ]);
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
    assert.match(log, new RegExp(`r2 object put test-bucket/artifacts/sha256/${ABC_SHA256}/identity`));
    assert.match(log, /\/br\/[a-f0-9]{64} .*--content-encoding br --remote/);
    assert.match(log, /s3api put-object --bucket test-bucket --key releases\/v2-execute\.json .*--if-none-match \* .*--endpoint-url https:\/\/r2\.invalid --region auto/);
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
    const target = `test-bucket/artifacts/sha256/${ABC_SHA256}/identity`;
    await mkdir(r2Root, { recursive: true });
    await writeFile(mockR2ObjectPath(r2Root, target), 'abd');

    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
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

test('publish_hashed_artifacts_to_r2 fails when authoritative post-create verification detects a race', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-r2-publish-create-race-'));
  await writeFile(join(root, 'model.onnx'), 'abc');
  const releasePath = join(root, 'release.json');
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
  const server = createServer((_req, res) => res.writeHead(404).end());
  const port = await listen(server);
  try {
    const logPath = join(root, 'wrangler.log');
    const statePath = join(root, 'get-count');
    const wrangler = join(root, 'fake-racing-wrangler.sh');
    await writeFile(wrangler, `#!/bin/sh
printf "%s\\n" "$*" >> "$LOG"
if [ "$1 $2 $3" = "r2 object get" ]; then
  count=0
  if [ -f "$STATE" ]; then count=$(cat "$STATE"); fi
  count=$((count + 1))
  printf "%s" "$count" > "$STATE"
  if [ "$count" -eq 1 ]; then
    echo "The specified key does not exist." >&2
    exit 1
  fi
  printf "abd" > "$6"
  exit 0
fi
exit 0
`);
    await chmod(wrangler, 0o755);

    const result = await runNode([
      'scripts/publish_hashed_artifacts_to_r2.mjs',
      '--root', root,
      '--release', releasePath,
      '--bucket', 'test-bucket',
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--execute',
      '--wrangler-bin', wrangler,
    ], { env: { ...process.env, LOG: logPath, STATE: statePath } });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Refusing to overwrite immutable object .*remote content differs/);
    const log = await readFile(logPath, 'utf8');
    assert.match(log, /r2 object get .*identity/);
    assert.match(log, /r2 object put .*identity/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
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
    await mkdir(r2Root, { recursive: true });
    await writeFile(mockR2ObjectPath(r2Root, target), 'abc');
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
    await mkdir(r2Root, { recursive: true });
    await writeFile(mockR2ObjectPath(r2Root, releaseTarget), '{"different":true}\n');
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
    assert.match(result.stderr, /Refusing to overwrite immutable object .*releases\/test-release\.json/);
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
    await writeFile(wrangler, '#!/bin/sh\nif [ "$1 $2 $3" = "r2 object get" ]; then echo "authentication failed" >&2; exit 1; fi\nexit 0\n');
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
    await mkdir(r2Root, { recursive: true });
    await writeFile(mockR2ObjectPath(r2Root, releaseTarget), await readFile(releasePath));
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
  assert.match(rejected[0].result.stderr, /Refusing to overwrite immutable object .*releases\/test-release\.json: remote content differs/);

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
