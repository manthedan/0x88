#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

function usage() {
  console.log(`Usage: node scripts/publish_hashed_artifacts_to_r2.mjs --release public/releases/ID.json --bucket BUCKET [options]\n\nOptions:\n  --root DIR          Repository root (default .)\n  --execute           Actually call wrangler; default is dry-run\n  --allow-missing     Skip artifacts whose localPath is absent\n  --wrangler-bin BIN  Wrangler binary (default wrangler)\n  --channel-manifest PATH  Optional generated channel manifest to publish after the release\n  -h, --help          Show help\n\nThe script publishes each local artifact to artifacts/sha256/<sha>/<file>. It verifies\nlocal size and SHA-256 before upload and intentionally has no overwrite flag. It also\npublishes the release manifest to releases/<file> and, when provided, the mutable channel\nmanifest to channels/<file> after artifact/release uploads. Configure R2 lifecycle/retention\nseparately; routine releases should never replace existing hashed keys.\n`);
}

function parseArgs(argv) {
  const args = { root: '.', execute: false, allowMissing: false, wranglerBin: 'wrangler' };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--root' && next) { args.root = next; i += 1; continue; }
    if (arg === '--release' && next) { args.release = next; i += 1; continue; }
    if (arg === '--bucket' && next) { args.bucket = next; i += 1; continue; }
    if (arg === '--wrangler-bin' && next) { args.wranglerBin = next; i += 1; continue; }
    if (arg === '--channel-manifest' && next) { args.channelManifest = next; i += 1; continue; }
    if (arg === '--execute') { args.execute = true; continue; }
    if (arg === '--allow-missing') { args.allowMissing = true; continue; }
    if (arg === '-h' || arg === '--help') { usage(); process.exit(0); }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.release) throw new Error('--release is required');
  if (!args.bucket) throw new Error('--bucket is required');
  return args;
}

async function sha256File(path) {
  const buf = await readFile(path);
  return { bytes: buf.byteLength, sha256: createHash('sha256').update(buf).digest('hex') };
}

function keyFromArtifact(artifact) {
  const marker = '/artifacts/sha256/';
  const url = artifact.artifactUrl;
  const idx = url.indexOf(marker);
  if (idx < 0) throw new Error(`Artifact URL is not content-addressed: ${url}`);
  return url.slice(idx + 1);
}

function sha256FromArtifactKey(key) {
  const match = key.match(/^artifacts\/sha256\/([a-f0-9]{64})\//);
  return match?.[1];
}

async function assertRemoteObjectMissing(args, target) {
  const tempDir = mkdtemp(join(tmpdir(), 'lc0-r2-exists-'));
  return tempDir.then((dir) => {
    const file = join(dir, 'object');
    const child = spawnSync(args.wranglerBin, ['r2', 'object', 'get', target, '--file', file, '--remote'], { stdio: 'ignore' });
    return rm(dir, { recursive: true, force: true }).then(() => {
      if (child.status === 0) throw new Error(`Refusing to overwrite immutable release manifest ${target}`);
    });
  });
}

async function manifestPublishItems(args, release) {
  const releaseKey = `releases/${basename(args.release)}`;
  const items = [{
    type: 'release-manifest',
    localPath: args.release,
    key: releaseKey,
    contentType: 'application/json; charset=utf-8',
    cacheControl: 'public, max-age=300, stale-while-revalidate=86400',
  }];
  if (args.channelManifest) {
    const channel = JSON.parse(await readFile(args.channelManifest, 'utf8'));
    if (channel.schema !== 'lc0_browser.artifact_channel_manifest.v1') throw new Error(`Unexpected channel schema: ${channel.schema}`);
    if (channel.releaseId !== release.releaseId) throw new Error(`Channel releaseId ${channel.releaseId} does not match release ${release.releaseId}`);
    if (channel.releaseManifestUrl !== `/${releaseKey}`) {
      throw new Error(`Channel releaseManifestUrl ${channel.releaseManifestUrl} does not match /${releaseKey}`);
    }
    items.push({
      type: 'channel-manifest',
      localPath: args.channelManifest,
      key: `channels/${basename(args.channelManifest)}`,
      contentType: 'application/json; charset=utf-8',
      cacheControl: 'no-cache',
    });
  }
  return items;
}

async function main() {
  const args = parseArgs(process.argv);
  const release = JSON.parse(await readFile(args.release, 'utf8'));
  if (release.schema !== 'lc0_browser.artifact_release_manifest.v1') throw new Error(`Unexpected release schema: ${release.schema}`);
  const planned = [];
  const skipped = [];
  for (const artifact of release.artifacts ?? []) {
    if (!artifact.localPath) { skipped.push({ logicalUrl: artifact.logicalUrl, reason: 'no localPath' }); continue; }
    const localPath = `${args.root.replace(/\/$/, '')}/${artifact.localPath}`;
    if (!existsSync(localPath)) {
      if (args.allowMissing) { skipped.push({ logicalUrl: artifact.logicalUrl, reason: 'missing localPath', localPath }); continue; }
      throw new Error(`Missing local artifact for ${artifact.logicalUrl}: ${localPath}`);
    }
    const actual = await sha256File(localPath);
    if (actual.bytes !== artifact.bytes) throw new Error(`Size mismatch for ${artifact.logicalUrl}: got ${actual.bytes}, expected ${artifact.bytes}`);
    if (actual.sha256 !== artifact.sha256) throw new Error(`SHA-256 mismatch for ${artifact.logicalUrl}: got ${actual.sha256}, expected ${artifact.sha256}`);
    const key = keyFromArtifact(artifact);
    const keySha256 = sha256FromArtifactKey(key);
    if (keySha256 !== artifact.sha256.toLowerCase()) {
      throw new Error(`Content-addressed key mismatch for ${artifact.logicalUrl}: key has ${keySha256 ?? 'no sha256'}, manifest has ${artifact.sha256}`);
    }
    planned.push({ logicalUrl: artifact.logicalUrl, localPath, key, bytes: artifact.bytes, sha256: artifact.sha256, contentType: artifact.contentType ?? 'application/octet-stream' });
  }

  const manifests = await manifestPublishItems(args, release);

  if (args.execute && skipped.length) {
    throw new Error('Refusing to publish release/channel manifests when artifacts were skipped; rerun without --allow-missing or verify/upload all artifacts first');
  }

  if (args.execute) {
    for (const item of planned) {
      const target = `${args.bucket}/${item.key}`;
      const child = spawnSync(args.wranglerBin, [
        'r2', 'object', 'put', target,
        '--file', item.localPath,
        '--content-type', item.contentType,
        '--cache-control', 'public, max-age=31536000, immutable',
        '--remote',
      ], { stdio: 'inherit' });
      if (child.status !== 0) throw new Error(`wrangler failed for ${target}`);
    }
    for (const item of manifests) {
      const target = `${args.bucket}/${item.key}`;
      if (item.type === 'release-manifest') await assertRemoteObjectMissing(args, target);
      const child = spawnSync(args.wranglerBin, [
        'r2', 'object', 'put', target,
        '--file', item.localPath,
        '--content-type', item.contentType,
        '--cache-control', item.cacheControl,
        '--remote',
      ], { stdio: 'inherit' });
      if (child.status !== 0) throw new Error(`wrangler failed for ${target}`);
    }
  }

  console.log(JSON.stringify({
    schema: 'lc0_browser.r2_hashed_artifact_publish_plan.v1',
    releaseId: release.releaseId,
    execute: args.execute,
    bucket: args.bucket,
    plannedCount: planned.length,
    skippedCount: skipped.length,
    planned,
    skipped,
    manifests,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
