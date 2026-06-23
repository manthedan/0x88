#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

const DEFAULT_ARTIFACT_BASE = 'https://assets.0x88.app';

function usage() {
  console.log(`Usage: node scripts/publish_hashed_artifacts_to_r2.mjs --release public/releases/ID.json --bucket BUCKET [options]\n\nOptions:\n  --root DIR          Repository root (default .)\n  --execute           Actually call wrangler; default is dry-run\n  --allow-missing     Skip artifacts whose localPath is absent\n  --wrangler-bin BIN  Wrangler binary (default wrangler)\n  --channel-manifest PATH  Optional generated channel manifest to publish after the release\n  --artifact-base URL Public artifact origin used to probe relative artifactUrl values (default https://assets.0x88.app)\n  --probe-existing    In dry-run mode, validate artifact URLs and mark existing uploads as skipped\n  -h, --help          Show help\n\nThe script publishes each local artifact to artifacts/sha256/<sha>/<file>. It verifies\nlocal size and SHA-256 before upload and intentionally has no overwrite flag. It also\npublishes the release manifest to releases/<file> and, when provided, the mutable channel\nmanifest to channels/<file> after artifact/release uploads. Configure R2 lifecycle/retention\nseparately; routine releases should never replace existing hashed keys.\n`);
}

function parseArgs(argv) {
  const args = { root: '.', execute: false, allowMissing: false, wranglerBin: 'wrangler', probeExisting: false, artifactBase: process.env.LC0_ARTIFACT_BASE_URL ?? DEFAULT_ARTIFACT_BASE };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--root' && next) { args.root = next; i += 1; continue; }
    if (arg === '--release' && next) { args.release = next; i += 1; continue; }
    if (arg === '--bucket' && next) { args.bucket = next; i += 1; continue; }
    if (arg === '--wrangler-bin' && next) { args.wranglerBin = next; i += 1; continue; }
    if (arg === '--channel-manifest' && next) { args.channelManifest = next; i += 1; continue; }
    if (arg === '--artifact-base' && next) { args.artifactBase = next; i += 1; continue; }
    if (arg === '--probe-existing') { args.probeExisting = true; continue; }
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

function publicArtifactUrl(args, artifact) {
  try {
    const parsed = new URL(artifact.artifactUrl);
    if (parsed.protocol === 'https:' || parsed.protocol === 'http:') return parsed.href;
  } catch {
    // Resolve relative artifact URLs below when a public artifact base is supplied.
  }
  if (!args.artifactBase) return undefined;
  return new URL(artifact.artifactUrl, args.artifactBase).href;
}

function headerValue(headers, name) {
  return headers.get(name) ?? headers.get(name.toLowerCase());
}

function cacheControlDirective(cacheControl, directive, expectedValue) {
  const wanted = directive.toLowerCase();
  for (const part of cacheControl.split(',')) {
    const [rawName, rawValue] = part.trim().split('=', 2);
    if (rawName?.toLowerCase() !== wanted) continue;
    if (expectedValue === undefined) return true;
    return rawValue?.replace(/^"|"$/g, '') === expectedValue;
  }
  return false;
}

async function sha256RemoteUrl(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Artifact hash fetch failed for ${url}: HTTP ${response.status}`);
  const hash = createHash('sha256');
  let bytes = 0;
  if (!response.body) {
    const body = new Uint8Array(await response.arrayBuffer());
    hash.update(body);
    bytes = body.byteLength;
  } else {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      hash.update(value);
      bytes += value.byteLength;
    }
  }
  return { bytes, sha256: hash.digest('hex') };
}

async function probeExistingArtifact(args, artifact) {
  const url = publicArtifactUrl(args, artifact);
  if (!url) return { state: 'unchecked', reason: 'no public artifact URL; pass --artifact-base for relative artifactUrl values' };
  const response = await fetch(url, { method: 'HEAD', cache: 'no-cache' });
  if (response.status === 404) return { state: 'missing', url, status: response.status };
  if (!response.ok) throw new Error(`Artifact probe failed for ${url}: HTTP ${response.status}`);
  const lengthHeader = headerValue(response.headers, 'x-artifact-content-length') ?? headerValue(response.headers, 'content-length');
  const length = Number(lengthHeader ?? '');
  if (!Number.isFinite(length) || length !== artifact.bytes) {
    throw new Error(`Remote artifact size mismatch for ${artifact.logicalUrl}: got ${lengthHeader ?? 'missing'}, expected ${artifact.bytes}`);
  }
  const cacheControl = headerValue(response.headers, 'cache-control') ?? '';
  if (!cacheControlDirective(cacheControl, 'immutable') || !cacheControlDirective(cacheControl, 'max-age', '31536000')) {
    throw new Error(`Remote artifact cache policy is not immutable for ${artifact.logicalUrl}: ${cacheControl || 'missing'}`);
  }
  const actual = await sha256RemoteUrl(url);
  if (actual.bytes !== artifact.bytes) {
    throw new Error(`Remote artifact body size mismatch for ${artifact.logicalUrl}: got ${actual.bytes}, expected ${artifact.bytes}`);
  }
  if (actual.sha256 !== artifact.sha256.toLowerCase()) {
    throw new Error(`Remote artifact SHA-256 mismatch for ${artifact.logicalUrl}: got ${actual.sha256}, expected ${artifact.sha256.toLowerCase()}`);
  }
  return { state: 'existing', url, status: response.status, bytes: length, sha256: actual.sha256 };
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
    const probe = (args.probeExisting || args.execute) ? await probeExistingArtifact(args, artifact) : undefined;
    planned.push({
      logicalUrl: artifact.logicalUrl,
      localPath,
      key,
      bytes: artifact.bytes,
      sha256: artifact.sha256,
      contentType: artifact.contentType ?? 'application/octet-stream',
      artifactUrl: publicArtifactUrl(args, artifact),
      remoteState: probe?.state ?? 'not-probed',
      uploadAction: probe?.state === 'existing' ? 'skip-existing' : 'upload',
      remoteProbe: probe,
    });
  }

  const manifests = await manifestPublishItems(args, release);

  if (args.execute && skipped.length) {
    throw new Error('Refusing to publish release/channel manifests when artifacts were skipped; rerun without --allow-missing or verify/upload all artifacts first');
  }

  if (args.execute) {
    for (const item of planned) {
      const target = `${args.bucket}/${item.key}`;
      if (item.remoteState === 'existing') continue;
      if (item.remoteState === 'unchecked') {
        throw new Error(`Cannot safely publish ${item.logicalUrl}: ${item.remoteProbe?.reason ?? 'remote artifact existence was not checked'}`);
      }
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
