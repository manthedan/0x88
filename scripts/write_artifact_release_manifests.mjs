#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, join, posix, relative } from 'node:path';
import { spawnSync } from 'node:child_process';

const DEFAULT_SOURCE_MANIFESTS = [
  'public/models/lc0/manifest.json',
  'public/models/maia3/manifest.json',
  'public/stockfish/stockfish-18.0.7.manifest.json',
  'public/viridithas/viridithas-wasip1.manifest.json',
  'public/berserk/berserk-emscripten-single-thread.manifest.json',
  'public/plentychess/plentychess-emscripten-single-thread.manifest.json',
];

function usage() {
  console.log(`Usage: node scripts/write_artifact_release_manifests.mjs [options]\n\nOptions:\n  --root DIR             Repository root (default .)\n  --release-id ID        Immutable release id (default date + git short sha)\n  --channel NAME         Channel name to write (default stable)\n  --out-dir DIR          Public output root (default public under --root)\n  --asset-origin URL     Absolute asset origin prefix (default https://assets.0x88.app)\n  --manifest PATH        Source manifest to include; may be repeated\n  --generated-at ISO     Override generatedAt for reproducible checks\n  --check                Verify existing outputs match instead of writing\n  -h, --help             Show help\n`);
}

function parseArgs(argv) {
  const args = { root: '.', channel: 'stable', manifests: [], check: false };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--root' && next) { args.root = next; i += 1; continue; }
    if (arg === '--release-id' && next) { args.releaseId = next; i += 1; continue; }
    if (arg === '--channel' && next) { args.channel = next; i += 1; continue; }
    if (arg === '--out-dir' && next) { args.outDir = next; i += 1; continue; }
    if (arg === '--asset-origin' && next) { args.assetOrigin = next.replace(/\/+$/, ''); i += 1; continue; }
    if (arg === '--manifest' && next) { args.manifests.push(next); i += 1; continue; }
    if (arg === '--generated-at' && next) { args.generatedAt = next; i += 1; continue; }
    if (arg === '--check') { args.check = true; continue; }
    if (arg === '-h' || arg === '--help') { usage(); process.exit(0); }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.releaseId) args.releaseId = defaultReleaseId(args.root);
  if (!args.assetOrigin) args.assetOrigin = process.env.LC0_ARTIFACT_ASSET_ORIGIN ?? 'https://assets.0x88.app';
  if (!args.outDir) args.outDir = join(args.root, 'public');
  if (!args.manifests.length) args.manifests = DEFAULT_SOURCE_MANIFESTS;
  return args;
}

function defaultReleaseId(root) {
  const day = new Date().toISOString().slice(0, 10);
  const git = spawnSync('git', ['rev-parse', '--short=12', 'HEAD'], { cwd: root, encoding: 'utf8' });
  const sha = git.status === 0 ? git.stdout.trim() : 'nogit';
  return `${day}.${sha}`;
}

function contentTypeFor(file) {
  if (file.endsWith('.wasm')) return 'application/wasm';
  if (file.endsWith('.js')) return 'text/javascript; charset=utf-8';
  if (file.endsWith('.json')) return 'application/json';
  if (file.endsWith('.onnx')) return 'application/octet-stream';
  if (file.endsWith('.data') || file.endsWith('.bin') || file.endsWith('.nn') || file.endsWith('.nnue')) return 'application/octet-stream';
  if (file.endsWith('.gz')) return 'application/gzip';
  return 'application/octet-stream';
}

function artifactUrlFor(sha256, file, assetOrigin) {
  const path = `/artifacts/sha256/${sha256}/${encodeURIComponent(file)}`;
  return assetOrigin ? `${assetOrigin}${path}` : path;
}

function logicalUrlFromPublicPath(path) {
  return `/${path.replace(/^public\//, '')}`;
}

function artifactFromModelEntry(entry, sourceManifest, manifestPath, args) {
  if (!entry?.sha256 || !Number.isFinite(entry.bytes)) return undefined;
  const file = entry.file ?? basename(entry.url ?? 'artifact');
  const logicalUrl = entry.url?.startsWith('/') ? entry.url : logicalUrlFromPublicPath(posix.join(dirname(manifestPath).replace(/\\/g, '/'), file));
  return {
    logicalUrl,
    artifactUrl: artifactUrlFor(entry.sha256, file, args.assetOrigin),
    sha256: entry.sha256.toLowerCase(),
    bytes: entry.bytes,
    file,
    kind: 'model',
    contentType: contentTypeFor(file),
    sourceManifest,
    localPath: posix.join(dirname(manifestPath).replace(/\\/g, '/'), file),
    status: entry.mode,
  };
}

function artifactFromEngineEntry(entry, sourceManifest, args) {
  if (!entry?.sha256 || !Number.isFinite(entry.bytes) || !entry.path) return undefined;
  const file = basename(entry.path);
  const kind = file.includes('source') || file.endsWith('.tar.gz') ? 'source' : 'engine';
  return {
    logicalUrl: logicalUrlFromPublicPath(entry.path),
    artifactUrl: artifactUrlFor(entry.sha256, file, args.assetOrigin),
    sha256: entry.sha256.toLowerCase(),
    bytes: entry.bytes,
    file,
    kind,
    contentType: contentTypeFor(file),
    sourceManifest,
    localPath: entry.path,
  };
}

async function localFileDigest(path) {
  if (!existsSync(path)) return undefined;
  const buf = await readFile(path);
  return { bytes: buf.byteLength, sha256: createHash('sha256').update(buf).digest('hex') };
}

async function artifactsFromPackEntry(pack, sourceManifest, manifestPath, args) {
  if (!pack?.url) return [];
  const artifacts = [];
  const packDir = posix.dirname(pack.url);
  const localDir = posix.dirname(manifestPath).replace(/\\/g, '/');
  const metadataFile = basename(pack.url);
  const metadataLocalPath = posix.join(localDir, pack.id ?? basename(packDir), metadataFile);
  const metadataDigest = await localFileDigest(posix.join(args.root.replace(/\/$/, ''), metadataLocalPath));
  if (metadataDigest) {
    artifacts.push({
      logicalUrl: pack.url,
      artifactUrl: artifactUrlFor(metadataDigest.sha256, metadataFile, args.assetOrigin),
      sha256: metadataDigest.sha256,
      bytes: metadataDigest.bytes,
      file: metadataFile,
      kind: 'model',
      contentType: contentTypeFor(metadataFile),
      sourceManifest,
      localPath: metadataLocalPath,
      status: pack.mode,
    });
  }
  for (const shard of pack.shards ?? []) {
    if (!shard.sha256 || !Number.isFinite(shard.bytes) || !shard.file) continue;
    artifacts.push({
      logicalUrl: posix.join(packDir, shard.file),
      artifactUrl: artifactUrlFor(shard.sha256.toLowerCase(), shard.file, args.assetOrigin),
      sha256: shard.sha256.toLowerCase(),
      bytes: shard.bytes,
      file: shard.file,
      kind: 'model',
      contentType: contentTypeFor(shard.file),
      sourceManifest,
      localPath: posix.join(localDir, pack.id ?? basename(packDir), shard.file),
      status: pack.mode,
    });
  }
  return artifacts;
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function verifyLocalArtifact(artifact, args) {
  if (!artifact.localPath) throw new Error(`Artifact ${artifact.logicalUrl} has no localPath; cannot publish a verifiable release entry`);
  const absolute = posix.join(args.root.replace(/\/$/, ''), artifact.localPath);
  const digest = await localFileDigest(absolute);
  if (!digest) throw new Error(`Missing local artifact for ${artifact.logicalUrl}: ${artifact.localPath}`);
  if (digest.bytes !== artifact.bytes || digest.sha256 !== artifact.sha256) {
    throw new Error(`Manifest metadata mismatch for ${artifact.logicalUrl}: local ${digest.bytes}/${digest.sha256}, manifest ${artifact.bytes}/${artifact.sha256}`);
  }
  return artifact;
}

async function collectArtifacts(args) {
  const artifacts = [];
  const sourceManifests = [];
  for (const manifestArg of args.manifests) {
    const manifestPath = manifestArg.startsWith('public/') ? manifestArg : relative(args.root, manifestArg).replace(/\\/g, '/');
    const absolute = join(args.root, manifestPath);
    if (!existsSync(absolute)) continue;
    sourceManifests.push(manifestPath);
    const manifest = await readJson(absolute);
    if (Array.isArray(manifest.models)) {
      for (const entry of manifest.models) {
        const artifact = artifactFromModelEntry(entry, manifestPath, manifestPath, args);
        if (artifact) artifacts.push(artifact);
      }
    }
    if (Array.isArray(manifest.packs)) {
      for (const pack of manifest.packs) artifacts.push(...await artifactsFromPackEntry(pack, manifestPath, manifestPath, args));
    }
    if (Array.isArray(manifest.artifacts)) {
      for (const entry of manifest.artifacts) {
        const artifact = artifactFromEngineEntry(entry, manifestPath, args);
        if (artifact) artifacts.push(artifact);
      }
    }
    if (manifest.sourceArchive?.sha256 && manifest.sourceArchive?.path) {
      const artifact = artifactFromEngineEntry(manifest.sourceArchive, manifestPath, args);
      if (artifact) artifacts.push(artifact);
    }
  }
  const byLogical = new Map();
  for (const artifact of artifacts) {
    const verified = await verifyLocalArtifact(artifact, args);
    if (verified) byLogical.set(verified.logicalUrl, verified);
  }
  return { artifacts: [...byLogical.values()].sort((a, b) => a.logicalUrl.localeCompare(b.logicalUrl)), sourceManifests };
}

async function writeOrCheck(path, value, check) {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  if (check) {
    const existing = await readFile(path, 'utf8');
    if (existing !== text) throw new Error(`${path} is stale; rerun write_artifact_release_manifests.mjs`);
    return;
  }
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, text);
}

async function main() {
  const args = parseArgs(process.argv);
  const generatedAt = args.generatedAt ?? new Date().toISOString();
  const { artifacts, sourceManifests } = await collectArtifacts(args);
  const releaseManifestUrl = `/releases/${args.releaseId}.json`;
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: args.releaseId,
    generatedAt,
    channel: args.channel,
    sourceManifests,
    artifacts,
  };
  const channel = {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    channel: args.channel,
    releaseId: args.releaseId,
    releaseManifestUrl,
    generatedAt,
  };

  const releasePath = join(args.outDir, 'releases', `${args.releaseId}.json`);
  const channelPath = join(args.outDir, 'channels', `${args.channel}.json`);
  await writeOrCheck(releasePath, release, args.check);
  await writeOrCheck(channelPath, channel, args.check);
  console.log(JSON.stringify({ ok: true, releasePath, channelPath, artifactCount: artifacts.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
