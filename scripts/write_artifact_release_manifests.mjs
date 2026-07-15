#!/usr/bin/env node
import { createHash, randomBytes } from 'node:crypto';
import { constants as fsConstants, createReadStream, createWriteStream, existsSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, isAbsolute, join, posix, relative, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { pipeline } from 'node:stream/promises';
import { constants as zlibConstants, createBrotliCompress } from 'node:zlib';
import {
  ARTIFACT_RELEASE_V1_SCHEMA,
  ARTIFACT_RELEASE_V2_SCHEMAS,
  isArtifactReleaseV2,
  releaseCatalogEntries,
} from './engine_artifact_registry.mjs';

const DEFAULT_SOURCE_MANIFESTS = [
  'public/models/lc0/r2-v0-present.manifest.json',
  'public/models/maia3/manifest.json',
  'public/models/centipawn.manifest.json',
  'public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json',
  'public/stockfish/stockfish-18.0.7.manifest.json',
  'public/reckless/reckless-wasip1.manifest.json',
  'public/viridithas/viridithas-wasip1.manifest.json',
  'public/berserk/berserk-emscripten-single-thread.manifest.json',
  'public/plentychess/plentychess-emscripten-single-thread.manifest.json',
  'public/stormphrax/stormphrax-emscripten-single-thread.manifest.json',
];
const DEFAULT_OUTPUT_DIRECTORY = '.local-dev-artifacts/artifact-releases';

function usage() {
  console.log(`Usage: node scripts/write_artifact_release_manifests.mjs [options]\n\nOptions:\n  --root DIR             Repository root (default .)\n  --release-id ID        Immutable release id (default date + git short sha)\n  --channel NAME         Channel name to write (default stable)\n  --out-dir DIR          Staging output root (default .local-dev-artifacts/artifact-releases under --root)\n  --asset-origin URL     Absolute asset origin prefix (default https://assets.0x88.app)\n  --manifest PATH        Source manifest to include; may be repeated\n  --base-release PATH    Carry forward immutable entries from an existing v1/v2 release\n  --generated-at ISO     Override generatedAt for reproducible checks\n  --no-brotli            Emit identity representations only\n  --brotli-quality N     Brotli quality 0-11 (default 5)\n  --check                Verify existing outputs match instead of writing\n  -h, --help             Show help\n`);
}

function parseArgs(argv) {
  const args = { root: '.', channel: 'stable', manifests: [], check: false, brotli: true, brotliQuality: 5 };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--root' && next) { args.root = next; i += 1; continue; }
    if (arg === '--release-id' && next) { args.releaseId = next; i += 1; continue; }
    if (arg === '--channel' && next) { args.channel = next; i += 1; continue; }
    if (arg === '--out-dir' && next) { args.outDir = next; i += 1; continue; }
    if (arg === '--asset-origin' && next) { args.assetOrigin = next.replace(/\/+$/, ''); i += 1; continue; }
    if (arg === '--manifest' && next) { args.manifests.push(next); i += 1; continue; }
    if (arg === '--base-release' && next) { args.baseRelease = next; i += 1; continue; }
    if (arg === '--generated-at' && next) { args.generatedAt = next; i += 1; continue; }
    if (arg === '--no-brotli') { args.brotli = false; continue; }
    if (arg === '--brotli-quality' && next) { args.brotliQuality = Number(next); i += 1; continue; }
    if (arg === '--check') { args.check = true; continue; }
    if (arg === '-h' || arg === '--help') { usage(); process.exit(0); }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.releaseId) args.releaseId = defaultReleaseId(args.root);
  if (!args.assetOrigin) args.assetOrigin = process.env.LC0_ARTIFACT_ASSET_ORIGIN ?? 'https://assets.0x88.app';
  if (!args.outDir) args.outDir = join(args.root, DEFAULT_OUTPUT_DIRECTORY);
  if (!args.manifests.length) args.manifests = DEFAULT_SOURCE_MANIFESTS;
  if (!/^[A-Za-z0-9._-]+$/.test(args.releaseId)) {
    throw new Error('--release-id must contain only letters, numbers, dots, underscores, and hyphens');
  }
  if (!/^[A-Za-z0-9._-]+$/.test(args.channel)) {
    throw new Error('--channel must contain only letters, numbers, dots, underscores, and hyphens');
  }
  if (!Number.isInteger(args.brotliQuality) || args.brotliQuality < 0 || args.brotliQuality > 11) {
    throw new Error('--brotli-quality must be an integer from 0 through 11');
  }
  if (args.generatedAt && !isIsoTimestamp(args.generatedAt)) {
    throw new Error('--generated-at must be an ISO timestamp');
  }
  return args;
}

function isIsoTimestamp(value) {
  return typeof value === 'string'
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)
    && Number.isFinite(Date.parse(value));
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
  if (file.endsWith('.patch')) return 'text/plain; charset=utf-8';
  if (file.endsWith('.onnx')) return 'application/octet-stream';
  if (file.endsWith('.data') || file.endsWith('.bin') || file.endsWith('.nn') || file.endsWith('.nnue')) return 'application/octet-stream';
  if (file.endsWith('.gz')) return 'application/gzip';
  return 'application/octet-stream';
}

function artifactUrlForKey(key, assetOrigin) {
  const path = `/${key}`;
  return assetOrigin ? `${assetOrigin}${path}` : path;
}

function logicalUrlFromPublicPath(path) {
  return `/${path.replace(/^public\//, '')}`;
}

function artifactFromModelEntry(entry, sourceManifest, manifestPath, args) {
  if (!entry?.sha256 || !Number.isFinite(entry.bytes) || (!entry.file && !entry.url)) return undefined;
  const file = entry.file ?? basename(entry.url ?? 'artifact');
  const logicalUrl = entry.url?.startsWith('/') ? entry.url : logicalUrlFromPublicPath(posix.join(dirname(manifestPath).replace(/\\/g, '/'), file));
  return {
    logicalUrl,
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
    sha256: entry.sha256.toLowerCase(),
    bytes: entry.bytes,
    file,
    kind,
    contentType: contentTypeFor(file),
    sourceManifest,
    localPath: entry.path,
  };
}

function artifactFromTvmjsFile(entry, sourceManifest, manifestPath, args) {
  if (!entry?.path || !entry?.sha256 || !Number.isFinite(entry.bytes)) return undefined;
  const normalizedPath = posix.normalize(String(entry.path).replace(/\\/g, '/'));
  if (normalizedPath.startsWith('../') || posix.isAbsolute(normalizedPath)) {
    throw new Error(`Unsafe TVMJS artifact path in ${sourceManifest}: ${entry.path}`);
  }
  const manifestDir = posix.dirname(manifestPath).replace(/\\/g, '/');
  const file = basename(normalizedPath);
  return {
    logicalUrl: logicalUrlFromPublicPath(posix.join(manifestDir, normalizedPath)),
    sha256: entry.sha256.toLowerCase(),
    bytes: entry.bytes,
    file,
    kind: 'runtime',
    contentType: contentTypeFor(file),
    sourceManifest,
    localPath: posix.join(manifestDir, normalizedPath),
  };
}

async function localFileDigest(path) {
  if (!existsSync(path)) return undefined;
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  return { bytes, sha256: hash.digest('hex') };
}

function storedLocalPath(path, args) {
  const candidate = relative(resolve(args.root), path).replace(/\\/g, '/');
  return candidate.startsWith('../') ? path : candidate;
}

async function installWriteOnce(sourcePath, targetPath, expected, check) {
  const existing = await localFileDigest(targetPath);
  if (existing) {
    if (existing.bytes !== expected.bytes || existing.sha256 !== expected.sha256) {
      throw new Error(`Content-addressed key collision or corruption at ${targetPath}`);
    }
    return;
  }
  if (check) throw new Error(`Missing content-addressed representation: ${targetPath}`);
  await mkdir(dirname(targetPath), { recursive: true });
  try {
    await copyFile(sourcePath, targetPath, fsConstants.COPYFILE_EXCL);
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
    const raced = await localFileDigest(targetPath);
    if (!raced || raced.bytes !== expected.bytes || raced.sha256 !== expected.sha256) {
      throw new Error(`Content-addressed key collision or corruption at ${targetPath}`);
    }
  }
}

async function materializeRepresentations(artifact, args) {
  const raw = { bytes: artifact.bytes, sha256: artifact.sha256 };
  const identityKey = `artifacts/sha256/${raw.sha256}/identity`;
  const identityTarget = resolve(args.outDir, identityKey);
  const sourcePath = isAbsolute(artifact.localPath) ? artifact.localPath : resolve(args.root, artifact.localPath);
  await installWriteOnce(sourcePath, identityTarget, raw, args.check);
  const representations = [{
    encoding: 'identity',
    url: artifactUrlForKey(identityKey, args.assetOrigin),
    sha256: raw.sha256,
    bytes: raw.bytes,
    localPath: storedLocalPath(identityTarget, args),
  }];
  if (args.brotli) {
    const tempDirectory = args.check ? await mkdtemp(join(tmpdir(), 'lc0-artifact-brotli-')) : undefined;
    const tempPath = tempDirectory
      ? join(tempDirectory, 'representation.br')
      : resolve(args.outDir, `.artifact-brotli-${process.pid}-${randomBytes(6).toString('hex')}`);
    try {
      if (!tempDirectory) await mkdir(dirname(tempPath), { recursive: true });
      await pipeline(
        createReadStream(sourcePath),
        createBrotliCompress({ params: { [zlibConstants.BROTLI_PARAM_QUALITY]: args.brotliQuality } }),
        createWriteStream(tempPath, { flags: 'wx' }),
      );
      const encoded = await localFileDigest(tempPath);
      const key = `artifacts/sha256/${raw.sha256}/br/${encoded.sha256}`;
      const target = resolve(args.outDir, key);
      await installWriteOnce(tempPath, target, encoded, args.check);
      representations.push({
        encoding: 'br',
        url: artifactUrlForKey(key, args.assetOrigin),
        sha256: encoded.sha256,
        bytes: encoded.bytes,
        localPath: storedLocalPath(target, args),
      });
    } finally {
      await rm(tempDirectory ?? tempPath, { recursive: Boolean(tempDirectory), force: true });
    }
  }
  return {
    logicalUrl: artifact.logicalUrl,
    file: artifact.file,
    kind: artifact.kind,
    contentType: artifact.contentType,
    sourceManifest: artifact.sourceManifest,
    status: artifact.status,
    localPath: artifact.localPath,
    raw,
    representations,
    url: representations[0].url,
    bytes: raw.bytes,
    sha256: raw.sha256,
  };
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
    if (Array.isArray(manifest.files)) {
      for (const entry of manifest.files) {
        const artifact = artifactFromTvmjsFile(entry, manifestPath, manifestPath, args);
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

async function sourceManifestArtifact(sourceManifest, args) {
  const absolute = join(args.root, sourceManifest);
  const digest = await localFileDigest(absolute);
  if (!digest) throw new Error(`Missing source manifest: ${sourceManifest}`);
  const file = basename(sourceManifest);
  return {
    logicalUrl: logicalUrlFromPublicPath(sourceManifest),
    sha256: digest.sha256,
    bytes: digest.bytes,
    file,
    kind: 'manifest',
    contentType: 'application/json',
    sourceManifest,
    localPath: sourceManifest,
  };
}

async function writeMutableOrCheck(path, value, check) {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  if (check) {
    const existing = await readFile(path, 'utf8');
    if (existing !== text) throw new Error(`${path} is stale; rerun write_artifact_release_manifests.mjs`);
    return;
  }
  await mkdir(dirname(path), { recursive: true });
  const tempPath = `${path}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`;
  try {
    await writeFile(tempPath, text, { flag: 'wx' });
    await rename(tempPath, path);
  } finally {
    await rm(tempPath, { force: true });
  }
}

async function writeReleaseOnce(path, value, check) {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  if (existsSync(path)) {
    const existing = await readFile(path, 'utf8');
    if (existing !== text) throw new Error(`Refusing to overwrite immutable release manifest: ${path}`);
    return;
  }
  if (check) throw new Error(`Missing immutable release manifest: ${path}`);
  await mkdir(dirname(path), { recursive: true });
  try {
    await writeFile(path, text, { flag: 'wx' });
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
    const existing = await readFile(path, 'utf8');
    if (existing !== text) throw new Error(`Refusing to overwrite immutable release manifest: ${path}`);
  }
}

function logicalIdentity(artifact) {
  return artifact.logicalUrl ?? artifact.name;
}

async function reusableGeneratedAt(path, args) {
  if (args.generatedAt || !existsSync(path)) return args.generatedAt;
  let existing;
  try {
    existing = await readJson(path);
  } catch {
    throw new Error(`Existing immutable release manifest is not valid JSON: ${path}`);
  }
  if (existing.schema !== ARTIFACT_RELEASE_V2_SCHEMAS[0]
    || existing.releaseId !== args.releaseId
    || existing.channel !== args.channel
    || existing.immutable !== true
    || existing.representationKeyIncludesEncoding !== true
    || existing.integrityIdentity !== 'decoded-sha256') {
    throw new Error(`Existing immutable release identity does not match requested release: ${path}`);
  }
  if (!isIsoTimestamp(existing.generatedAt)) {
    throw new Error(`Existing immutable release generatedAt is not a valid ISO timestamp: ${path}`);
  }
  return existing.generatedAt;
}

async function main() {
  const args = parseArgs(process.argv);
  const releasePath = join(args.outDir, 'releases', `${args.releaseId}.json`);
  const channelPath = join(args.outDir, 'channels', `${args.channel}.json`);
  const generatedAt = await reusableGeneratedAt(releasePath, args) ?? new Date().toISOString();
  const collected = await collectArtifacts(args);
  const updatedSourceManifests = collected.sourceManifests;
  const localArtifacts = [...collected.artifacts];
  let sourceManifests = updatedSourceManifests;
  let baseReleaseId;
  let inheritedArtifacts = [];
  if (args.baseRelease) {
    const basePath = args.baseRelease.startsWith('/') ? args.baseRelease : join(args.root, args.baseRelease);
    const base = await readJson(basePath);
    if (base.schema !== ARTIFACT_RELEASE_V1_SCHEMA && !isArtifactReleaseV2(base)) {
      throw new Error(`Unexpected base release schema: ${base.schema}`);
    }
    if (base.schema === ARTIFACT_RELEASE_V1_SCHEMA && (base.artifacts ?? []).length) {
      throw new Error('Cannot carry v1 artifacts into a v2 release without identity representations');
    }
    if (isArtifactReleaseV2(base)) releaseCatalogEntries(base);
    baseReleaseId = base.releaseId;
    inheritedArtifacts = (base.artifacts ?? []).map((artifact) => ({ ...artifact, carriedForwardFrom: base.releaseId }));
    sourceManifests = [...new Set([...(base.sourceManifests ?? []), ...sourceManifests])];
  }
  // Newly supplied source manifests are release artifacts too. Inherited
  // manifests retain their base release entries instead of being rebuilt from
  // potentially changed or unavailable local files.
  for (const sourceManifest of updatedSourceManifests) {
    localArtifacts.push(await sourceManifestArtifact(sourceManifest, args));
  }
  const materializedByRaw = new Map();
  const materializedArtifacts = [];
  for (const artifact of localArtifacts) {
    let materialized = materializedByRaw.get(artifact.sha256);
    if (!materialized) {
      materialized = await materializeRepresentations(artifact, args);
      materializedByRaw.set(artifact.sha256, materialized);
    }
    materializedArtifacts.push({
      ...materialized,
      logicalUrl: artifact.logicalUrl,
      file: artifact.file,
      kind: artifact.kind,
      contentType: artifact.contentType,
      sourceManifest: artifact.sourceManifest,
      status: artifact.status,
      localPath: artifact.localPath,
    });
  }
  const merged = new Map(inheritedArtifacts.map((artifact) => [logicalIdentity(artifact), artifact]));
  for (const artifact of materializedArtifacts) merged.set(logicalIdentity(artifact), artifact);
  const artifacts = [...merged.values()].sort((a, b) => logicalIdentity(a).localeCompare(logicalIdentity(b)));
  const releaseManifestUrl = `/releases/${args.releaseId}.json`;
  const release = {
    schema: ARTIFACT_RELEASE_V2_SCHEMAS[0],
    releaseId: args.releaseId,
    generatedAt,
    channel: args.channel,
    immutable: true,
    representationKeyIncludesEncoding: true,
    integrityIdentity: 'decoded-sha256',
    ...(baseReleaseId ? { baseReleaseId } : {}),
    sourceManifests,
    artifacts,
  };
  const channel = {
    schema: 'lc0_browser.artifact_channel_manifest.v2',
    channel: args.channel,
    releaseId: args.releaseId,
    releaseManifestUrl,
    releaseUrl: releaseManifestUrl,
    generatedAt,
  };

  releaseCatalogEntries(release);
  await writeReleaseOnce(releasePath, release, args.check);
  await writeMutableOrCheck(channelPath, channel, args.check);
  console.log(JSON.stringify({
    ok: true,
    releasePath,
    channelPath,
    artifactCount: artifacts.length,
    uniqueBodyCount: materializedByRaw.size,
    releaseSchema: release.schema,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
