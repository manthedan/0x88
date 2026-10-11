#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isArtifactChannelManifest, releaseCatalogEntries } from './engine_artifact_registry.mjs';
import { parseScriptArgs } from './lib/cli.mjs';

const DEFAULT_ROOT = '../models/lc0-bestnets';
const ASSET_ORIGIN = 'https://assets.0x88.app';
const PINNED_MANIFEST_URL = new URL('../public/models/lc0/manifest.json', import.meta.url);
const COMMITTED_CHANNEL_URL = new URL('../public/channels/stable.json', import.meta.url);
// Required parity fixtures. Integrity comes from the committed model manifest,
// not from whatever the live stable channel currently points at.
export const REQUIRED_ASSETS = Object.freeze([
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.onnx',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch4.f16.onnx',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.onnx',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.000.bin',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.001.bin',
  '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.002.bin',
]);

const USAGE = `Usage: node scripts/stage_lc0_test_assets.mjs [options]

Verify locally pinned LC0 fixtures by the committed manifest hash first.
A complete local cache does not touch the network. Missing assets are fetched
from the commit-pinned release catalog (public/channels/stable.json →
public/releases/<id>.json) and accepted only after the bytes match that hash.

Options:
  --root DIR         LC0 model workspace root (default ${DEFAULT_ROOT})
  --manifest PATH    Pinned integrity manifest (default public/models/lc0/manifest.json)
  --required LIST    Comma-separated logical URLs to stage (default: the parity set)
  --release PATH     Local release catalog instead of the committed channel pin
  --channel-url URL  Fetch a channel manifest instead of the committed pin. Ignored when the local cache is complete
  --check            Verify assets without downloading
  --offline          Refuse every fetch. Missing or corrupt assets fail
  -h, --help         Show this help
`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      root: { type: 'string', default: DEFAULT_ROOT },
      manifest: { type: 'string' },
      required: { type: 'string' },
      release: { type: 'string' },
      'channel-url': { type: 'string' },
      check: { type: 'boolean', default: false },
      offline: { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  if (args.required !== undefined) {
    args.required = args.required
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean);
    if (!args.required.length) throw new Error('--required must list at least one logical URL');
  }
  return args;
}

async function fetchJson(url, label) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`${label} fetch failed: ${response.status} ${response.statusText}`);
  return response.json();
}

function targetPath(root, logicalUrl) {
  const prefix = '/models/lc0/';
  if (!logicalUrl.startsWith(prefix)) throw new Error(`Unexpected LC0 logical URL: ${logicalUrl}`);
  const relative = logicalUrl.slice(prefix.length);
  if (relative.includes('.lc0web/')) return join(root, 'lc0web', relative);
  return join(root, 'onnx', relative);
}

export async function pinnedAssetMetadata(manifestUrl = PINNED_MANIFEST_URL, required = REQUIRED_ASSETS) {
  const manifest = JSON.parse(await readFile(manifestUrl, 'utf8'));
  const pinned = new Map();
  for (const model of manifest.models ?? []) pinned.set(model.url, { bytes: model.bytes, sha256: model.sha256?.toLowerCase() });
  for (const pack of manifest.packs ?? []) {
    pinned.set(pack.url, { bytes: pack.metadataBytes, sha256: pack.metadataSha256?.toLowerCase() });
    const prefix = pack.url.slice(0, pack.url.lastIndexOf('/') + 1);
    for (const shard of pack.shards ?? []) pinned.set(`${prefix}${shard.file}`, { bytes: shard.bytes, sha256: shard.sha256?.toLowerCase() });
  }
  for (const logicalUrl of required) {
    const expected = pinned.get(logicalUrl);
    if (!Number.isFinite(expected?.bytes) || !/^[a-f0-9]{64}$/.test(expected?.sha256 ?? '')) {
      throw new Error(`Committed LC0 manifest has invalid integrity metadata for ${logicalUrl}`);
    }
  }
  return pinned;
}

function assertPublishedIntegrity(logicalUrl, published, expected) {
  if (published.bytes !== expected.bytes || published.sha256 !== expected.sha256) {
    throw new Error(
      `Stable artifact release integrity does not match the committed LC0 manifest for ${logicalUrl}: committed ${expected.bytes}/${expected.sha256}, published ${published.bytes}/${published.sha256}`,
    );
  }
}

function identityMap(release) {
  return new Map(
    releaseCatalogEntries(release)
      .filter((entry) => entry.encoding === 'identity')
      .map((entry) => [entry.logicalUrl, entry]),
  );
}

function publishedIdentity(entry, logicalUrl) {
  const published = {
    bytes: entry.artifact.raw?.bytes ?? entry.artifact.bytes,
    sha256: (entry.artifact.raw?.sha256 ?? entry.artifact.sha256)?.toLowerCase(),
  };
  if (!Number.isFinite(published.bytes) || !/^[a-f0-9]{64}$/.test(published.sha256 ?? '')) {
    throw new Error(`Stable artifact release has invalid integrity metadata for ${logicalUrl}`);
  }
  return published;
}

/** Absolute http(s) URLs win. Root-relative identity paths use the catalog's http(s) origin, or the public asset origin when the catalog is a local file. */
export function resolveArtifactDownloadUrl(rawUrl, releaseBase) {
  if (!rawUrl) throw new Error('Artifact release entry has no download URL');
  const httpBase = releaseBase && (releaseBase.protocol === 'http:' || releaseBase.protocol === 'https:') ? releaseBase : ASSET_ORIGIN;
  let url;
  try {
    url = new URL(rawUrl, httpBase);
  } catch {
    throw new Error(`Artifact download URL is not usable: ${rawUrl}`);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error(`Artifact download URL must be http(s): ${url.href}`);
  if (!url.pathname.replace(/^\/+/, '').startsWith('artifacts/sha256/')) {
    throw new Error(`Artifact download URL is not an identity object: ${url.href}`);
  }
  return url;
}

async function loadCommitPinnedRelease() {
  const channel = JSON.parse(await readFile(COMMITTED_CHANNEL_URL, 'utf8'));
  if (!isArtifactChannelManifest(channel)) throw new Error(`Unexpected committed artifact channel schema: ${channel?.schema}`);
  const releasePath = channel.releaseManifestUrl || channel.releaseUrl;
  if (!releasePath?.startsWith('/releases/')) throw new Error('Committed channel does not point at an immutable /releases manifest');
  const releaseUrl = new URL(`../public${releasePath}`, import.meta.url);
  const release = JSON.parse(await readFile(releaseUrl, 'utf8'));
  return { release, releaseBase: releaseUrl, releaseId: release.releaseId ?? channel.releaseId ?? null };
}

async function loadCatalog(args) {
  if (args.release) {
    const releasePath = resolve(args.release);
    const release = JSON.parse(await readFile(releasePath, 'utf8'));
    return { release, releaseBase: pathToFileURL(releasePath), releaseId: release.releaseId ?? null };
  }
  if (args.channelUrl) {
    const channelUrl = new URL(args.channelUrl);
    const channel = await fetchJson(channelUrl, 'Artifact channel');
    if (!isArtifactChannelManifest(channel)) throw new Error(`Unexpected artifact channel schema: ${channel?.schema}`);
    const releasePath = channel.releaseManifestUrl || channel.releaseUrl;
    if (!releasePath) throw new Error('Artifact channel has no release manifest URL');
    const releaseUrl = new URL(releasePath, channelUrl);
    const release = await fetchJson(releaseUrl, 'Artifact release');
    return { release, releaseBase: releaseUrl, releaseId: release.releaseId ?? null };
  }
  return loadCommitPinnedRelease();
}

async function checksum(filePath) {
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  return { bytes, sha256: hash.digest('hex') };
}

async function existingAssetMatches(filePath, expected) {
  try {
    const info = await stat(filePath);
    if (!info.isFile() || info.size !== expected.bytes) return false;
    const observed = await checksum(filePath);
    return observed.bytes === expected.bytes && observed.sha256 === expected.sha256;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function downloadAsset(url, filePath, expected) {
  await mkdir(dirname(filePath), { recursive: true });
  const temporary = `${filePath}.partial-${process.pid}`;
  await rm(temporary, { force: true });
  const response = await fetch(url, { headers: { 'Accept-Encoding': 'identity' } });
  if (!response.ok || !response.body) throw new Error(`Artifact fetch failed for ${url}: ${response.status} ${response.statusText}`);
  const hash = createHash('sha256');
  let bytes = 0;
  const verifier = new Transform({
    transform(chunk, _encoding, callback) {
      hash.update(chunk);
      bytes += chunk.byteLength;
      callback(null, chunk);
    },
  });
  try {
    await pipeline(Readable.fromWeb(response.body), verifier, createWriteStream(temporary, { flags: 'wx' }));
    const sha256 = hash.digest('hex');
    if (bytes !== expected.bytes || sha256 !== expected.sha256) {
      throw new Error(`Artifact integrity mismatch for ${basename(filePath)}: expected ${expected.bytes}/${expected.sha256}, received ${bytes}/${sha256}`);
    }
    await rename(temporary, filePath);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const required = args.required ?? REQUIRED_ASSETS;
  const manifestUrl = args.manifest ? pathToFileURL(resolve(args.manifest)) : PINNED_MANIFEST_URL;
  const pinnedAssets = await pinnedAssetMetadata(manifestUrl, required);
  const root = resolve(args.root);
  const staged = [];
  const missing = [];
  for (const logicalUrl of required) {
    const expected = pinnedAssets.get(logicalUrl);
    const filePath = targetPath(root, logicalUrl);
    if (await existingAssetMatches(filePath, expected)) staged.push({ logicalUrl, path: filePath, status: 'verified', ...expected });
    else missing.push({ logicalUrl, path: filePath, expected });
  }
  // Hash-verified local bytes are the pin. Do not consult a channel, even a
  // caller-supplied one, once every required asset has matched.
  if (!missing.length) {
    console.log(JSON.stringify({ ok: true, network: false, releaseId: null, root, staged }, null, 2));
    return;
  }
  if (args.offline) {
    throw new Error(`Required LC0 test asset is missing or corrupt: ${missing.map((item) => item.path).join(', ')} (offline; refusing to fetch)`);
  }
  const catalog = await loadCatalog(args);
  const identityEntries = identityMap(catalog.release);
  const downloads = [];
  for (const logicalUrl of required) {
    const entry = identityEntries.get(logicalUrl);
    if (!entry) throw new Error(`Stable artifact release is missing ${logicalUrl}`);
    const published = publishedIdentity(entry, logicalUrl);
    assertPublishedIntegrity(logicalUrl, published, pinnedAssets.get(logicalUrl));
    const rawUrl = entry.representation?.url ?? entry.artifact.artifactUrl ?? `/${entry.key}`;
    downloads.push(resolveArtifactDownloadUrl(rawUrl, catalog.releaseBase));
  }
  if (args.check) throw new Error(`Required LC0 test asset is missing or corrupt: ${missing.map((item) => item.path).join(', ')}`);
  for (const item of missing) {
    const index = required.indexOf(item.logicalUrl);
    await downloadAsset(downloads[index], item.path, item.expected);
    staged.push({ logicalUrl: item.logicalUrl, path: item.path, status: 'downloaded', ...item.expected });
  }
  console.log(JSON.stringify({ ok: true, network: true, releaseId: catalog.releaseId, root, staged }, null, 2));
}

const invokedDirectly = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
