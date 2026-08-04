#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { isArtifactChannelManifest, releaseCatalogEntries } from './engine_artifact_registry.mjs';
import { parseScriptArgs } from './lib/cli.mjs';

const DEFAULT_CHANNEL_URL = 'https://assets.0x88.app/channels/stable.json';
const DEFAULT_ROOT = '../models/lc0-bestnets';
const PINNED_MANIFEST_URL = new URL('../public/models/lc0/manifest.json', import.meta.url);
const REQUIRED_ASSETS = Object.freeze([
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

Options:
  --channel-url URL  Artifact channel manifest (default ${DEFAULT_CHANNEL_URL})
  --root DIR         LC0 model workspace root (default ${DEFAULT_ROOT})
  --check            Verify assets without downloading
  -h, --help         Show this help
`;

function parseArgs(argv) {
  return parseScriptArgs(argv, {
    options: {
      'channel-url': { type: 'string', default: DEFAULT_CHANNEL_URL },
      root: { type: 'string', default: DEFAULT_ROOT },
      check: { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
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
async function pinnedAssetMetadata() {
  const manifest = JSON.parse(await readFile(PINNED_MANIFEST_URL, 'utf8'));
  const pinned = new Map();
  for (const model of manifest.models ?? []) pinned.set(model.url, { bytes: model.bytes, sha256: model.sha256?.toLowerCase() });
  for (const pack of manifest.packs ?? []) {
    pinned.set(pack.url, { bytes: pack.metadataBytes, sha256: pack.metadataSha256?.toLowerCase() });
    const prefix = pack.url.slice(0, pack.url.lastIndexOf('/') + 1);
    for (const shard of pack.shards ?? []) pinned.set(`${prefix}${shard.file}`, { bytes: shard.bytes, sha256: shard.sha256?.toLowerCase() });
  }
  for (const logicalUrl of REQUIRED_ASSETS) {
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

async function checksum(path) {
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  return { bytes, sha256: hash.digest('hex') };
}

async function existingAssetMatches(path, expected) {
  try {
    const info = await stat(path);
    if (!info.isFile() || info.size !== expected.bytes) return false;
    const observed = await checksum(path);
    return observed.bytes === expected.bytes && observed.sha256 === expected.sha256;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function downloadAsset(url, path, expected) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.partial-${process.pid}`;
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
      throw new Error(`Artifact integrity mismatch for ${basename(path)}: expected ${expected.bytes}/${expected.sha256}, received ${bytes}/${sha256}`);
    }
    await rename(temporary, path);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const channelUrl = new URL(args.channelUrl);
  const channel = await fetchJson(channelUrl, 'Artifact channel');
  if (!isArtifactChannelManifest(channel)) throw new Error(`Unexpected artifact channel schema: ${channel?.schema}`);
  const releasePath = channel.releaseManifestUrl || channel.releaseUrl;
  if (!releasePath) throw new Error('Artifact channel has no release manifest URL');
  const releaseUrl = new URL(releasePath, channelUrl);
  const release = await fetchJson(releaseUrl, 'Artifact release');
  const identityEntries = new Map(
    releaseCatalogEntries(release)
      .filter((entry) => entry.encoding === 'identity')
      .map((entry) => [entry.logicalUrl, entry]),
  );
  const pinnedAssets = await pinnedAssetMetadata();
  const root = resolve(args.root);
  const staged = [];
  for (const logicalUrl of REQUIRED_ASSETS) {
    const entry = identityEntries.get(logicalUrl);
    if (!entry) throw new Error(`Stable artifact release is missing ${logicalUrl}`);
    const published = {
      bytes: entry.artifact.raw?.bytes ?? entry.artifact.bytes,
      sha256: (entry.artifact.raw?.sha256 ?? entry.artifact.sha256)?.toLowerCase(),
    };
    if (!Number.isFinite(published.bytes) || !/^[a-f0-9]{64}$/.test(published.sha256 ?? '')) {
      throw new Error(`Stable artifact release has invalid integrity metadata for ${logicalUrl}`);
    }
    const expected = pinnedAssets.get(logicalUrl);
    assertPublishedIntegrity(logicalUrl, published, expected);
    const path = targetPath(root, logicalUrl);
    if (await existingAssetMatches(path, expected)) {
      staged.push({ logicalUrl, path, status: 'verified', ...expected });
      continue;
    }
    if (args.check) throw new Error(`Required LC0 test asset is missing or corrupt: ${path}`);
    const representationUrl = new URL(entry.representation?.url ?? entry.artifact.artifactUrl ?? `/${entry.key}`, releaseUrl);
    await downloadAsset(representationUrl, path, expected);
    staged.push({ logicalUrl, path, status: 'downloaded', ...expected });
  }
  console.log(JSON.stringify({ ok: true, releaseId: release.releaseId, root, staged }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
