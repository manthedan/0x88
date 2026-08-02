#!/usr/bin/env node
/**
 * Materialize a write-once, representation-aware artifact tree and immutable
 * release/mutable channel manifests. Bodies are keyed only by content identity;
 * logical filenames remain release-manifest metadata.
 *
 * Example:
 *   node scripts/publish_content_addressed_release.mjs \
 *     --root .release-public --release-id 2026-07-14.abc123 \
 *     --channel stable --asset lc0-small=/path/to/model.onnx
 */
import { createHash, randomBytes } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { createBrotliCompress, constants as zlibConstants } from 'node:zlib';

function usage(message) {
  if (message) console.error(message);
  console.error(
    'Usage: publish_content_addressed_release.mjs --release-id ID --asset NAME=PATH [--asset NAME=PATH ...] [--root DIR] [--channel NAME] [--no-brotli] [--brotli-quality 0..11]',
  );
  process.exit(2);
}

const args = process.argv.slice(2);
const options = { root: 'public', channel: 'stable', assets: [], brotli: true, brotliQuality: 5 };
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  const value = args[i + 1];
  if (arg === '--root' && value) {
    options.root = value;
    i++;
  } else if (arg === '--release-id' && value) {
    options.releaseId = value;
    i++;
  } else if (arg === '--channel' && value) {
    options.channel = value;
    i++;
  } else if (arg === '--asset' && value) {
    options.assets.push(value);
    i++;
  } else if (arg === '--no-brotli') options.brotli = false;
  else if (arg === '--brotli-quality' && value) {
    options.brotliQuality = Number(value);
    i++;
  } else usage(`Unknown or incomplete argument: ${arg}`);
}
if (!options.releaseId || !/^[A-Za-z0-9._-]+$/.test(options.releaseId)) usage('--release-id is required and must be path-safe');
if (!/^[A-Za-z0-9._-]+$/.test(options.channel)) usage('--channel must be path-safe');
if (!options.assets.length) usage('At least one --asset NAME=PATH is required');
if (!Number.isInteger(options.brotliQuality) || options.brotliQuality < 0 || options.brotliQuality > 11)
  usage('--brotli-quality must be an integer from 0 through 11');

function parseAsset(spec) {
  const equals = spec.indexOf('=');
  if (equals <= 0 || equals === spec.length - 1) usage(`Invalid --asset ${spec}; expected NAME=PATH`);
  const name = spec.slice(0, equals);
  const path = resolve(spec.slice(equals + 1));
  if (!/^[A-Za-z0-9._-]+$/.test(name)) usage(`Asset name is not path-safe: ${name}`);
  if (!existsSync(path)) usage(`Asset does not exist: ${path}`);
  return { name, path };
}

function contentType(path) {
  if (path.endsWith('.wasm')) return 'application/wasm';
  if (path.endsWith('.json')) return 'application/json; charset=utf-8';
  if (path.endsWith('.js') || path.endsWith('.mjs')) return 'text/javascript; charset=utf-8';
  if (path.endsWith('.css')) return 'text/css; charset=utf-8';
  return 'application/octet-stream';
}

async function hashFile(path) {
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  return { sha256: hash.digest('hex'), bytes };
}

async function installWriteOnce(source, target, expected) {
  if (existsSync(target)) {
    const actual = await hashFile(target);
    if (actual.sha256 !== expected.sha256 || actual.bytes !== expected.bytes) {
      throw new Error(`Content-addressed key collision or corruption at ${target}`);
    }
    return;
  }
  mkdirSync(dirname(target), { recursive: true });
  await pipeline(createReadStream(source), createWriteStream(target, { flags: 'wx' }));
}

function writeJsonOnce(path, value) {
  const encoded = `${JSON.stringify(value, null, 2)}\n`;
  if (existsSync(path)) {
    if (readFileSync(path, 'utf8') !== encoded) throw new Error(`Refusing to overwrite immutable release manifest: ${path}`);
    return;
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, encoded, { flag: 'wx' });
}

const root = resolve(options.root);
const tempRoot = resolve(root, `.artifact-publish-${process.pid}-${randomBytes(6).toString('hex')}`);
const artifacts = [];
try {
  for (const spec of options.assets) {
    const asset = parseAsset(spec);
    const raw = await hashFile(asset.path);
    const identityUrl = `/artifacts/sha256/${raw.sha256}/identity`;
    await installWriteOnce(asset.path, resolve(root, identityUrl.slice(1)), raw);
    const representations = [{ encoding: 'identity', url: identityUrl, sha256: raw.sha256, bytes: raw.bytes }];

    if (options.brotli) {
      mkdirSync(tempRoot, { recursive: true });
      const compressedTemp = resolve(tempRoot, `${artifacts.length}.br`);
      await pipeline(
        createReadStream(asset.path),
        createBrotliCompress({ params: { [zlibConstants.BROTLI_PARAM_QUALITY]: options.brotliQuality } }),
        createWriteStream(compressedTemp, { flags: 'wx' }),
      );
      const encoded = await hashFile(compressedTemp);
      const brUrl = `/artifacts/sha256/${raw.sha256}/br/${encoded.sha256}`;
      await installWriteOnce(compressedTemp, resolve(root, brUrl.slice(1)), encoded);
      representations.push({ encoding: 'br', url: brUrl, sha256: encoded.sha256, bytes: encoded.bytes });
    }

    artifacts.push({
      name: asset.name,
      file: basename(asset.path),
      contentType: contentType(asset.path),
      raw: { sha256: raw.sha256, bytes: raw.bytes },
      representations,
      // Compatibility fields for consumers that have not adopted v2 selection.
      url: identityUrl,
      bytes: raw.bytes,
      sha256: raw.sha256,
    });
  }
} finally {
  rmSync(tempRoot, { recursive: true, force: true });
}
artifacts.sort((a, b) => a.name.localeCompare(b.name));

const releaseUrl = `/releases/${options.releaseId}.json`;
const release = {
  schema: 'lc0-webgpu.artifact-release.v2',
  releaseId: options.releaseId,
  immutable: true,
  representationKeyIncludesEncoding: true,
  integrityIdentity: 'decoded-sha256',
  artifacts,
};
writeJsonOnce(resolve(root, releaseUrl.slice(1)), release);

const channelPath = resolve(root, 'channels', `${options.channel}.json`);
mkdirSync(dirname(channelPath), { recursive: true });
writeFileSync(
  channelPath,
  `${JSON.stringify(
    {
      schema: 'lc0-webgpu.artifact-channel.v2',
      channel: options.channel,
      releaseId: options.releaseId,
      releaseUrl,
    },
    null,
    2,
  )}\n`,
);

console.log(JSON.stringify({ status: 'CONTENT_ADDRESSED_RELEASE_V2_READY', root, releaseUrl, channel: options.channel, artifacts }, null, 2));
