#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { createReadStream } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { basename, isAbsolute, join } from 'node:path';
import {
  ARTIFACT_RELEASE_V1_SCHEMA,
  ARTIFACT_RELEASE_V2_SCHEMAS,
  buildArtifactReleaseCatalog,
  isArtifactChannelManifest,
} from './engine_artifact_registry.mjs';

const DEFAULT_ARTIFACT_BASE = 'https://assets.0x88.app';

function usage() {
  console.log(`Usage: node scripts/publish_hashed_artifacts_to_r2.mjs --release .local-dev-artifacts/artifact-releases/releases/ID.json --bucket BUCKET [options]\n\nOptions:\n  --root DIR          Repository or materialized release root (default .)\n  --execute           Actually call wrangler/AWS CLI; default is dry-run\n  --allow-missing     Skip artifacts whose localPath is absent\n  --wrangler-bin BIN  Wrangler binary (default wrangler)\n  --aws-bin BIN       AWS CLI binary used for atomic immutable object creation (default aws)\n  --r2-endpoint URL   R2 S3 endpoint (or R2_ENDPOINT / R2_ACCOUNT_ID)\n  --channel-manifest PATH  Optional generated channel manifest to publish after the release\n  --artifact-base URL Public artifact origin used to probe relative representation URLs (default https://assets.0x88.app)\n  --probe-existing    In dry-run mode, validate representation URLs and mark existing uploads as skipped\n  -h, --help          Show help\n\nBoth legacy v1 releases and representation-aware v2 releases are accepted. V2 identity\nobjects use artifacts/sha256/<decoded-sha256>/identity; Brotli objects use\nartifacts/sha256/<decoded-sha256>/br/<encoded-sha256>. Existing v2 bodies are\nvalidated with immutable HEAD metadata plus decoded full-body integrity until trusted\nR2 verification metadata is available. Legacy filename-keyed bodies retain v1 checks. Immutable bodies and release manifests are\ncreated atomically with S3 If-None-Match and channel pointers are published last.\n`);
}

function parseArgs(argv) {
  const accountId = process.env.R2_ACCOUNT_ID ?? process.env.CLOUDFLARE_ACCOUNT_ID;
  const args = {
    root: '.',
    execute: false,
    allowMissing: false,
    wranglerBin: 'wrangler',
    awsBin: 'aws',
    r2Endpoint: process.env.R2_ENDPOINT ?? (accountId ? `https://${accountId}.r2.cloudflarestorage.com` : undefined),
    probeExisting: false,
    artifactBase: process.env.LC0_ARTIFACT_BASE_URL ?? DEFAULT_ARTIFACT_BASE,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--root' && next) { args.root = next; i += 1; continue; }
    if (arg === '--release' && next) { args.release = next; i += 1; continue; }
    if (arg === '--bucket' && next) { args.bucket = next; i += 1; continue; }
    if (arg === '--wrangler-bin' && next) { args.wranglerBin = next; i += 1; continue; }
    if (arg === '--aws-bin' && next) { args.awsBin = next; i += 1; continue; }
    if (arg === '--r2-endpoint' && next) { args.r2Endpoint = next; i += 1; continue; }
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
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  return { bytes, sha256: hash.digest('hex') };
}

function artifactKeyFromUrl(rawUrl) {
  const parsed = new URL(rawUrl, 'https://assets.invalid');
  const key = parsed.pathname.replace(/^\/+/, '');
  if (!key.startsWith('artifacts/sha256/')) throw new Error(`Artifact URL is not content-addressed: ${rawUrl}`);
  return key;
}

function publicUrl(args, rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol === 'https:' || parsed.protocol === 'http:') return parsed.href;
  } catch {
    // Resolve relative URLs below.
  }
  if (!args.artifactBase) return undefined;
  return new URL(rawUrl, args.artifactBase).href;
}

function localPathFor(args, candidate, key) {
  if (candidate) return isAbsolute(candidate) ? candidate : join(args.root, candidate);
  return join(args.root, key);
}

function entryFromLegacyArtifact(artifact, args) {
  const key = artifactKeyFromUrl(artifact.artifactUrl);
  const match = key.match(/^artifacts\/sha256\/([a-f0-9]{64})\//);
  const sha256 = artifact.sha256?.toLowerCase();
  if (match?.[1] !== sha256) {
    throw new Error(`Content-addressed key mismatch for ${artifact.logicalUrl}: key has ${match?.[1] ?? 'no sha256'}, manifest has ${artifact.sha256}`);
  }
  return {
    logicalUrls: [artifact.logicalUrl],
    key,
    localPath: localPathFor(args, artifact.localPath, key),
    carriedForward: Boolean(artifact.carriedForwardFrom),
    bytes: artifact.bytes,
    sha256,
    decodedBytes: artifact.bytes,
    decodedSha256: sha256,
    contentType: artifact.contentType ?? 'application/octet-stream',
    contentEncoding: undefined,
    url: publicUrl(args, artifact.artifactUrl),
    verification: 'legacy-full-body',
  };
}

function entriesFromV1Release(release, args) {
  return (release.artifacts ?? []).map((artifact) => entryFromLegacyArtifact(artifact, args));
}

function validateV2Representation(artifact, representation) {
  const rawSha256 = artifact.raw?.sha256?.toLowerCase();
  const rawBytes = artifact.raw?.bytes;
  if (!rawSha256 || !Number.isFinite(rawBytes)) throw new Error(`Invalid v2 raw metadata for ${artifact.logicalUrl ?? artifact.name}`);
  const key = artifactKeyFromUrl(representation.url);
  const encodedSha256 = representation.sha256?.toLowerCase();
  if (!encodedSha256 || !Number.isFinite(representation.bytes)) throw new Error(`Invalid v2 representation metadata for ${artifact.logicalUrl ?? artifact.name}`);
  if (representation.encoding === 'identity') {
    if (key !== `artifacts/sha256/${rawSha256}/identity` || encodedSha256 !== rawSha256 || representation.bytes !== rawBytes) {
      throw new Error(`Invalid identity representation for ${artifact.logicalUrl ?? artifact.name}`);
    }
  } else if (representation.encoding === 'br') {
    if (key !== `artifacts/sha256/${rawSha256}/br/${encodedSha256}`) {
      throw new Error(`Invalid Brotli representation key for ${artifact.logicalUrl ?? artifact.name}`);
    }
  } else {
    throw new Error(`Unsupported artifact encoding: ${representation.encoding}`);
  }
  return { key, rawSha256, rawBytes, encodedSha256 };
}

function entriesFromV2Release(release, args) {
  const byKey = new Map();
  for (const artifact of release.artifacts ?? []) {
    if (!Array.isArray(artifact.representations) || !artifact.representations.length) {
      if (artifact.artifactUrl) {
        const legacy = entryFromLegacyArtifact(artifact, args);
        const existing = byKey.get(legacy.key);
        if (existing) {
          if (!existing.logicalUrls.includes(artifact.logicalUrl)) existing.logicalUrls.push(artifact.logicalUrl);
          continue;
        }
        byKey.set(legacy.key, legacy);
        continue;
      }
      throw new Error(`V2 artifact has no representations: ${artifact.logicalUrl ?? artifact.name}`);
    }
    for (const representation of artifact.representations) {
      const validated = validateV2Representation(artifact, representation);
      const logicalUrl = artifact.logicalUrl ?? artifact.name;
      const carriedForward = Boolean(artifact.carriedForwardFrom);
      const localPath = localPathFor(args, representation.localPath ?? (representation.encoding === 'identity' ? artifact.localPath : undefined), validated.key);
      const existing = byKey.get(validated.key);
      if (existing) {
        if (existing.bytes !== representation.bytes || existing.sha256 !== validated.encodedSha256 || existing.contentEncoding !== (representation.encoding === 'identity' ? undefined : representation.encoding)) {
          throw new Error(`Conflicting v2 representation metadata for ${validated.key}`);
        }
        if (logicalUrl && !existing.logicalUrls.includes(logicalUrl)) existing.logicalUrls.push(logicalUrl);
        // Equal bodies may be inherited by one logical entry and materialized by
        // another. Any local source is sufficient to make the deduplicated body
        // uploadable when the remote object is absent.
        if (!carriedForward) {
          const availableLocalPath = [existing.localPath, localPath].find((candidate) => candidate && existsSync(candidate));
          existing.carriedForward = false;
          existing.localPath = availableLocalPath ?? existing.localPath ?? localPath;
        }
        continue;
      }
      byKey.set(validated.key, {
        logicalUrls: logicalUrl ? [logicalUrl] : [],
        key: validated.key,
        localPath,
        carriedForward,
        bytes: representation.bytes,
        sha256: validated.encodedSha256,
        decodedBytes: validated.rawBytes,
        decodedSha256: validated.rawSha256,
        contentType: artifact.contentType ?? 'application/octet-stream',
        contentEncoding: representation.encoding === 'identity' ? undefined : representation.encoding,
        url: publicUrl(args, representation.url),
        verification: 'v2-immutable-head',
      });
    }
  }
  return [...byKey.values()];
}

function releaseEntries(release, args) {
  if (release.schema === ARTIFACT_RELEASE_V1_SCHEMA) return entriesFromV1Release(release, args);
  if (ARTIFACT_RELEASE_V2_SCHEMAS.includes(release.schema)) return entriesFromV2Release(release, args);
  throw new Error(`Unexpected release schema: ${release.schema}`);
}

function headerValue(headers, name) {
  return headers.get(name) ?? headers.get(name.toLowerCase());
}

function positiveIntegerHeader(headers, name) {
  const raw = headerValue(headers, name);
  if (raw === null) return undefined;
  if (!/^[1-9]\d*$/.test(raw)) {
    throw new Error(`Remote artifact ${name} is not a positive integer: ${raw}`);
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`Remote artifact ${name} is not a safe positive integer: ${raw}`);
  }
  return { raw, value };
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
  const response = await fetch(url, { cache: 'no-cache', headers: { 'Accept-Encoding': 'identity' } });
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

async function probeExistingEntry(entry) {
  if (!entry.url) return { state: 'unchecked', reason: 'no public artifact URL; pass --artifact-base for relative representation URLs' };
  const response = await fetch(entry.url, { method: 'HEAD', cache: 'no-cache', headers: { 'Accept-Encoding': 'identity' } });
  if (response.status === 404) return { state: 'missing', url: entry.url, status: response.status };
  if (!response.ok) throw new Error(`Artifact probe failed for ${entry.url}: HTTP ${response.status}`);
  const encodedLengthMetadata = positiveIntegerHeader(response.headers, 'X-Artifact-Encoded-Length')
    ?? positiveIntegerHeader(response.headers, 'Content-Length')
    ?? (!entry.contentEncoding ? positiveIntegerHeader(response.headers, 'X-Artifact-Content-Length') : undefined);
  const encodedLength = encodedLengthMetadata?.value;
  if (encodedLength !== entry.bytes) {
    throw new Error(`Remote artifact size mismatch for ${entry.logicalUrls.join(', ')}: got ${encodedLengthMetadata?.raw ?? 'missing'}, expected ${entry.bytes}`);
  }
  const decodedLengthHeader = headerValue(response.headers, 'x-artifact-content-length');
  if (decodedLengthHeader !== null && Number(decodedLengthHeader) !== entry.decodedBytes) {
    throw new Error(`Remote decoded artifact size mismatch for ${entry.logicalUrls.join(', ')}: got ${decodedLengthHeader}, expected ${entry.decodedBytes}`);
  }
  const cacheControl = headerValue(response.headers, 'cache-control') ?? '';
  if (!cacheControlDirective(cacheControl, 'immutable') || !cacheControlDirective(cacheControl, 'max-age', '31536000')) {
    throw new Error(`Remote artifact cache policy is not immutable for ${entry.logicalUrls.join(', ')}: ${cacheControl || 'missing'}`);
  }
  if (entry.contentEncoding) {
    const actualEncoding = headerValue(response.headers, 'content-encoding');
    if (actualEncoding !== entry.contentEncoding) throw new Error(`Remote artifact encoding mismatch for ${entry.logicalUrls.join(', ')}: got ${actualEncoding ?? 'identity'}, expected ${entry.contentEncoding}`);
  }
  const actual = await sha256RemoteUrl(entry.url);
  const expectedBodyBytes = entry.verification === 'legacy-full-body' ? entry.bytes : entry.decodedBytes;
  const expectedBodySha256 = entry.verification === 'legacy-full-body' ? entry.sha256 : entry.decodedSha256;
  if (actual.bytes !== expectedBodyBytes) {
    throw new Error(`Remote artifact body size mismatch for ${entry.logicalUrls.join(', ')}: got ${actual.bytes}, expected ${expectedBodyBytes}`);
  }
  if (actual.sha256 !== expectedBodySha256) {
    throw new Error(`Remote artifact SHA-256 mismatch for ${entry.logicalUrls.join(', ')}: got ${actual.sha256}, expected ${expectedBodySha256}`);
  }
  return {
    state: 'existing',
    url: entry.url,
    status: response.status,
    bytes: encodedLength,
    sha256: actual.sha256,
    verification: entry.verification === 'legacy-full-body' ? 'full-body' : 'decoded-full-body',
  };
}

async function verifyRemoteImmutableObject(args, target, expected) {
  const dir = await mkdtemp(join(tmpdir(), 'lc0-r2-exists-'));
  try {
    const file = join(dir, 'object');
    const child = spawnSync(args.wranglerBin, ['r2', 'object', 'get', target, '--file', file, '--remote'], { encoding: 'utf8' });
    if (child.status !== 0) {
      const output = `${child.stdout ?? ''}\n${child.stderr ?? ''}`;
      if (/specified key does not exist/i.test(output)) return 'missing';
      throw new Error(`Unable to verify immutable object ${target}; refusing to upload`);
    }
    if (!existsSync(file)) {
      throw new Error(`Refusing to overwrite immutable object ${target}: existing object could not be verified`);
    }
    const actual = await sha256File(file);
    if (actual.bytes !== expected.bytes || actual.sha256 !== expected.sha256) {
      throw new Error(`Refusing to overwrite immutable object ${target}: remote content differs`);
    }
    return 'identical';
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function isConditionalCreateConflict(output) {
  return /PreconditionFailed|precondition(?: condition)? failed|status code:\s*412|HTTP\s*412/i.test(output);
}

async function createImmutableObjectAtomically(args, item) {
  if (!args.r2Endpoint) {
    throw new Error('Atomic immutable object creation requires --r2-endpoint, R2_ENDPOINT, R2_ACCOUNT_ID, or CLOUDFLARE_ACCOUNT_ID');
  }
  const target = `${args.bucket}/${item.key}`;
  const command = [
    's3api', 'put-object',
    '--bucket', args.bucket,
    '--key', item.key,
    '--body', item.localPath,
    '--content-type', item.contentType,
    '--cache-control', item.cacheControl,
    '--if-none-match', '*',
    '--endpoint-url', args.r2Endpoint,
    '--region', 'auto',
  ];
  if (item.contentEncoding) command.push('--content-encoding', item.contentEncoding);
  const child = spawnSync(args.awsBin, command, { encoding: 'utf8' });
  const output = `${child.stdout ?? ''}\n${child.stderr ?? ''}`;
  if (child.status !== 0 && !isConditionalCreateConflict(output)) {
    throw new Error(`Atomic immutable object create failed for ${target}: ${output.trim() || `exit ${child.status}`}`);
  }

  const remoteState = await verifyRemoteImmutableObject(args, target, item);
  if (remoteState !== 'identical') {
    throw new Error(`Atomic immutable object create verification failed for ${target}`);
  }
  return child.status === 0 ? 'created' : 'identical';
}

async function manifestPublishItems(args, release) {
  if (typeof release.releaseId !== 'string' || !/^[A-Za-z0-9._-]+$/.test(release.releaseId)) {
    throw new Error('release.releaseId must contain only letters, numbers, dots, underscores, and hyphens');
  }
  const releaseKey = `releases/${release.releaseId}.json`;
  const releaseDigest = await sha256File(args.release);
  const items = [{
    type: 'release-manifest',
    localPath: args.release,
    key: releaseKey,
    bytes: releaseDigest.bytes,
    sha256: releaseDigest.sha256,
    contentType: 'application/json; charset=utf-8',
    cacheControl: 'public, max-age=31536000, immutable',
    remoteState: 'not-checked',
    uploadAction: 'conditional-create-or-verify',
  }];
  if (args.channelManifest) {
    const channel = JSON.parse(await readFile(args.channelManifest, 'utf8'));
    if (!isArtifactChannelManifest(channel)) throw new Error(`Unexpected channel schema: ${channel.schema}`);
    const expectedChannelFilename = `${channel.channel}.json`;
    const actualChannelFilename = basename(args.channelManifest);
    if (actualChannelFilename !== expectedChannelFilename) {
      throw new Error(`Channel manifest filename ${actualChannelFilename} does not match channel.channel ${channel.channel}`);
    }
    if (channel.releaseId !== release.releaseId) throw new Error(`Channel releaseId ${channel.releaseId} does not match release ${release.releaseId}`);
    const releaseUrl = channel.releaseManifestUrl || channel.releaseUrl;
    if (releaseUrl !== `/${releaseKey}`) throw new Error(`Channel release URL ${releaseUrl} does not match /${releaseKey}`);
    items.push({
      type: 'channel-manifest',
      localPath: args.channelManifest,
      key: `channels/${expectedChannelFilename}`,
      contentType: 'application/json; charset=utf-8',
      cacheControl: 'no-cache',
      remoteState: 'mutable',
      uploadAction: 'update-last',
    });
  }
  return items;
}

async function main() {
  const args = parseArgs(process.argv);
  const release = JSON.parse(await readFile(args.release, 'utf8'));
  const entries = releaseEntries(release, args);
  const catalog = buildArtifactReleaseCatalog([release]);
  const plannedKeys = new Set(entries.map((entry) => entry.key));
  if (catalog.size !== plannedKeys.size || [...catalog.keys()].some((key) => !plannedKeys.has(key))) {
    throw new Error(`Release catalog mismatch: catalog has ${catalog.size} representation keys, publisher planned ${plannedKeys.size}`);
  }
  const planned = [];
  const skipped = [];

  for (const entry of entries) {
    const localExists = Boolean(!entry.carriedForward && entry.localPath && existsSync(entry.localPath));
    if (localExists) {
      const actual = await sha256File(entry.localPath);
      if (actual.bytes !== entry.bytes) throw new Error(`Size mismatch for ${entry.logicalUrls.join(', ')}: got ${actual.bytes}, expected ${entry.bytes}`);
      if (actual.sha256 !== entry.sha256) throw new Error(`SHA-256 mismatch for ${entry.logicalUrls.join(', ')}: got ${actual.sha256}, expected ${entry.sha256}`);
    } else if (!entry.carriedForward) {
      if (args.allowMissing) { skipped.push({ logicalUrls: entry.logicalUrls, reason: entry.localPath ? 'missing localPath' : 'no localPath', localPath: entry.localPath }); continue; }
      throw new Error(`Missing local artifact for ${entry.logicalUrls.join(', ')}: ${entry.localPath ?? 'no localPath'}`);
    }

    // HEAD verifies representation metadata and a decoded full-body pass proves
    // integrity. This can become HEAD-only once uploads persist a trustworthy R2
    // verification digest; a hash-shaped key alone is not proof of stored bytes.
    const probe = (args.probeExisting || args.execute || !localExists) ? await probeExistingEntry(entry) : undefined;
    if (!localExists && probe?.state !== 'existing') throw new Error(`Carried-forward artifact is not available remotely: ${entry.logicalUrls.join(', ')}`);
    planned.push({
      logicalUrl: entry.logicalUrls[0],
      logicalUrls: entry.logicalUrls,
      localPath: entry.localPath,
      key: entry.key,
      bytes: entry.bytes,
      sha256: entry.sha256,
      decodedBytes: entry.decodedBytes,
      decodedSha256: entry.decodedSha256,
      contentType: entry.contentType,
      contentEncoding: entry.contentEncoding,
      cacheControl: 'public, max-age=31536000, immutable',
      artifactUrl: entry.url,
      remoteState: probe?.state ?? 'not-probed',
      uploadAction: probe?.state === 'existing' ? 'conditional-create-or-verify' : 'conditional-create',
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
      if (item.remoteState === 'unchecked') throw new Error(`Cannot safely publish ${item.logicalUrls.join(', ')}: ${item.remoteProbe?.reason ?? 'remote artifact existence was not checked'}`);
      if (!item.localPath || !existsSync(item.localPath)) {
        const remoteState = await verifyRemoteImmutableObject(args, target, item);
        if (remoteState === 'identical') {
          item.remoteState = 'identical-r2';
          item.uploadAction = 'skip-identical-r2';
          continue;
        }
        throw new Error(`Cannot upload missing R2 artifact ${item.logicalUrls.join(', ')} without a local file: ${item.localPath ?? 'no localPath'}`);
      }
      const uploadState = await createImmutableObjectAtomically(args, item);
      item.remoteState = uploadState === 'identical' ? 'identical-r2' : 'created-r2';
      item.uploadAction = uploadState === 'identical' ? 'skip-identical-r2' : 'uploaded';
    }
    for (const item of manifests) {
      const target = `${args.bucket}/${item.key}`;
      if (item.type === 'release-manifest') {
        const uploadState = await createImmutableObjectAtomically(args, item);
        item.remoteState = uploadState === 'identical' ? 'identical' : 'created';
        item.uploadAction = uploadState === 'identical' ? 'skip-identical' : 'uploaded';
        continue;
      }
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
    schema: 'lc0_browser.r2_hashed_artifact_publish_plan.v2',
    releaseId: release.releaseId,
    releaseSchema: release.schema,
    catalogObjectCount: catalog.size,
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
