#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const MIN_CHUNK_BYTES = 16 * 1024 * 1024;
const MAX_CHUNK_BYTES = 32 * 1024 * 1024;

function parseArgs(argv) {
  const args = { input: undefined, output: undefined, chunkMib: 16 };
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === '--input' && next) { args.input = next; index += 1; continue; }
    if (arg === '--output' && next) { args.output = next; index += 1; continue; }
    if (arg === '--chunk-mib' && next) { args.chunkMib = Number(next); index += 1; continue; }
    if (arg === '-h' || arg === '--help') {
      console.log('Usage: node scripts/generate_resumable_model_shards.mjs --input model.onnx --output output-dir [--chunk-mib 16]');
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.input || !args.output) throw new Error('--input and --output are required');
  if (!Number.isInteger(args.chunkMib) || args.chunkMib < 16 || args.chunkMib > 32) {
    throw new Error('--chunk-mib must be an integer from 16 through 32');
  }
  return {
    inputPath: resolve(args.input),
    outputDir: resolve(args.output),
    chunkBytes: args.chunkMib * 1024 * 1024,
  };
}

async function existingShardIsValid(path, expectedBytes, expectedSha256) {
  try {
    if ((await stat(path)).size !== expectedBytes) return false;
    const hash = createHash('sha256');
    for await (const chunk of createReadStream(path)) hash.update(chunk);
    return hash.digest('hex') === expectedSha256;
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    return false;
  }
}

async function writeShardVerified(path, bytes, sha256) {
  if (await existingShardIsValid(path, bytes.byteLength, sha256)) return false;
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp-${process.pid}-${Date.now()}`;
  try {
    await writeFile(temporaryPath, bytes, { flag: 'wx' });
    await rename(temporaryPath, path);
  } catch (error) {
    try { await unlink(temporaryPath); } catch { /* best-effort cleanup of this script's exact temporary file */ }
    throw error;
  }
  return true;
}

export async function generateResumableModelShards({ inputPath, outputDir, chunkBytes = MIN_CHUNK_BYTES }) {
  if (!Number.isInteger(chunkBytes) || chunkBytes < MIN_CHUNK_BYTES || chunkBytes > MAX_CHUNK_BYTES) {
    throw new Error('chunkBytes must be an integer from 16 MiB through 32 MiB');
  }
  const input = await open(inputPath, 'r');
  const fullHash = createHash('sha256');
  const shards = [];
  let totalBytes = 0;
  let uniqueBytesWritten = 0;
  let deduplicatedShards = 0;
  try {
    for (;;) {
      const chunk = Buffer.allocUnsafe(chunkBytes);
      let filled = 0;
      while (filled < chunk.byteLength) {
        const { bytesRead } = await input.read(chunk, filled, chunk.byteLength - filled);
        if (bytesRead === 0) break;
        filled += bytesRead;
      }
      if (filled === 0) break;
      const bytes = chunk.subarray(0, filled);
      fullHash.update(bytes);
      const sha256 = createHash('sha256').update(bytes).digest('hex');
      const shardPath = join(outputDir, 'shards', 'sha256', `${sha256}.bin`);
      if (await writeShardVerified(shardPath, bytes, sha256)) uniqueBytesWritten += bytes.byteLength;
      else deduplicatedShards += 1;
      shards.push({
        bytes: bytes.byteLength,
        sha256,
        url: relative(outputDir, shardPath).split(sep).join('/'),
      });
      totalBytes += bytes.byteLength;
      if (filled < chunkBytes) break;
    }
  } finally {
    await input.close();
  }
  if (totalBytes === 0) throw new Error('Cannot shard an empty model');

  const manifest = {
    schema: 'lc0_browser.resumable_model_shards.v1',
    researchOnly: true,
    sourceFile: basename(inputPath),
    chunkBytes,
    decoded: {
      bytes: totalBytes,
      sha256: fullHash.digest('hex'),
    },
    shards,
  };
  await mkdir(outputDir, { recursive: true });
  const manifestPath = join(outputDir, `${basename(inputPath)}.resumable.json`);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return {
    manifest,
    manifestPath,
    shardReferences: shards.length,
    uniqueShardCount: new Set(shards.map((shard) => shard.sha256)).size,
    uniqueBytesWritten,
    deduplicatedShards,
  };
}

async function main() {
  const result = await generateResumableModelShards(parseArgs(process.argv));
  console.log(JSON.stringify({
    schema: 'lc0_browser.resumable_model_shard_generation.v1',
    researchOnly: true,
    manifestPath: result.manifestPath,
    decodedBytes: result.manifest.decoded.bytes,
    decodedSha256: result.manifest.decoded.sha256,
    chunkBytes: result.manifest.chunkBytes,
    shardReferences: result.shardReferences,
    uniqueShardCount: result.uniqueShardCount,
    uniqueBytesWritten: result.uniqueBytesWritten,
    deduplicatedShards: result.deduplicatedShards,
    productionRecommendation: 'blocked pending successful live Artifact v2 rollout and startup evidence',
  }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
