#!/usr/bin/env node
import { constants } from 'node:fs';
import { access, mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { loadResumableLc0ModelForOrt } from '../src/lc0/modelCache.ts';

function parseArgs(argv) {
  const args = { manifest: undefined, output: undefined, cacheDir: undefined, concurrency: 3 };
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === '--manifest' && next) { args.manifest = next; index += 1; continue; }
    if (arg === '--output' && next) { args.output = next; index += 1; continue; }
    if (arg === '--cache-dir' && next) { args.cacheDir = next; index += 1; continue; }
    if (arg === '--concurrency' && next) { args.concurrency = Number(next); index += 1; continue; }
    if (arg === '-h' || arg === '--help') {
      console.log('Usage: node --experimental-strip-types scripts/reconstruct_resumable_model_shards.mjs --manifest model.resumable.json --output model.onnx --cache-dir shard-cache [--concurrency 3]');
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.manifest || !args.output || !args.cacheDir) {
    throw new Error('--manifest, --output, and --cache-dir are required');
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1 || args.concurrency > 8) {
    throw new Error('--concurrency must be an integer from 1 through 8');
  }
  return {
    manifestPath: resolve(args.manifest),
    outputPath: resolve(args.output),
    cacheDir: resolve(args.cacheDir),
    concurrency: args.concurrency,
  };
}

function exactArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function exists(path) {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export class FileModelShardStore {
  constructor(directory) {
    this.directory = directory;
  }

  path(sha256) {
    return join(this.directory, `${sha256}.bin`);
  }

  async get(sha256) {
    try {
      return exactArrayBuffer(await readFile(this.path(sha256)));
    } catch (error) {
      if (error?.code === 'ENOENT') return undefined;
      throw error;
    }
  }

  async put(sha256, bytes) {
    await mkdir(this.directory, { recursive: true });
    const destination = this.path(sha256);
    if (await exists(destination)) return;
    const temporary = `${destination}.tmp-${process.pid}-${Date.now()}`;
    try {
      await writeFile(temporary, new Uint8Array(bytes), { flag: 'wx' });
      try {
        await rename(temporary, destination);
      } catch (error) {
        if (!await exists(destination)) throw error;
        await unlink(temporary);
      }
    } catch (error) {
      try { await unlink(temporary); } catch { /* best-effort cleanup of this exact temporary file */ }
      throw error;
    }
  }

  async delete(sha256) {
    try {
      await unlink(this.path(sha256));
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
}

export function localFileFetch() {
  return async (input) => {
    const raw = typeof input === 'string' ? input : input.url;
    const url = new URL(raw);
    if (url.protocol !== 'file:') return new Response(null, { status: 404 });
    try {
      const bytes = await readFile(fileURLToPath(url));
      return new Response(bytes);
    } catch (error) {
      if (error?.code === 'ENOENT') return new Response(null, { status: 404 });
      throw error;
    }
  };
}

export async function reconstructResumableModelShards({
  manifestPath,
  cacheDir,
  concurrency = 3,
  signal,
  onProgress,
}) {
  return loadResumableLc0ModelForOrt(pathToFileURL(manifestPath).href, {
    researchOnly: true,
    concurrency,
    signal,
    onProgress,
    fetchFn: localFileFetch(),
    shardStore: new FileModelShardStore(cacheDir),
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const result = await reconstructResumableModelShards(args);
  await mkdir(dirname(args.outputPath), { recursive: true });
  await writeFile(args.outputPath, new Uint8Array(result.model));
  console.log(JSON.stringify({
    schema: 'lc0_browser.resumable_model_reconstruction.v1',
    researchOnly: true,
    outputPath: args.outputPath,
    sha256: result.sha256,
    bytes: result.bytes,
    downloadedBytes: result.downloadedBytes,
    reusedBytes: result.reusedBytes,
    downloadedShards: result.downloadedShards,
    reusedShards: result.reusedShards,
    corruptShardsEvicted: result.corruptShardsEvicted,
    corruptionRetries: result.corruptionRetries,
    peakTemporaryBytes: result.peakTemporaryBytes,
    elapsedMs: Number(result.elapsedMs.toFixed(3)),
    productionRecommendation: 'blocked pending successful live Artifact v2 rollout and startup evidence',
  }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
