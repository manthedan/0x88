#!/usr/bin/env node
import { constants, createReadStream } from 'node:fs';
import { access, mkdir, readFile, readdir, rename, stat, unlink, utimes, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { Readable } from 'node:stream';
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

  async touch(sha256) {
    const now = new Date();
    try {
      await utimes(this.path(sha256), now, now);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }

  async list() {
    let names;
    try {
      names = await readdir(this.directory);
    } catch (error) {
      if (error?.code === 'ENOENT') return [];
      throw error;
    }
    const entries = [];
    for (const name of names) {
      const match = /^([0-9a-f]{64})\.bin$/i.exec(name);
      if (!match) continue;
      const metadata = await stat(join(this.directory, name));
      entries.push({
        sha256: match[1].toLowerCase(),
        bytes: metadata.size,
        lastUsedAt: metadata.mtimeMs,
      });
    }
    return entries;
  }

  async clear() {
    const entries = await this.list();
    await Promise.all(entries.map((entry) => this.delete(entry.sha256)));
    return entries.length;
  }
}

export function localFileFetch() {
  const expectedBytesByUrl = new Map();
  return async (input, init = {}) => {
    const raw = typeof input === 'string' ? input : input.url;
    const url = new URL(raw);
    if (url.protocol !== 'file:') return new Response(null, { status: 404 });
    const signal = init.signal ?? (typeof input === 'string' ? undefined : input.signal);
    if (signal?.aborted) throw new DOMException('The operation was aborted', 'AbortError');
    const path = fileURLToPath(url);
    const expectedBytes = expectedBytesByUrl.get(url.href);
    try {
      if (expectedBytes === undefined && url.pathname.endsWith('.json')) {
        const bytes = await readFile(path, { signal });
        if (url.pathname.endsWith('.resumable.json')) {
          try {
            const manifest = JSON.parse(bytes);
            if (Array.isArray(manifest?.shards)) {
              for (const shard of manifest.shards) {
                if (typeof shard?.url === 'string' && Number.isSafeInteger(shard.bytes) && shard.bytes >= 0) {
                  expectedBytesByUrl.set(new URL(shard.url, url).href, shard.bytes);
                }
              }
            }
          } catch {
            // The loader reports malformed manifest JSON with its normal validation error.
          }
        }
        return new Response(bytes, { headers: { 'content-type': 'application/json' } });
      }
      const metadata = await stat(path);
      if (expectedBytes !== undefined && metadata.size > expectedBytes) {
        throw new Error(`Local shard file exceeded expected length ${expectedBytes}: ${path}`);
      }
      let loaded = 0;
      const source = Readable.toWeb(createReadStream(path, { signal }));
      const bounded = source.pipeThrough(new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (expectedBytes !== undefined && loaded > expectedBytes) {
            controller.error(new Error(`Local shard file exceeded expected length ${expectedBytes}: ${path}`));
            return;
          }
          controller.enqueue(chunk);
        },
      }), { signal });
      return new Response(bounded, {
        headers: {
          'content-length': String(metadata.size),
        },
      });
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
