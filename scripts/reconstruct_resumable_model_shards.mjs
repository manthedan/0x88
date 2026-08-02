#!/usr/bin/env node
import { randomUUID } from 'node:crypto';
import { constants, createReadStream } from 'node:fs';
import { access, lstat, mkdir, open, readdir, readFile, realpath, rename, stat, unlink, utimes, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { loadResumableLc0ModelForOrt } from '../src/lc0/modelCache.ts';

function optionalPositiveSafeInteger(value, option) {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${option} must be a positive safe integer`);
  }
  return value;
}

export function parseArgs(argv) {
  const args = {
    manifest: undefined,
    output: undefined,
    cacheDir: undefined,
    concurrency: 3,
    maxManifestBytes: undefined,
    maxDecodedBytes: undefined,
    maxShardReferences: undefined,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === '--manifest' && next) {
      args.manifest = next;
      index += 1;
      continue;
    }
    if (arg === '--output' && next) {
      args.output = next;
      index += 1;
      continue;
    }
    if (arg === '--cache-dir' && next) {
      args.cacheDir = next;
      index += 1;
      continue;
    }
    if (arg === '--concurrency' && next) {
      args.concurrency = Number(next);
      index += 1;
      continue;
    }
    if (arg === '--max-manifest-bytes' && next) {
      args.maxManifestBytes = Number(next);
      index += 1;
      continue;
    }
    if (arg === '--max-decoded-bytes' && next) {
      args.maxDecodedBytes = Number(next);
      index += 1;
      continue;
    }
    if (arg === '--max-shard-references' && next) {
      args.maxShardReferences = Number(next);
      index += 1;
      continue;
    }
    if (arg === '-h' || arg === '--help') {
      console.log(
        'Usage: node --experimental-strip-types scripts/reconstruct_resumable_model_shards.mjs --manifest model.resumable.json --output model.onnx --cache-dir shard-cache [--concurrency 3] [--max-manifest-bytes bytes] [--max-decoded-bytes bytes] [--max-shard-references count]',
      );
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
    maxManifestBytes: optionalPositiveSafeInteger(args.maxManifestBytes, '--max-manifest-bytes'),
    maxDecodedBytes: optionalPositiveSafeInteger(args.maxDecodedBytes, '--max-decoded-bytes'),
    maxShardReferences: optionalPositiveSafeInteger(args.maxShardReferences, '--max-shard-references'),
  };
}

function exactArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function throwIfAborted(signal) {
  if (!signal?.aborted) return;
  const error = new Error('The operation was aborted');
  error.name = 'AbortError';
  throw error;
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
    this.directory = resolve(directory);
    this.persistenceDomainKey = `file-model-shard-store:${pathToFileURL(this.directory).href}`;
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

  async getBounded(sha256, expectedBytes, signal) {
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 0) {
      throw new Error('Expected shard bytes must be a non-negative safe integer');
    }
    throwIfAborted(signal);
    const path = this.path(sha256);
    let metadata;
    try {
      metadata = await stat(path);
    } catch (error) {
      if (error?.code === 'ENOENT') return undefined;
      throw error;
    }
    throwIfAborted(signal);
    if (metadata.size > expectedBytes) {
      await this.delete(sha256);
      throw new Error(`Resumable model shard response exceeded expected length ${expectedBytes}`);
    }
    const target = new Uint8Array(expectedBytes);
    let loaded = 0;
    try {
      for await (const chunk of createReadStream(path, { signal })) {
        throwIfAborted(signal);
        if (loaded + chunk.byteLength > expectedBytes) {
          throw new Error(`Resumable model shard response exceeded expected length ${expectedBytes}`);
        }
        target.set(chunk, loaded);
        loaded += chunk.byteLength;
      }
      throwIfAborted(signal);
    } catch (error) {
      if (error?.code === 'ENOENT') return undefined;
      if (error instanceof Error && error.message.startsWith('Resumable model shard response exceeded expected length')) {
        await this.delete(sha256);
      }
      throw error;
    }
    return loaded === expectedBytes ? target.buffer : target.buffer.slice(0, loaded);
  }

  async put(sha256, bytes) {
    await mkdir(this.directory, { recursive: true });
    const destination = this.path(sha256);
    if (await exists(destination)) return;
    const temporary = `${destination}.tmp-${process.pid}-${randomUUID()}`;
    try {
      await writeFile(temporary, new Uint8Array(bytes), { flag: 'wx' });
      try {
        await rename(temporary, destination);
      } catch (error) {
        if (!(await exists(destination))) throw error;
        await unlink(temporary);
      }
    } catch (error) {
      try {
        await unlink(temporary);
      } catch {
        /* best-effort cleanup of this exact temporary file */
      }
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

function pathIsWithin(root, candidate) {
  const relativePath = relative(root, candidate);
  return (
    relativePath === '' || (!isAbsolute(relativePath) && relativePath !== '..' && !relativePath.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
  );
}

function resolveLocalShardUrl(shardUrl, manifestBaseUrl, publicationRoot) {
  try {
    new URL(shardUrl);
    throw new Error(`Local resumable model shard URL must be relative: ${shardUrl}`);
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('Local resumable model shard URL must be relative:')) throw error;
  }
  const rawPath = shardUrl.split(/[?#]/, 1)[0];
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(rawPath).replaceAll('\\', '/');
  } catch {
    throw new Error(`Invalid local resumable model shard URL: ${shardUrl}`);
  }
  if (decodedPath.startsWith('/') || decodedPath.split('/').includes('..')) {
    throw new Error(`Local resumable model shard URL escapes the publication root: ${shardUrl}`);
  }
  const resolvedUrl = new URL(shardUrl, manifestBaseUrl);
  if (resolvedUrl.protocol !== 'file:') {
    throw new Error(`Local resumable model shard URL must use the manifest publication root: ${shardUrl}`);
  }
  const resolvedPath = fileURLToPath(resolvedUrl);
  if (!pathIsWithin(publicationRoot, resolvedPath)) {
    throw new Error(`Local resumable model shard URL escapes the publication root: ${shardUrl}`);
  }
  return resolvedUrl.href;
}

export function localFileFetch(publicationRoot) {
  const resolvedPublicationRoot = resolve(publicationRoot);
  const publicationRootRealPath = realpath(resolvedPublicationRoot);
  return async (input, init = {}) => {
    const raw = typeof input === 'string' ? input : input.url;
    const url = new URL(raw);
    if (url.protocol !== 'file:') return new Response(null, { status: 404 });
    const signal = init.signal ?? (typeof input === 'string' ? undefined : input.signal);
    if (signal?.aborted) throw new DOMException('The operation was aborted', 'AbortError');
    const path = fileURLToPath(url);
    if (!pathIsWithin(resolvedPublicationRoot, path)) {
      throw new Error(`Local resumable model file escapes the publication root: ${path}`);
    }
    try {
      const [metadata, canonicalRoot, canonicalPath] = await Promise.all([lstat(path), publicationRootRealPath, realpath(path)]);
      throwIfAborted(signal);
      if (!metadata.isFile() || !pathIsWithin(canonicalRoot, canonicalPath)) {
        throw new Error(`Local resumable model file must be a regular file within the publication root: ${path}`);
      }
      const source = Readable.toWeb(createReadStream(path, { signal }));
      const bounded = source.pipeThrough(
        new TransformStream({
          transform(chunk, controller) {
            controller.enqueue(chunk);
          },
        }),
        { signal },
      );
      return new Response(bounded, {
        headers: {
          'content-length': String(metadata.size),
          ...(url.pathname.endsWith('.json') ? { 'content-type': 'application/json' } : {}),
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
  maxManifestBytes,
  maxDecodedBytes,
  maxShardReferences,
  signal,
  onProgress,
}) {
  optionalPositiveSafeInteger(maxManifestBytes, 'maxManifestBytes');
  optionalPositiveSafeInteger(maxDecodedBytes, 'maxDecodedBytes');
  optionalPositiveSafeInteger(maxShardReferences, 'maxShardReferences');
  const resolvedManifestPath = resolve(manifestPath);
  const publicationRoot = dirname(resolvedManifestPath);
  return loadResumableLc0ModelForOrt(pathToFileURL(resolvedManifestPath).href, {
    researchOnly: true,
    concurrency,
    maxManifestBytes,
    maxDecodedBytes,
    maxShardReferences,
    signal,
    onProgress,
    fetchFn: localFileFetch(publicationRoot),
    shardStore: new FileModelShardStore(cacheDir),
    resolveShardUrl: (shardUrl, manifestBaseUrl) => resolveLocalShardUrl(shardUrl, manifestBaseUrl, publicationRoot),
  });
}

export async function writeReconstructedModelAtomically(outputPath, model, renameFile = rename) {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.tmp-${process.pid}-${randomUUID()}`;
  let file;
  let temporaryCreated = false;
  try {
    file = await open(temporaryPath, 'wx');
    temporaryCreated = true;
    await file.writeFile(new Uint8Array(model));
    await file.sync();
    await file.close();
    file = undefined;
    await renameFile(temporaryPath, outputPath);
    temporaryCreated = false;
  } catch (error) {
    if (file) {
      try {
        await file.close();
      } catch {
        /* best-effort close before cleanup */
      }
    }
    if (temporaryCreated) {
      try {
        await unlink(temporaryPath);
      } catch {
        /* best-effort cleanup of this exact temporary file */
      }
    }
    throw error;
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const result = await reconstructResumableModelShards(args);
  await writeReconstructedModelAtomically(args.outputPath, result.model);
  console.log(
    JSON.stringify(
      {
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
      },
      null,
      2,
    ),
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
