#!/usr/bin/env node
import { mkdir, mkdtemp, readdir, stat } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { InferenceSession } from 'onnxruntime-web';
import { generateResumableModelShards } from './generate_resumable_model_shards.mjs';
import { reconstructResumableModelShards } from './reconstruct_resumable_model_shards.mjs';

function optionalPositiveSafeInteger(value, option) {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${option} must be a positive safe integer`);
  }
  return value;
}

export function parseArgs(argv) {
  const args = {
    model: undefined,
    workDir: undefined,
    chunkMib: 16,
    concurrency: 3,
    maxManifestBytes: undefined,
    maxDecodedBytes: undefined,
    maxShardReferences: undefined,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === '--model' && next) { args.model = next; index += 1; continue; }
    if (arg === '--work-dir' && next) { args.workDir = next; index += 1; continue; }
    if (arg === '--chunk-mib' && next) { args.chunkMib = Number(next); index += 1; continue; }
    if (arg === '--concurrency' && next) { args.concurrency = Number(next); index += 1; continue; }
    if (arg === '--max-manifest-bytes' && next) { args.maxManifestBytes = Number(next); index += 1; continue; }
    if (arg === '--max-decoded-bytes' && next) { args.maxDecodedBytes = Number(next); index += 1; continue; }
    if (arg === '--max-shard-references' && next) { args.maxShardReferences = Number(next); index += 1; continue; }
    if (arg === '-h' || arg === '--help') {
      console.log('Usage: node --experimental-strip-types scripts/bench_resumable_model_shards.mjs --model model.onnx --work-dir output [--chunk-mib 16] [--concurrency 3] [--max-manifest-bytes bytes] [--max-decoded-bytes bytes] [--max-shard-references count]');
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.model || !args.workDir) throw new Error('--model and --work-dir are required');
  if (!Number.isInteger(args.chunkMib) || args.chunkMib < 16 || args.chunkMib > 32) {
    throw new Error('--chunk-mib must be an integer from 16 through 32');
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1 || args.concurrency > 8) {
    throw new Error('--concurrency must be an integer from 1 through 8');
  }
  return {
    modelPath: resolve(args.model),
    workDir: resolve(args.workDir),
    chunkBytes: args.chunkMib * 1024 * 1024,
    concurrency: args.concurrency,
    maxManifestBytes: optionalPositiveSafeInteger(args.maxManifestBytes, '--max-manifest-bytes'),
    maxDecodedBytes: optionalPositiveSafeInteger(args.maxDecodedBytes, '--max-decoded-bytes'),
    maxShardReferences: optionalPositiveSafeInteger(args.maxShardReferences, '--max-shard-references'),
  };
}

function memorySnapshot() {
  const memory = process.memoryUsage();
  return { rss: memory.rss, heapUsed: memory.heapUsed, arrayBuffers: memory.arrayBuffers };
}

async function measured(run) {
  const baseline = memorySnapshot();
  let peak = { ...baseline };
  const sample = () => {
    const current = memorySnapshot();
    peak = {
      rss: Math.max(peak.rss, current.rss),
      heapUsed: Math.max(peak.heapUsed, current.heapUsed),
      arrayBuffers: Math.max(peak.arrayBuffers, current.arrayBuffers),
    };
  };
  const timer = setInterval(sample, 10);
  const started = performance.now();
  try {
    const value = await run();
    sample();
    return {
      value,
      elapsedMs: performance.now() - started,
      processPeakDeltaBytes: {
        rss: peak.rss - baseline.rss,
        heapUsed: peak.heapUsed - baseline.heapUsed,
        arrayBuffers: peak.arrayBuffers - baseline.arrayBuffers,
      },
    };
  } finally {
    clearInterval(timer);
  }
}

async function createOrtSession(model) {
  const started = performance.now();
  const session = await InferenceSession.create(new Uint8Array(model), { executionProviders: ['wasm'] });
  const elapsedMs = performance.now() - started;
  const inputs = session.inputNames;
  const outputs = session.outputNames;
  await session.release();
  return { elapsedMs, inputs, outputs };
}

async function loadAndCreateSession(manifestPath, cacheDir, options) {
  const load = await reconstructResumableModelShards({ manifestPath, cacheDir, ...options });
  const ort = await createOrtSession(load.model);
  delete load.model;
  return { load, ort };
}

export async function createBenchmarkRunDirectory(workDir) {
  await mkdir(workDir, { recursive: true });
  return mkdtemp(join(workDir, 'run-'));
}

async function main() {
  const args = parseArgs(process.argv);
  const runDir = await createBenchmarkRunDirectory(args.workDir);
  const generation = await measured(() => generateResumableModelShards({
    inputPath: args.modelPath,
    outputDir: join(runDir, 'published'),
    chunkBytes: args.chunkBytes,
  }));
  const reconstructionOptions = {
    concurrency: args.concurrency,
    maxManifestBytes: args.maxManifestBytes,
    maxDecodedBytes: args.maxDecodedBytes,
    maxShardReferences: args.maxShardReferences,
  };

  const cold = await measured(() => loadAndCreateSession(
    generation.value.manifestPath,
    join(runDir, 'cold-cache'),
    reconstructionOptions,
  ));

  const resumeCache = join(runDir, 'resume-cache');
  const controller = new AbortController();
  let abortProgress;
  const interrupted = await measured(async () => {
    try {
      await reconstructResumableModelShards({
        manifestPath: generation.value.manifestPath,
        cacheDir: resumeCache,
        concurrency: 1,
        maxManifestBytes: reconstructionOptions.maxManifestBytes,
        maxDecodedBytes: reconstructionOptions.maxDecodedBytes,
        maxShardReferences: reconstructionOptions.maxShardReferences,
        signal: controller.signal,
        onProgress(progress) {
          if (progress.phase === 'download' && progress.completedBytes >= progress.totalBytes * 0.4) {
            abortProgress = progress;
            controller.abort();
          }
        },
      });
      throw new Error('Expected the 40% resumable benchmark load to abort');
    } catch (error) {
      if (error?.name !== 'AbortError') throw error;
    }
    return { persistedShardFiles: (await readdir(resumeCache)).length };
  });

  const resumed = await measured(() => loadAndCreateSession(
    generation.value.manifestPath,
    resumeCache,
    reconstructionOptions,
  ));
  const modelStat = await stat(args.modelPath);
  const summarize = (measurement) => ({
    startupMs: Number(measurement.elapsedMs.toFixed(3)),
    reconstructionMs: Number(measurement.value.load.elapsedMs.toFixed(3)),
    ortSessionMs: Number(measurement.value.ort.elapsedMs.toFixed(3)),
    downloadedBytes: measurement.value.load.downloadedBytes,
    reusedBytes: measurement.value.load.reusedBytes,
    downloadedShards: measurement.value.load.downloadedShards,
    reusedShards: measurement.value.load.reusedShards,
    loaderPeakTemporaryBytes: measurement.value.load.peakTemporaryBytes,
    processPeakDeltaBytes: measurement.processPeakDeltaBytes,
    decodedSha256: measurement.value.load.sha256,
    ortInputs: measurement.value.ort.inputs,
    ortOutputs: measurement.value.ort.outputs,
  });
  console.log(JSON.stringify({
    schema: 'lc0_browser.resumable_model_shard_benchmark.v1',
    researchOnly: true,
    model: { path: args.modelPath, file: basename(args.modelPath), bytes: modelStat.size },
    chunkBytes: args.chunkBytes,
    concurrency: args.concurrency,
    generation: {
      elapsedMs: Number(generation.elapsedMs.toFixed(3)),
      manifestPath: generation.value.manifestPath,
      decodedSha256: generation.value.manifest.decoded.sha256,
      shardReferences: generation.value.shardReferences,
      uniqueShardCount: generation.value.uniqueShardCount,
    },
    interrupted: {
      elapsedMs: Number(interrupted.elapsedMs.toFixed(3)),
      completedBytes: abortProgress?.completedBytes,
      completedShards: abortProgress?.completedShards,
      totalBytes: abortProgress?.totalBytes,
      totalShards: abortProgress?.totalShards,
      persistedShardFiles: interrupted.value.persistedShardFiles,
    },
    cold: summarize(cold),
    resumed: summarize(resumed),
    byteIdentical: cold.value.load.sha256 === generation.value.manifest.decoded.sha256
      && resumed.value.load.sha256 === generation.value.manifest.decoded.sha256,
    productionRecommendation: {
      recommended: false,
      blocker: 'Successful live Artifact v2 rollout and representative startup evidence are required before production consideration.',
    },
  }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
