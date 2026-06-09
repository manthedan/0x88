#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile, stat, mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

const DEFAULT_MANIFEST = 'public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json';
const DEFAULT_TVM_SRC = '../.deps/tvm-webgpu-src';
const DEFAULT_OUT = 'artifacts/tvm/lc0_tvmjs_weight_cache_plan.json';

function usage() {
  console.log(`Usage: node scripts/analyze_lc0_tvmjs_weight_cache_plan.mjs [options]\n\nResearch-only analyzer for TVMJS tensor-cache weight separation readiness.\nIt does not generate or publish separated params; it validates local TVM API support and quantifies the current embedded-per-batch wasm footprint.\n\nOptions:\n  --manifest PATH  Staged TVMJS manifest (default ${DEFAULT_MANIFEST})\n  --tvm-src PATH   Durable TVM source checkout (default ${DEFAULT_TVM_SRC})\n  --out PATH       Output JSON artifact (default ${DEFAULT_OUT})\n  --no-write       Print only\n  -h, --help       Show help\n`);
}

function parseArgs(argv) {
  const args = { manifest: DEFAULT_MANIFEST, tvmSrc: DEFAULT_TVM_SRC, out: DEFAULT_OUT, write: true };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--manifest') args.manifest = next();
    else if (arg === '--tvm-src') args.tvmSrc = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--no-write') args.write = false;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

async function loadJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function fileInfo(path) {
  const bytes = (await stat(path)).size;
  const sha256 = createHash('sha256').update(await readFile(path)).digest('hex');
  return { bytes, sha256 };
}

async function sourceContains(path, needle) {
  try { return (await readFile(path, 'utf8')).includes(needle); } catch { return false; }
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function percent(value, base) {
  if (!Number.isFinite(value) || !Number.isFinite(base) || base <= 0) return undefined;
  return Number((100 * value / base).toFixed(2));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const manifestPath = args.manifest;
  const manifestDir = dirname(manifestPath);
  const tvmSrc = resolve(args.tvmSrc);
  const manifest = await loadJson(manifestPath);
  const modelFiles = [];
  for (const model of manifest.models ?? []) {
    const info = await fileInfo(join(manifestDir, model.wasm));
    modelFiles.push({ batch: model.batch, path: model.wasm, bytes: info.bytes, sha256: info.sha256 });
  }
  const runtimeFiles = [];
  for (const rel of [manifest.runtime?.tvmjsBundle, manifest.runtime?.tvmjsRuntimeWasm].filter(Boolean)) {
    const info = await fileInfo(join(manifestDir, rel));
    runtimeFiles.push({ path: rel, ...info });
  }
  const modelWasmBytes = sum(modelFiles.map((file) => file.bytes));
  const runtimeBytes = sum(runtimeFiles.map((file) => file.bytes));
  const largestModelWasmBytes = modelFiles.length ? Math.max(...modelFiles.map((file) => file.bytes)) : 0;
  const duplicateBatchWasmUpperBoundBytes = Math.max(0, modelWasmBytes - largestModelWasmBytes);
  const apiSupport = {
    pythonDumpTensorCache: await sourceContains(join(tvmSrc, 'python/tvm/contrib/tvmjs.py'), 'def dump_tensor_cache('),
    pythonLoadTensorCache: await sourceContains(join(tvmSrc, 'python/tvm/contrib/tvmjs.py'), 'def load_tensor_cache('),
    browserFetchTensorCache: await sourceContains(join(tvmSrc, 'web/src/runtime.ts'), 'fetchTensorCache('),
    browserArtifactCache: await sourceContains(join(tvmSrc, 'web/src/artifact_cache.ts'), 'tensor-cache.json'),
    relaxDetachParams: await sourceContains(join(tvmSrc, 'python/tvm/relax/frontend/common.py'), 'def detach_params('),
  };
  const currentParameterStrategy = manifest.parameterStrategy?.current ?? 'embedded-in-per-batch-wasm';
  const requiredApisPresent = apiSupport.pythonDumpTensorCache && apiSupport.browserFetchTensorCache && apiSupport.browserArtifactCache;
  const out = {
    schema: 'lc0_browser.tvmjs_weight_cache_plan.v1',
    generatedAt: new Date().toISOString(),
    ok: requiredApisPresent && modelFiles.length > 0,
    researchOnly: true,
    noStableRuntimePromotion: true,
    manifest: manifestPath,
    manifestSchema: manifest.schema,
    modelFamily: manifest.modelFamily,
    dtype: manifest.dtype,
    version: manifest.version,
    requiredFeatures: manifest.requiredFeatures,
    currentParameterStrategy,
    apiSupport,
    footprint: {
      runtimeBytes,
      modelWasmBytes,
      totalRuntimeAndModelBytes: runtimeBytes + modelWasmBytes,
      modelWasmSharePct: percent(modelWasmBytes, runtimeBytes + modelWasmBytes),
      modelFiles,
      runtimeFiles,
      largestModelWasmBytes,
      duplicateBatchWasmUpperBoundBytes,
      duplicateBatchWasmUpperBoundPctOfRuntimeAndModel: percent(duplicateBatchWasmUpperBoundBytes, runtimeBytes + modelWasmBytes),
      caveat: 'Upper bound assumes the largest batch wasm would remain and the other batch wasm bytes are duplicated. Exact savings require a real detached-param export because TVM code/metadata also differs by batch.',
    },
    recommendedNextRecipe: [
      'Use Relax/frontend parameter detachment where possible before VM build, or otherwise export params through tvm.contrib.tvmjs.dump_tensor_cache with encode_format="raw" for f16 weights.',
      'Stage tensor-cache.json plus params_shard_* files under an immutable versioned runtime path and add hashes/bytes to the manifest.',
      'Update the browser loader to fetch shared params with tvm.fetchTensorCache before VM invocation, preserving shader-f16 gating.',
      'Compare embedded-vs-tensor-cache cold start, repeat-load cache hit behavior, raw/gzip/Brotli footprint, and search parity before any release decision.',
    ],
    blockers: requiredApisPresent ? [] : ['Local TVM checkout is missing one or more tensor-cache APIs expected by the runbook.'],
  };
  if (args.write) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, `${JSON.stringify(out, null, 2)}\n`);
  }
  console.log(JSON.stringify(out, null, 2));
  if (!out.ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
});
