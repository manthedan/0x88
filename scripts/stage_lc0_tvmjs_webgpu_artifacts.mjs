#!/usr/bin/env node
import { existsSync, mkdirSync, copyFileSync, statSync, readFileSync, writeFileSync, readdirSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { basename, join, relative, resolve } from 'node:path';

function arg(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((item) => item.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}
function flag(name) { return process.argv.includes(`--${name}`); }
function sha256(path) { return createHash('sha256').update(readFileSync(path)).digest('hex'); }
function requireFile(path) { if (!existsSync(path) || !statSync(path).isFile()) throw new Error(`Required file not found: ${path}`); }
function copyTracked(src, dst, root) {
  requireFile(src);
  mkdirSync(resolve(dst, '..'), { recursive: true });
  copyFileSync(src, dst);
  return { path: relative(root, dst).replaceAll('\\', '/'), bytes: statSync(dst).size, sha256: sha256(dst) };
}
function optionalArgPath(name, envName) {
  const value = arg(name, process.env[envName] ?? '');
  return value ? resolve(value) : undefined;
}
function tensorCacheFileNames(cacheDir) {
  requireFile(join(cacheDir, 'tensor-cache.json'));
  return readdirSync(cacheDir)
    .filter((name) => name === 'tensor-cache.json' || name === 'tensor-cache-b16.json' || /^params_shard/.test(name))
    .sort((a, b) => a.localeCompare(b));
}
function copyTensorCache(cacheDir, outDir, root) {
  const fileNames = tensorCacheFileNames(cacheDir);
  const copied = fileNames.map((name) => copyTracked(join(cacheDir, name), join(outDir, 'tensor-cache', name), root));
  const manifest = copied.find((file) => file.path.endsWith('/tensor-cache.json'));
  const shards = copied.filter((file) => /\/params_shard/.test(file.path));
  return {
    manifest: manifest?.path,
    directory: 'tensor-cache',
    shardCount: shards.length,
    totalBytes: copied.reduce((sum, file) => sum + file.bytes, 0),
    files: copied,
  };
}

function parseBatches(raw) {
  const tokens = String(raw).split(',').map((item) => item.trim());
  if (!tokens.length) throw new Error(`Invalid --batches: ${raw}`);
  const batches = [];
  for (const token of tokens) {
    if (!/^[-+]?\d+$/.test(token)) throw new Error(`Invalid positive integer batch token '${token}' in --batches=${raw}`);
    const value = Number(token);
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Invalid positive integer batch token '${token}' in --batches=${raw}`);
    batches.push(value);
  }
  if (!batches.length) throw new Error(`Invalid --batches: ${raw}`);
  return batches;
}
function renderTemplate(template, values) {
  return String(template).replaceAll('{modelFamily}', values.modelFamily).replaceAll('{dtype}', values.dtype).replaceAll('{batch}', String(values.batch));
}
function commandOutput(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', ...options });
  if (result.status !== 0) return undefined;
  return String(result.stdout ?? '').trim() || undefined;
}
function contains(path, needle) {
  try { return readFileSync(path, 'utf8').includes(needle); } catch { return false; }
}
function publicPathLabel(value, fallback) {
  return basename(String(value || fallback));
}
function tvmProvenance(tvmSrc) {
  const webRuntime = join(tvmSrc, 'web/src/runtime.ts');
  const tvmjsPy = join(tvmSrc, 'python/tvm/contrib/tvmjs.py');
  const emccCommand = process.env.EMCC ?? 'emcc';
  return {
    gitCommit: commandOutput('git', ['rev-parse', 'HEAD'], { cwd: tvmSrc }),
    gitDescribe: commandOutput('git', ['describe', '--always', '--dirty', '--tags'], { cwd: tvmSrc }),
    gitDirty: commandOutput('git', ['status', '--short'], { cwd: tvmSrc }) ? true : false,
    tvmBuildDir: publicPathLabel(process.env.TVM_BUILD_DIR, 'build-tvmjs'),
    note: 'Local filesystem paths are intentionally omitted because this manifest can be browser-fetchable when artifacts are published.',
    emscripten: {
      emcc: publicPathLabel(emccCommand, 'emcc'),
      version: commandOutput(emccCommand, ['--version'])?.split('\n')[0],
    },
    webgpu: {
      requiredFeatures: ['webgpu', 'shader-f16'],
      fetchTensorCacheApiPresent: contains(webRuntime, 'fetchTensorCache('),
      dumpTensorCacheApiPresent: contains(tvmjsPy, 'def dump_tensor_cache('),
    },
  };
}

const repo = process.cwd();
const modelFamily = arg('model-family', process.env.LC0_TVMJS_MODEL_FAMILY ?? 't1-256x10-distilled-swa-2432500');
const dtype = arg('dtype', process.env.LC0_TVMJS_DTYPE ?? 'f16');
const version = arg('version', process.env.LC0_TVMJS_VERSION ?? 'v1');
const batches = parseBatches(arg('batches', process.env.LC0_TVMJS_BATCHES ?? '1,4,8'));
const stemTemplate = arg('stem-template', process.env.LC0_TVMJS_STEM_TEMPLATE ?? '{modelFamily}.batch{batch}.{dtype}.webgpu.tvmjs-wasm.probe');
const out = resolve(arg('out', process.env.LC0_TVMJS_OUT ?? `public/runtimes/lc0-tvmjs-webgpu/${modelFamily}/${dtype}/${version}`));
if (flag('clean')) {
  rmSync(out, { recursive: true, force: true });
  console.log(JSON.stringify({ ok: true, action: 'clean', out }, null, 2));
  process.exit(0);
}
const artifacts = resolve(arg('artifacts', 'artifacts/tvm'));
const tvmSrc = resolve(arg('tvm-src', '../.deps/tvm-webgpu-src'));
const tvmjsBundle = resolve(arg('tvmjs-bundle', join(tvmSrc, 'web/dist/tvmjs.bundle.js')));
const tvmjsRuntimeWasm = resolve(arg('tvmjs-runtime-wasm', join(tvmSrc, 'web/dist/wasm/tvmjs_runtime.wasm')));
const manifestName = arg('manifest-name', 'manifest.json');
const tensorCacheDir = optionalArgPath('tensor-cache-dir', 'LC0_TVMJS_TENSOR_CACHE_DIR');
const paramsMode = arg('params', null);
if (paramsMode === 'detached' && !tensorCacheDir) {
  throw new Error('--params=detached requires --tensor-cache-dir so the staged manifest includes the tensor-cache sidecar required at runtime');
}

mkdirSync(out, { recursive: true });
const files = [];
files.push(copyTracked(tvmjsBundle, join(out, 'tvmjs.bundle.js'), out));
files.push(copyTracked(tvmjsRuntimeWasm, join(out, 'tvmjs_runtime.wasm'), out));
const models = [];
for (const batch of batches) {
  const stem = renderTemplate(stemTemplate, { modelFamily, dtype, batch });
  const wasm = join(artifacts, `${stem}.tvmjs.wasm`);
  const probe = join(artifacts, `${stem}.json`);
  files.push(copyTracked(wasm, join(out, basename(wasm)), out));
  files.push(copyTracked(probe, join(out, basename(probe)), out));
  models.push({
    batch,
    wasm: basename(wasm),
    probe: basename(probe),
    bytes: statSync(wasm).size,
    sha256: sha256(wasm),
  });
}
const tensorCache = tensorCacheDir ? copyTensorCache(tensorCacheDir, out, out) : undefined;
if (tensorCache) files.push(...tensorCache.files);
const manifest = {
  schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
  modelFamily,
  dtype,
  version,
  target: 'webgpu',
  hostTarget: { kind: 'llvm', mtriple: 'wasm32-unknown-unknown-wasm' },
  generatedAt: new Date().toISOString(),
  requiredFeatures: ['webgpu', 'shader-f16'],
  runtime: {
    tvmjsBundle: 'tvmjs.bundle.js',
    tvmjsRuntimeWasm: 'tvmjs_runtime.wasm',
    note: 'TVMJS/WebGPU whole-model export bundle. Runtime/parity/perf still require browser validation.',
  },
  parameterStrategy: {
    current: paramsMode === 'detached'
      ? 'detached-tensor-cache'
      : tensorCache ? 'embedded-wasm-plus-staged-tensor-cache' : 'embedded-in-per-batch-wasm',
    note: paramsMode === 'detached'
      ? 'Model wasm artifacts were built with --detach-params; weights live only in the tensor-cache sidecar and the browser must fetchTensorCache before invoking.'
      : tensorCache
      ? 'A TVM tensor-cache sidecar is staged for research comparison. Model wasm artifacts may still contain embedded params until the export flow detaches params before build.'
      : 'Current research staging embeds model weights in each batch-specific TVMJS wasm. Future release candidates should evaluate TVM tensor-cache weight separation before publication.',
    tensorCacheApis: {
      pythonDumpTensorCache: 'tvm.contrib.tvmjs.dump_tensor_cache',
      browserFetchTensorCache: 'tvm.fetchTensorCache',
    },
  },
  ...(tensorCache ? { tensorCache: {
    manifest: tensorCache.manifest,
    directory: tensorCache.directory,
    shardCount: tensorCache.shardCount,
    totalBytes: tensorCache.totalBytes,
    files: tensorCache.files.map((file) => file.path),
  } } : {}),
  compilerProvenance: tvmProvenance(tvmSrc),
  models,
  files,
};
writeFileSync(join(out, manifestName), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ ok: true, action: 'stage', out, manifest: join(out, manifestName), files: files.length }, null, 2));
