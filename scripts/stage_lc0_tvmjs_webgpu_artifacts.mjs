#!/usr/bin/env node
import { existsSync, mkdirSync, copyFileSync, statSync, readFileSync, writeFileSync, readdirSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
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

const repo = process.cwd();
const out = resolve(arg('out', 'public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1'));
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

const batches = [1, 4, 8];
mkdirSync(out, { recursive: true });
const files = [];
files.push(copyTracked(tvmjsBundle, join(out, 'tvmjs.bundle.js'), out));
files.push(copyTracked(tvmjsRuntimeWasm, join(out, 'tvmjs_runtime.wasm'), out));
const models = [];
for (const batch of batches) {
  const stem = `t1-256x10-distilled-swa-2432500.batch${batch}.f16.webgpu.tvmjs-wasm.probe`;
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
const manifest = {
  schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
  modelFamily: 't1-256x10-distilled-swa-2432500',
  dtype: 'f16',
  target: 'webgpu',
  hostTarget: { kind: 'llvm', mtriple: 'wasm32-unknown-unknown-wasm' },
  generatedAt: new Date().toISOString(),
  requiredFeatures: ['webgpu', 'shader-f16'],
  runtime: {
    tvmjsBundle: 'tvmjs.bundle.js',
    tvmjsRuntimeWasm: 'tvmjs_runtime.wasm',
    note: 'TVMJS/WebGPU whole-model export bundle. Runtime/parity/perf still require browser validation.',
  },
  models,
  files,
};
writeFileSync(join(out, manifestName), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ ok: true, action: 'stage', out, manifest: join(out, manifestName), files: files.length }, null, 2));
