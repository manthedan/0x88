#!/usr/bin/env node
import { existsSync, readdirSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { ORT_RUNTIME_WASM_FILES } from './ort_runtime_assets.mjs';

const root = resolve(process.argv[2] ?? 'dist-client');
// The staged WebGPU (asyncify) and CPU-only wasm binaries.
const canonical = ORT_RUNTIME_WASM_FILES.map((name) => join(root, 'ort', name));

function walk(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

const missing = canonical.filter((path) => !existsSync(path));
if (missing.length) {
  throw new Error(`Missing canonical staged ORT WASM: ${missing.map((path) => relative(process.cwd(), path)).join(', ')}`);
}

const bundledCopies = walk(join(root, '_app')).filter((path) => /ort-wasm.*\.wasm$/.test(path));
if (bundledCopies.length) {
  throw new Error(`Vite emitted duplicate ORT WASM assets: ${bundledCopies.map((path) => relative(process.cwd(), path)).join(', ')}`);
}

console.log(JSON.stringify({
  status: 'ORT_WASM_DEDUP_CHECK_DONE',
  root: relative(process.cwd(), root) || '.',
  canonical: canonical.map((path) => relative(process.cwd(), path)),
  bundledCopies: [],
}, null, 2));
