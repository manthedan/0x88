#!/usr/bin/env node
import { existsSync, readdirSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? 'dist-client');
const canonical = join(root, 'ort', 'ort-wasm-simd-threaded.asyncify.wasm');

function walk(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

if (!existsSync(canonical)) {
  throw new Error(`Missing canonical staged ORT WASM: ${relative(process.cwd(), canonical)}`);
}

const bundledCopies = walk(join(root, '_app')).filter((path) => /ort-wasm.*\.wasm$/.test(path));
if (bundledCopies.length) {
  throw new Error(`Vite emitted duplicate ORT WASM assets: ${bundledCopies.map((path) => relative(process.cwd(), path)).join(', ')}`);
}

console.log(JSON.stringify({
  status: 'ORT_WASM_DEDUP_CHECK_DONE',
  root: relative(process.cwd(), root) || '.',
  canonical: relative(process.cwd(), canonical),
  bundledCopies: [],
}, null, 2));
