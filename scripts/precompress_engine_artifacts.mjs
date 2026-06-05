#!/usr/bin/env node
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, extname, join, relative, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? 'public');
const allowMissing = process.argv.includes('--allow-missing');
const force = process.argv.includes('--force');
const engines = ['berserk', 'plentychess'];
const compressibleExts = new Set(['.js', '.wasm', '.data', '.nn', '.nnue', '.bin']);

function walk(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

function shouldCompress(path) {
  if (path.endsWith('.br') || path.endsWith('.gz')) return false;
  return compressibleExts.has(extname(path));
}

function needsWrite(source, target) {
  if (force || !existsSync(target)) return true;
  return statSync(target).mtimeMs < statSync(source).mtimeMs;
}

async function writeIfNeeded(path, suffix, bytes) {
  const target = `${path}${suffix}`;
  if (!needsWrite(path, target)) return { target, skipped: true, bytes: statSync(target).size };
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes);
  return { target, skipped: false, bytes: bytes.byteLength };
}

if (!existsSync(root)) {
  if (allowMissing) process.exit(0);
  throw new Error(`Missing artifact root: ${root}`);
}

let sources = [];
for (const engine of engines) {
  const dir = join(root, engine);
  if (!existsSync(dir)) {
    if (!allowMissing) throw new Error(`Missing engine artifact dir: ${dir}`);
    continue;
  }
  sources.push(...walk(dir).filter(shouldCompress));
}

sources = sources.sort();
if (!sources.length) {
  console.log(`No engine artifacts to precompress under ${relative(process.cwd(), root) || '.'}`);
  process.exit(0);
}

for (const source of sources) {
  const input = await readFile(source);
  const gzip = gzipSync(input, { level: 9 });
  const brotli = brotliCompressSync(input, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 } });
  const gz = await writeIfNeeded(source, '.gz', gzip);
  const br = await writeIfNeeded(source, '.br', brotli);
  const rel = relative(process.cwd(), source);
  const action = gz.skipped && br.skipped ? 'kept' : 'wrote';
  console.log(`${action} ${rel} -> gzip ${gz.bytes} bytes, brotli ${br.bytes} bytes`);
}
