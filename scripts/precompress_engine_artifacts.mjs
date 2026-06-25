#!/usr/bin/env node
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, extname, join, relative, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? 'public');
const allowMissing = process.argv.includes('--allow-missing');
const force = process.argv.includes('--force');
const excludedEngines = new Set(process.argv.flatMap((arg, index, args) => {
  if (arg === '--exclude') return args[index + 1] ? [args[index + 1]] : [];
  if (arg.startsWith('--exclude=')) return [arg.slice('--exclude='.length)];
  return [];
}));
// All directories that serve large fetchable artifacts. Missing dirs are
// skipped under --allow-missing.
const engines = ['berserk', 'plentychess', 'stockfish', 'viridithas', 'reckless', 'ort', 'models', 'monty'].filter((engine) => !excludedEngines.has(engine));
// .onnx: f16/int8 model weights only compress ~0.90, but at current sizes
// that is still ~2MB (t1 qdq) to ~35MB (BT4) per asset. .mjs covers ORT's
// glue sidecars in /ort/.
const compressibleExts = new Set(['.js', '.mjs', '.wasm', '.data', '.nn', '.nnue', '.bin', '.onnx']);

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

async function writeCompressed(path, suffix, bytes) {
  const target = `${path}${suffix}`;
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes);
  return { target, skipped: false, bytes: bytes.byteLength };
}

function keptCompressed(path, suffix) {
  const target = `${path}${suffix}`;
  return { target, skipped: true, bytes: statSync(target).size };
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
  const needGzip = needsWrite(source, `${source}.gz`);
  const needBrotli = needsWrite(source, `${source}.br`);
  let gz;
  let br;
  if (needGzip || needBrotli) {
    const input = await readFile(source);
    if (needGzip) gz = await writeCompressed(source, '.gz', gzipSync(input, { level: 9 }));
    else gz = keptCompressed(source, '.gz');
    if (needBrotli) {
      // Brotli q11 takes minutes on the multi-hundred-MB nets for ~1% extra over
      // q5; use fast quality past 64MB so deploy builds stay quick.
      const brotliQuality = input.byteLength > 64 * 1024 * 1024 ? 5 : 11;
      br = await writeCompressed(source, '.br', brotliCompressSync(input, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: brotliQuality } }));
    } else {
      br = keptCompressed(source, '.br');
    }
  } else {
    gz = keptCompressed(source, '.gz');
    br = keptCompressed(source, '.br');
  }
  const rel = relative(process.cwd(), source);
  const action = gz.skipped && br.skipped ? 'kept' : 'wrote';
  console.log(`${action} ${rel} -> gzip ${gz.bytes} bytes, brotli ${br.bytes} bytes`);
}
