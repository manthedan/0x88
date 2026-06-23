#!/usr/bin/env node
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const bucket = process.env.R2_BUCKET ?? 'browser-chess-models';
const publicBase = process.env.R2_PUBLIC_BASE_URL ?? 'https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev';
const verifyOrigin = process.env.R2_VERIFY_ORIGIN ?? 'https://chess-engine-browser.netlify.app';
const outRoot = resolve(process.env.R2_BROTLI_OUT_DIR ?? '/tmp/r2-brotli-artifacts');
const remote = process.env.R2_REMOTE ?? '1';
const args = new Set(process.argv.slice(2));
const dryRun = args.has('--dry-run');
const verifyOnly = args.has('--verify-only');
const skipUpload = args.has('--skip-upload') || dryRun;

const immutable = 'public, max-age=31536000, immutable';
const shortCache = 'public, max-age=300';

const targets = [
  ['models/maia3/manifest.json', 'public/models/maia3/manifest.json', 'application/json', shortCache],
  ['models/maia3/maia3_simplified.qdq8.onnx', 'public/models/maia3/maia3_simplified.qdq8.onnx', 'application/octet-stream', immutable],
  ['models/maia3/maia3_simplified.onnx', 'public/models/maia3/maia3_simplified.onnx', 'application/octet-stream', immutable],
  ['models/lc0/manifest.json', 'public/models/lc0/manifest.json', 'application/json', shortCache],
  ['models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx', 'public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx', 'application/octet-stream', immutable],
  ['models/lc0/lqo_v2.f16.qdq8.onnx', 'public/models/lc0/lqo_v2.f16.qdq8.onnx', 'application/octet-stream', immutable],
  ['models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json', 'public/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json', 'application/json', immutable],
  ['models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.000.bin', 'public/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.000.bin', 'application/octet-stream', immutable],
  ['models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.001.bin', 'public/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.001.bin', 'application/octet-stream', immutable],
  ['models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.002.bin', 'public/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.002.bin', 'application/octet-stream', immutable],
  ['stockfish/stockfish-18.0.7.manifest.json', 'public/stockfish/stockfish-18.0.7.manifest.json', 'application/json', immutable],
  ['stockfish/stockfish-18.0.7-corresponding-source.tar.gz', 'public/stockfish/stockfish-18.0.7-corresponding-source.tar.gz', 'application/gzip', immutable],
  ['berserk/berserk-emscripten-single-thread.manifest.json', 'public/berserk/berserk-emscripten-single-thread.manifest.json', 'application/json', immutable],
  ['berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz', 'public/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz', 'application/gzip', immutable],
  ...['berserk-emscripten', 'berserk-emscripten-simd128', 'berserk-emscripten-relaxed-simd128'].flatMap((base) => [
    [`berserk/${base}.js`, `public/berserk/${base}.js`, 'text/javascript; charset=utf-8', immutable],
    [`berserk/${base}.wasm`, `public/berserk/${base}.wasm`, 'application/wasm', immutable],
    [`berserk/${base}.data`, `public/berserk/${base}.data`, 'application/octet-stream', immutable],
  ]),
  ...['plentychess-emscripten', 'plentychess-emscripten-sse41', 'plentychess-emscripten-relaxed-simd128'].flatMap((base) => [
    [`plentychess/${base}.js`, `public/plentychess/${base}.js`, 'text/javascript; charset=utf-8', immutable],
    [`plentychess/${base}.wasm`, `public/plentychess/${base}.wasm`, 'application/wasm', immutable],
    [`plentychess/${base}.data`, `public/plentychess/${base}.data`, 'application/octet-stream', immutable],
  ]),
  ['plentychess/plentychess-emscripten-single-thread.manifest.json', 'public/plentychess/plentychess-emscripten-single-thread.manifest.json', 'application/json', immutable],
  ['plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz', 'public/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz', 'application/gzip', immutable],
  ...['viridithas.wasm', 'viridithas-simd128.wasm', 'viridithas-relaxed-simd128.wasm'].map((name) => [
    `viridithas/${name}`,
    `public/viridithas/${name}`,
    'application/wasm',
    immutable,
  ]),
  ['viridithas/viridithas-wasip1.manifest.json', 'public/viridithas/viridithas-wasip1.manifest.json', 'application/json', immutable],
  ['viridithas/viridithas-wasip1-corresponding-source.tar.gz', 'public/viridithas/viridithas-wasip1-corresponding-source.tar.gz', 'application/gzip', immutable],
  ['reckless/NOTICE.md', 'public/reckless/NOTICE.md', 'text/markdown; charset=utf-8', immutable],
  ...['reckless.wasm', 'reckless-simd128.wasm', 'reckless-relaxed-simd128.wasm'].map((name) => [
    `reckless/${name}`,
    `public/reckless/${name}`,
    'application/wasm',
    immutable,
  ]),
  ...['reckless-scalar-corresponding-source.tar.gz', 'reckless-simd128-corresponding-source.tar.gz', 'reckless-relaxed-simd128-corresponding-source.tar.gz'].map((name) => [
    `reckless/${name}`,
    `public/reckless/${name}`,
    'application/gzip',
    immutable,
  ]),
];

function run(command, commandArgs, options = {}) {
  const proc = spawnSync(command, commandArgs, { stdio: options.stdio ?? 'inherit', encoding: options.encoding ?? 'utf8' });
  if (proc.status !== 0) throw new Error(`${command} ${commandArgs.join(' ')} failed${proc.stderr ? `: ${proc.stderr}` : ''}`);
  return proc;
}

function compressTarget([key, relPath]) {
  const src = join(repoRoot, relPath);
  if (!existsSync(src)) throw new Error(`missing source for ${key}: ${relPath}`);
  const out = join(outRoot, key);
  mkdirSync(dirname(out), { recursive: true });
  run('brotli', ['-f', '-q', '11', src, '-o', out], { stdio: dryRun ? 'pipe' : 'inherit' });
  return { key, src, out, rawBytes: statSync(src).size, brBytes: statSync(out).size };
}

function uploadTarget([key, _relPath, contentType, cacheControl]) {
  const file = join(outRoot, key);
  if (!existsSync(file)) throw new Error(`missing Brotli file for ${key}: ${file}`);
  const wranglerArgs = [
    'wrangler', 'r2', 'object', 'put', `${bucket}/${key}`,
    '--file', file,
    '--content-type', contentType,
    '--content-encoding', 'br',
    '--cache-control', cacheControl,
    remote === '1' ? '--remote' : '--local',
  ];
  if (dryRun) console.log(`[dry-run] npx ${wranglerArgs.join(' ')}`);
  else run('npx', wranglerArgs);
}

async function verifyTarget([key, relPath, contentType]) {
  const rawBytes = statSync(join(repoRoot, relPath)).size;
  const response = await fetch(`${publicBase.replace(/\/$/, '')}/${key}`, {
    headers: {
      Origin: verifyOrigin,
      'Accept-Encoding': 'br',
    },
  });
  const decodedBytes = (await response.arrayBuffer()).byteLength;
  const type = response.headers.get('content-type') ?? '';
  const row = {
    key,
    status: response.status,
    encoding: response.headers.get('content-encoding'),
    type,
    length: Number(response.headers.get('content-length') ?? 0),
    cors: response.headers.get('access-control-allow-origin'),
    decodedBytes,
    rawBytes,
  };
  console.log(JSON.stringify(row));
  if (!response.ok) throw new Error(`${key}: HTTP ${response.status}`);
  if (row.encoding !== 'br') throw new Error(`${key}: expected content-encoding br, got ${row.encoding}`);
  if (!type.toLowerCase().startsWith(contentType.split(';')[0].toLowerCase())) throw new Error(`${key}: expected content-type ${contentType}, got ${type}`);
  if (row.cors !== '*' && row.cors !== verifyOrigin) throw new Error(`${key}: expected CORS * or ${verifyOrigin}, got ${row.cors}`);
  if (decodedBytes !== rawBytes) throw new Error(`${key}: decoded ${decodedBytes} bytes, expected ${rawBytes}`);
  return row;
}

const rows = [];
if (!verifyOnly) {
  mkdirSync(outRoot, { recursive: true });
  for (const target of targets) rows.push(compressTarget(target));
  const totalRaw = rows.reduce((sum, row) => sum + row.rawBytes, 0);
  const totalBr = rows.reduce((sum, row) => sum + row.brBytes, 0);
  console.error(JSON.stringify({ phase: 'compressed', count: rows.length, totalRaw, totalBr, saved: totalRaw - totalBr, ratio: totalBr / totalRaw }, null, 2));
  if (!skipUpload) for (const target of targets) uploadTarget(target);
}

if (!dryRun) {
  const verified = [];
  for (const target of targets) verified.push(await verifyTarget(target));
  writeFileSync(join(outRoot, 'verification.json'), JSON.stringify({ publicBase, verified }, null, 2) + '\n');
}
