#!/usr/bin/env node
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const engineDir = path.resolve(process.env.BERSERK_BUILD_DIR ?? path.join(root, '.local_engines', 'berserk-emscripten-src'));
const netDir = path.resolve(process.env.BERSERK_NET_DIR ?? path.join(root, '.local_engines', 'berserk-nets'));
const repo = process.env.BERSERK_REPO ?? 'https://github.com/jhonnold/berserk.git';
const ref = process.env.BERSERK_REF ?? '8ae895a6151695be4a50d4fb65b0c131659c513a';
const netName = process.env.BERSERK_NETWORK ?? 'berserk-9b84c340af7e.nn';
const netUrl = process.env.BERSERK_NET_URL ?? `https://github.com/jhonnold/berserk-networks/releases/download/networks/${netName}`;
const patchPath = path.resolve(process.env.BERSERK_PATCH ?? path.join(root, 'patches', 'berserk-emscripten.patch'));
const jsOut = path.resolve(process.env.BERSERK_EMSCRIPTEN_JS_OUT ?? path.join(root, 'public', 'berserk', 'berserk-emscripten.js'));
const outBase = path.basename(jsOut, '.js');
const srcDir = path.join(engineDir, 'src');
const netPath = path.join(netDir, netName);
const emsdkImage = process.env.BERSERK_EMSDK_IMAGE ?? 'emscripten/emsdk:latest';
const skipGit = process.env.BERSERK_SKIP_GIT === '1';

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function canRun(command, args = ['--version']) {
  const result = spawnSync(command, args, { stdio: 'ignore' });
  return result.status === 0;
}

function verifyNetworkName(filePath, expectedName) {
  const digest = createHash('sha256').update(fs.readFileSync(filePath)).digest('hex').slice(0, 12);
  const actualName = `berserk-${digest}.nn`;
  if (actualName !== expectedName) {
    throw new Error(`Network checksum mismatch: expected ${expectedName}, got ${actualName}`);
  }
}

fs.mkdirSync(path.dirname(engineDir), { recursive: true });
fs.mkdirSync(netDir, { recursive: true });
fs.mkdirSync(path.dirname(jsOut), { recursive: true });

if (!skipGit) {
  if (!fs.existsSync(path.join(engineDir, '.git'))) {
    run('git', ['clone', repo, engineDir]);
  }
  run('git', ['fetch', '--tags', 'origin'], { cwd: engineDir });
  run('git', ['checkout', ref], { cwd: engineDir });
  run('git', ['reset', '--hard'], { cwd: engineDir });
  run('git', ['clean', '-fdx'], { cwd: engineDir });
} else if (!fs.existsSync(srcDir)) {
  throw new Error(`BERSERK_SKIP_GIT=1 requires an unpacked Berserk source tree at ${engineDir}`);
}

if (!fs.existsSync(netPath)) {
  run('curl', ['-L', '--fail', '-o', netPath, netUrl]);
}
verifyNetworkName(netPath, netName);
fs.copyFileSync(netPath, path.join(srcDir, netName));

run('git', ['apply', '--ignore-space-change', '--ignore-whitespace', patchPath], { cwd: engineDir });

const sources = [
  'attacks.c',
  'bench.c',
  'berserk.c',
  'bits.c',
  'board.c',
  'eval.c',
  'history.c',
  'move.c',
  'movegen.c',
  'movepick.c',
  'perft.c',
  'random.c',
  'search.c',
  'see.c',
  'tb.c',
  'thread.c',
  'transposition.c',
  'uci.c',
  'util.c',
  'zobrist.c',
  'nn/accumulator.c',
  'nn/evaluate.c',
];

const emccArgs = [
  '-std=gnu11',
  '-Wall',
  '-Wextra',
  '-Wshadow',
  process.env.BERSERK_EMSCRIPTEN_OPT ?? '-O2',
  '-DNDEBUG',
  '-DVERSION="14"',
  `-DEVALFILE="${netName}"`,
  '-DBERSERK_NO_TB',
  '-DBERSERK_SYNC_SEARCH',
  '-s',
  'MODULARIZE=1',
  '-s',
  'EXPORT_NAME="Berserk"',
  '-s',
  'ENVIRONMENT=web,worker,node',
  '-s',
  'ALLOW_MEMORY_GROWTH=1',
  '-s',
  `INITIAL_MEMORY=${process.env.BERSERK_INITIAL_MEMORY ?? '268435456'}`,
  '-s',
  `MAXIMUM_MEMORY=${process.env.BERSERK_MAXIMUM_MEMORY ?? '2147483648'}`,
  '-s',
  `STACK_SIZE=${process.env.BERSERK_STACK_SIZE ?? '67108864'}`,
  '-s',
  'EXIT_RUNTIME=0',
  '-s',
  'EXPORTED_RUNTIME_METHODS=ccall',
  '-s',
  'EXPORTED_FUNCTIONS=["_main","_command","_isReady","_isSearching"]',
  '--preload-file',
  `${netName}@/${netName}`,
  ...sources,
  '-lm',
  '-o',
  `${outBase}.js`,
];

if (process.env.BERSERK_EMCC || canRun('emcc')) {
  run(process.env.BERSERK_EMCC ?? 'emcc', emccArgs, { cwd: srcDir });
} else {
  run('docker', ['run', '--rm', '-v', `${srcDir}:/src`, '-w', '/src', emsdkImage, 'emcc', ...emccArgs]);
}

for (const ext of ['js', 'wasm', 'data']) {
  const built = path.join(srcDir, `${outBase}.${ext}`);
  const out = path.join(path.dirname(jsOut), `${outBase}.${ext}`);
  fs.copyFileSync(built, out);
  const size = fs.statSync(out).size;
  console.log(`Wrote ${out} (${size} bytes)`);
}
