#!/usr/bin/env node
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const engineDir = path.resolve(process.env.PLENTYCHESS_BUILD_DIR ?? path.join(root, '.local_engines', 'plentychess-emscripten-src'));
const netDir = path.resolve(process.env.PLENTYCHESS_NET_DIR ?? path.join(root, '.local_engines', 'plentychess-nets'));
const repo = process.env.PLENTYCHESS_REPO ?? 'https://github.com/Yoshie2000/PlentyChess.git';
const ref = process.env.PLENTYCHESS_REF ?? '58d8ba2505ae2b49f48dd410d214a457d15c12c6';
const netId = process.env.PLENTYCHESS_NET_ID ?? '0134-2r24-s0';
const netName = process.env.PLENTYCHESS_NETWORK ?? `${netId}.bin`;
const netUrl = process.env.PLENTYCHESS_NET_URL ?? `https://github.com/Yoshie2000/PlentyNetworks/releases/download/${netId}/${netName}`;
const netSha256 = process.env.PLENTYCHESS_NET_SHA256 ?? '550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061';
const processedSha256 = process.env.PLENTYCHESS_PROCESSED_SHA256 ?? '691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4';
const patchPath = path.resolve(process.env.PLENTYCHESS_PATCH ?? path.join(root, 'patches', 'plentychess-emscripten.patch'));
const jsOut = path.resolve(process.env.PLENTYCHESS_EMSCRIPTEN_JS_OUT ?? path.join(root, 'public', 'plentychess', 'plentychess-emscripten.js'));
const outBase = path.basename(jsOut, '.js');
const processedName = process.env.PLENTYCHESS_PROCESSED_NETWORK ?? 'processed.bin';
const processedPath = path.join(engineDir, processedName);
const emsdkImage = process.env.PLENTYCHESS_EMSDK_IMAGE ?? 'emscripten/emsdk:latest';
const nativeArch = process.env.PLENTYCHESS_NATIVE_ARCH ?? (process.arch === 'arm64' ? 'arm64' : 'generic');
const skipGit = process.env.PLENTYCHESS_SKIP_GIT === '1';

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function canRun(command, args = ['--version']) {
  const result = spawnSync(command, args, { stdio: 'ignore' });
  return result.status === 0;
}

function sha256(filePath) {
  return createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function verifySha256(filePath, expected, label) {
  if (!expected) return;
  const actual = sha256(filePath);
  if (actual !== expected) {
    throw new Error(`${label} checksum mismatch: expected ${expected}, got ${actual}`);
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
} else if (!fs.existsSync(path.join(engineDir, 'src')) || !fs.existsSync(path.join(engineDir, 'tools'))) {
  throw new Error(`PLENTYCHESS_SKIP_GIT=1 requires an unpacked PlentyChess source tree at ${engineDir}`);
}

const netPath = path.join(netDir, netName);
if (!fs.existsSync(netPath)) {
  run('curl', ['-L', '--fail', '-o', netPath, netUrl]);
}
verifySha256(netPath, netSha256, netName);
fs.copyFileSync(netPath, path.join(engineDir, netName));

// PlentyChess preprocesses the downloaded network before compiling. This is a
// native helper, not the browser target. Keep it separate from the Emscripten
// compile so the wasm build can preload processed.bin instead of relying on
// incbin assembler directives that do not work for wasm.
run('make', ['-C', 'tools', 'clean'], { cwd: engineDir });
run('make', ['-C', 'tools', `arch=${nativeArch}`], { cwd: engineDir });
run(path.join('.', 'tools', 'process_net'), ['false', netName, processedName], { cwd: engineDir });
verifySha256(processedPath, processedSha256, processedName);

run('git', ['apply', '--unidiff-zero', '--ignore-space-change', '--ignore-whitespace', patchPath], { cwd: engineDir });

const sources = [
  'src/engine.cpp',
  'src/board.cpp',
  'src/move.cpp',
  'src/uci.cpp',
  'src/search.cpp',
  'src/thread.cpp',
  'src/evaluation.cpp',
  'src/tt.cpp',
  'src/magic.cpp',
  'src/bitboard.cpp',
  'src/history.cpp',
  'src/nnue.cpp',
  'src/time.cpp',
  'src/spsa.cpp',
  'src/zobrist.cpp',
  'src/datagen.cpp',
  'src/threat-inputs.cpp',
  'src/debug.cpp',
  'src/fathom/src/tbprobe.c',
];

const emccArgs = [
  '-std=c++17',
  process.env.PLENTYCHESS_EMSCRIPTEN_OPT ?? '-O2',
  '-DNDEBUG',
  '-DARCH_X86',
  '-DPLENTY_SYNC_SEARCH',
  '-DTB_NO_THREADS',
  `-DEVALFILE="${processedName}"`,
  '-msimd128',
  '-mssse3',
  '-s',
  'USE_PTHREADS=0',
  '-s',
  'MODULARIZE=1',
  '-s',
  'EXPORT_NAME="PlentyChess"',
  '-s',
  'ENVIRONMENT=node,web,worker',
  '-s',
  'ALLOW_MEMORY_GROWTH=1',
  '-s',
  `INITIAL_MEMORY=${process.env.PLENTYCHESS_INITIAL_MEMORY ?? '268435456'}`,
  '-s',
  `MAXIMUM_MEMORY=${process.env.PLENTYCHESS_MAXIMUM_MEMORY ?? '2147483648'}`,
  '-s',
  `STACK_SIZE=${process.env.PLENTYCHESS_STACK_SIZE ?? '67108864'}`,
  '-s',
  'EXIT_RUNTIME=0',
  '-s',
  'EXPORTED_RUNTIME_METHODS=ccall',
  '-s',
  'EXPORTED_FUNCTIONS=["_main","_command","_isReady","_isSearching"]',
  '--preload-file',
  `${processedName}@/${processedName}`,
  ...sources,
  '-o',
  `${outBase}.js`,
];

if (process.env.PLENTYCHESS_EMXX || canRun('em++')) {
  run(process.env.PLENTYCHESS_EMXX ?? 'em++', emccArgs, { cwd: engineDir });
} else {
  run('docker', ['run', '--rm', '-v', `${engineDir}:/src`, '-w', '/src', emsdkImage, 'em++', ...emccArgs]);
}

for (const ext of ['js', 'wasm', 'data']) {
  const built = path.join(engineDir, `${outBase}.${ext}`);
  const out = path.join(path.dirname(jsOut), `${outBase}.${ext}`);
  fs.copyFileSync(built, out);
  const size = fs.statSync(out).size;
  console.log(`Wrote ${out} (${size} bytes)`);
}
