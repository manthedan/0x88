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
// Every SIMD variant preloads the identical processed NNUE, so all variants
// share one canonical `.data` (src/lc0/plentychessVariants.ts
// PLENTYCHESS_EMSCRIPTEN_DATA_URL).
const sharedDataPath = path.resolve(process.env.PLENTYCHESS_SHARED_DATA ?? path.join(path.dirname(jsOut), 'plentychess-emscripten.data'));
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

/**
 * Emscripten emits one `.data` per variant, but the preloaded NNUE is
 * byte-identical across SIMD tiers. Publish exactly one copy so browsers and
 * the CDN never hold duplicates, and fail loudly if a rebuild would have
 * changed the shared bytes.
 */
function publishSharedData(builtDataPath) {
  const builtDigest = sha256(builtDataPath);
  if (fs.existsSync(sharedDataPath)) {
    const sharedDigest = sha256(sharedDataPath);
    if (sharedDigest !== builtDigest) {
      throw new Error(
        `Shared PlentyChess .data mismatch: ${outBase}.data (sha256 ${builtDigest}) differs from ${sharedDataPath} (sha256 ${sharedDigest}).\n` +
        'All PlentyChess variants share one .data URL, so they must preload identical bytes. Rebuild every variant from the same network, or delete the stale shared .data and rebuild.',
      );
    }
    console.log(`Shared ${sharedDataPath} already matches ${outBase}.data (sha256 ${builtDigest})`);
  } else {
    fs.copyFileSync(builtDataPath, sharedDataPath);
    console.log(`Wrote ${sharedDataPath} (${fs.statSync(sharedDataPath).size} bytes, sha256 ${builtDigest})`);
  }
  const perVariantData = path.join(path.dirname(jsOut), `${outBase}.data`);
  if (perVariantData !== sharedDataPath && fs.existsSync(perVariantData)) {
    fs.rmSync(perVariantData);
    console.log(`Removed duplicate ${perVariantData}; all variants load ${path.basename(sharedDataPath)}`);
  }
}

fs.mkdirSync(path.dirname(engineDir), { recursive: true });
fs.mkdirSync(netDir, { recursive: true });
fs.mkdirSync(path.dirname(jsOut), { recursive: true });

if (!skipGit) {
  // Shallow pinned-ref fetch (same pattern as build_reckless_wasi.mjs): the
  // full clone + --tags fetch was flaky over slow links and pulls far more
  // history than a pinned corresponding-source build needs.
  if (!fs.existsSync(path.join(engineDir, '.git'))) {
    fs.mkdirSync(engineDir, { recursive: true });
    run('git', ['init'], { cwd: engineDir });
    run('git', ['remote', 'add', 'origin', repo], { cwd: engineDir });
  }
  run('git', ['fetch', '--depth=1', 'origin', ref], { cwd: engineDir });
  run('git', ['checkout', '--detach', 'FETCH_HEAD'], { cwd: engineDir });
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

// The default build keeps -mssse3 (the engine's SSSE3 dpbusd path). The SSE4.1
// build additionally engages the patched convertEpi8Epi16 gate (exact-equal
// single-op sign extension); the relaxed build adds the relaxed integer dot
// for dpbusd and the relaxed-madd vectorized f32 tail (the FMA/AVX2/ARM gates
// in nnue.cpp are never true under emcc, so the f32 layers otherwise run
// scalar std::fma loops).
const simdFlags = process.env.PLENTYCHESS_WASM_RELAXED_SIMD === '1'
  ? ['-msse4.1', '-mrelaxed-simd']
  : process.env.PLENTYCHESS_WASM_SSE41 === '1'
    ? ['-msse4.1']
    : [];

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
  ...simdFlags,
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

// Recovery model for an interrupted build is re-running it, not atomicity.
// A crash between publishing the canonical .data and copying the glue/wasm can
// leave a mixed set, but every path here is regenerated from the pinned
// upstream source on the next run, publishSharedData re-verifies the canonical
// bytes and fails loudly on divergence, and nothing deploys straight from this
// directory without the release gate. Staging all three outputs behind a
// temp-and-rename commit was considered and rejected as machinery that buys
// less than it costs in a developer-only build script.
// Validate and publish the shared preload FIRST. publishSharedData throws when
// a rebuilt variant's .data diverges from the canonical one, and if the glue and
// wasm were already in place by then the public tree would be left holding new
// code beside a stale network -- a mixed set that still looks deployable.
publishSharedData(path.join(engineDir, `${outBase}.data`));
for (const ext of ['js', 'wasm']) {
  const built = path.join(engineDir, `${outBase}.${ext}`);
  const out = path.join(path.dirname(jsOut), `${outBase}.${ext}`);
  fs.copyFileSync(built, out);
  console.log(`Wrote ${out} (${fs.statSync(out).size} bytes)`);
}
