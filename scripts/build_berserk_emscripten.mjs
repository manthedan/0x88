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
// Berserk's Emscripten `.data` is the raw NNUE packaged verbatim by
// file_packager, so the full raw-network digest is exactly the documented
// processed `.data` SHA-256 in docs/engine_artifact_distribution.md. The
// upstream filename embeds the first 12 hex digits of that same digest.
const netSha256 = process.env.BERSERK_NET_SHA256 ?? '9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084';
const patchPath = path.resolve(process.env.BERSERK_PATCH ?? path.join(root, 'patches', 'berserk-emscripten.patch'));
const jsOut = path.resolve(process.env.BERSERK_EMSCRIPTEN_JS_OUT ?? path.join(root, 'public', 'berserk', 'berserk-emscripten.js'));
const outBase = path.basename(jsOut, '.js');
// Every SIMD variant preloads the identical NNUE, so all variants share one
// canonical `.data` (see src/lc0/berserkVariants.ts BERSERK_EMSCRIPTEN_DATA_URL).
const sharedDataPath = path.resolve(process.env.BERSERK_SHARED_DATA ?? path.join(path.dirname(jsOut), 'berserk-emscripten.data'));
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

function sha256(filePath) {
  return createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

/**
 * The Berserk network is NOT redistributed from this repository: its
 * license/provenance is unresolved (docs/engine_artifact_distribution.md), so
 * the build fetches it from upstream on demand. Both the full SHA-256 and the
 * hash prefix embedded in the upstream filename must match, and a bad download
 * is deleted so a retry cannot reuse it.
 */
function fetchAndVerifyNetwork() {
  if (!fs.existsSync(netPath)) {
    console.log(`Berserk network ${netName} is not present locally; fetching from ${netUrl}`);
    run('curl', ['-L', '--fail', '--retry', '3', '-o', netPath, netUrl]);
  }
  const digest = sha256(netPath);
  const actualName = `berserk-${digest.slice(0, 12)}.nn`;
  const problems = [];
  if (netSha256 && digest !== netSha256) problems.push(`expected SHA-256 ${netSha256}, got ${digest}`);
  if (actualName !== netName) problems.push(`expected filename ${netName} for this content, got ${actualName}`);
  if (problems.length) {
    fs.rmSync(netPath, { force: true });
    throw new Error(`Berserk network verification FAILED for ${netUrl}\n  ${problems.join('\n  ')}\nThe unverified download was deleted. Refusing to build against an unknown network.`);
  }
  console.log(`Verified Berserk network ${netName} (sha256 ${digest})`);
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
        `Shared Berserk .data mismatch: ${outBase}.data (sha256 ${builtDigest}) differs from ${sharedDataPath} (sha256 ${sharedDigest}).\n` +
        'All Berserk variants share one .data URL, so they must preload identical bytes. Rebuild every variant from the same network, or delete the stale shared .data and rebuild.',
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

fetchAndVerifyNetwork();
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

// SIMD builds compile Berserk's own SSE4.1 NNUE path through Emscripten's SSE
// intrinsic emulation headers; the relaxed build additionally swaps the
// patched m128 dpbusd helpers to the relaxed integer dot.
const simdFlags = process.env.BERSERK_WASM_RELAXED_SIMD === '1'
  ? ['-msse4.1', '-msimd128', '-mrelaxed-simd']
  : process.env.BERSERK_WASM_SIMD === '1'
    ? ['-msse4.1', '-msimd128']
    : [];

const emccArgs = [
  '-std=gnu11',
  '-Wall',
  '-Wextra',
  '-Wshadow',
  process.env.BERSERK_EMSCRIPTEN_OPT ?? '-O2',
  ...simdFlags,
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

// The .js glue and .wasm stay per-variant; only the preload .data is shared.
for (const ext of ['js', 'wasm']) {
  const built = path.join(srcDir, `${outBase}.${ext}`);
  const out = path.join(path.dirname(jsOut), `${outBase}.${ext}`);
  fs.copyFileSync(built, out);
  const size = fs.statSync(out).size;
  console.log(`Wrote ${out} (${size} bytes)`);
}
publishSharedData(path.join(srcDir, `${outBase}.data`));
