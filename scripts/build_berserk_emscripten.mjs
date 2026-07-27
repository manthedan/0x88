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
 * The release distributes this network in both the Emscripten preload and the
 * corresponding-source archive. The build still fetches it from upstream on
 * demand so a clean source build is reproducible. Both the full SHA-256 and the
 * hash prefix embedded in the filename must match, and a bad download is
 * deleted so a retry cannot reuse it.
 */
function fetchAndVerifyNetwork() {
  if (!fs.existsSync(netPath)) {
    console.log(`Berserk network ${netName} is not present locally; fetching from ${netUrl}`);
    // Download to a temp path and rename only on success. curl writes
    // incrementally and `run` exits the process on a nonzero status, so
    // fetching straight to netPath would leave a truncated file behind that the
    // next build treats as cached, skips re-downloading, and then fails to
    // verify. Anything left at the temp path is never read, and is cleared
    // before the next attempt; the rename is the commit point. No try/finally:
    // run() exits the process on failure, so nothing after it would run anyway.
    const tmpPath = `${netPath}.download`;
    fs.rmSync(tmpPath, { force: true });
    run('curl', ['-L', '--fail', '--retry', '3', '-o', tmpPath, netUrl]);
    fs.renameSync(tmpPath, netPath);
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
    // Copy via a temp file and rename, so the canonical path only ever holds
    // complete bytes. Writing straight to it means an interrupt mid-copy leaves
    // a truncated file that every later run then hashes, reports as a mismatch,
    // and throws on WITHOUT replacing -- wedging all future builds until someone
    // deletes it by hand. rename(2) within one directory is atomic.
    const tmpDataPath = `${sharedDataPath}.partial`;
    fs.rmSync(tmpDataPath, { force: true });
    fs.copyFileSync(builtDataPath, tmpDataPath);
    fs.renameSync(tmpDataPath, sharedDataPath);
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

// Publish the whole variant as one step.
//
// Order matters first: publishSharedData throws when a rebuilt variant's .data
// diverges from the canonical one, so it runs before any code is placed --
// otherwise the tree would be left holding new glue beside a stale network.
//
// Atomicity matters second: the .js and .wasm must land together. A partial
// copy leaves new glue with an old wasm, which is an Emscripten LinkError at
// load, and release manifests hash whatever is on disk independently, so a
// mixed directory can be packaged and shipped. Re-running does not protect
// against that because the mixed state is already publishable. So stage both
// into the output directory under temp names, then rename them into place --
// rename(2) within a directory is atomic, and the window where only one of the
// two is live is a single syscall rather than a whole file copy.
publishSharedData(path.join(srcDir, `${outBase}.data`));
const staged = [];
for (const ext of ['js', 'wasm']) {
  const built = path.join(srcDir, `${outBase}.${ext}`);
  const out = path.join(path.dirname(jsOut), `${outBase}.${ext}`);
  const tmp = `${out}.partial`;
  fs.rmSync(tmp, { force: true });
  fs.copyFileSync(built, tmp);
  staged.push([tmp, out]);
}
for (const [tmp, out] of staged) {
  fs.renameSync(tmp, out);
  console.log(`Wrote ${out} (${fs.statSync(out).size} bytes)`);
}
// POSIX has no multi-file atomic rename, so the glue and wasm land in two
// adjacent syscalls; a crash between them is a microsecond window, and closing
// it properly would mean a versioned-directory swap that changes the output
// layout every consumer of public/<engine>/<name>.js depends on. That is not
// worth it here. What IS worth it is refusing to leave a mixed pair quietly:
// verify the published set before exiting, so an interrupted or failed publish
// is loud rather than something release tooling later hashes and ships.
for (const [, out] of staged) {
  if (!fs.existsSync(out) || fs.statSync(out).size === 0) {
    throw new Error(`Publish incomplete: ${out} is missing or empty. The output directory now holds a mixed artifact set; re-run this build before packaging or deploying.`);
  }
}
