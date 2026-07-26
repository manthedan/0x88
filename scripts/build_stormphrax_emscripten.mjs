#!/usr/bin/env node
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const engineDir = path.resolve(process.env.STORMPHRAX_BUILD_DIR ?? path.join(root, '.local_engines', 'stormphrax-emscripten-src'));
const netDir = path.resolve(process.env.STORMPHRAX_NET_DIR ?? path.join(root, '.local_engines', 'stormphrax-nets'));
const repo = process.env.STORMPHRAX_REPO ?? 'https://github.com/Ciekce/Stormphrax.git';
const ref = process.env.STORMPHRAX_REF ?? '582965517ed2032d41a6b4cd6c2e66b1b934e2ad';
const version = process.env.STORMPHRAX_VERSION ?? '8.0.0';
const netName = process.env.STORMPHRAX_NETWORK ?? 'undertown.nnue';
const netUrl = process.env.STORMPHRAX_NET_URL ?? `https://github.com/Ciekce/stormphrax-nets/releases/download/undertown/${netName}`;
const netSha256 = process.env.STORMPHRAX_NET_SHA256 ?? '04d651e078b7c7334709dbd772d40a23c0a5480e93e19521a03020c7d633f2cf';
const patchPath = path.resolve(process.env.STORMPHRAX_PATCH ?? path.join(root, 'patches', 'stormphrax-emscripten.patch'));
const jsOut = path.resolve(process.env.STORMPHRAX_EMSCRIPTEN_JS_OUT ?? path.join(root, 'public', 'stormphrax', 'stormphrax-emscripten.js'));
const outBase = path.basename(jsOut, '.js');
// Every SIMD variant preloads the identical undertown NNUE, so all variants
// share one canonical `.data` (src/lc0/stormphraxVariants.ts
// STORMPHRAX_EMSCRIPTEN_DATA_URL).
const sharedDataPath = path.resolve(process.env.STORMPHRAX_SHARED_DATA ?? path.join(path.dirname(jsOut), 'stormphrax-emscripten.data'));
const emsdkImage = process.env.STORMPHRAX_EMSDK_IMAGE ?? 'emscripten/emsdk:6.0.2';
const skipGit = process.env.STORMPHRAX_SKIP_GIT === '1';
const relaxedSimd = process.env.STORMPHRAX_WASM_RELAXED_SIMD === '1';
const relaxedSimdAudit = process.env.STORMPHRAX_RELAXED_SIMD_AUDIT === '1';

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function canRun(command, args = ['--version']) {
  return spawnSync(command, args, { stdio: 'ignore' }).status === 0;
}

function sha256(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
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
        `Shared Stormphrax .data mismatch: ${outBase}.data (sha256 ${builtDigest}) differs from ${sharedDataPath} (sha256 ${sharedDigest}).\n` +
        'All Stormphrax variants share one .data URL, so they must preload identical bytes. Rebuild every variant from the same network, or delete the stale shared .data and rebuild.',
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

function walkSources(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) walkSources(file, out);
    else if (entry.isFile() && entry.name.endsWith('.cpp') && entry.name !== 'numa_libnuma.cpp') out.push(path.relative(engineDir, file));
  }
  return out;
}

fs.mkdirSync(engineDir, { recursive: true });
fs.mkdirSync(netDir, { recursive: true });
fs.mkdirSync(path.dirname(jsOut), { recursive: true });

if (!skipGit) {
  if (!fs.existsSync(path.join(engineDir, '.git'))) {
    run('git', ['init'], { cwd: engineDir });
    run('git', ['remote', 'add', 'origin', repo], { cwd: engineDir });
  }
  run('git', ['fetch', '--depth=1', 'origin', ref], { cwd: engineDir });
  run('git', ['checkout', '--detach', 'FETCH_HEAD'], { cwd: engineDir });
  run('git', ['reset', '--hard'], { cwd: engineDir });
  run('git', ['clean', '-fdx'], { cwd: engineDir });
} else if (!fs.existsSync(path.join(engineDir, 'src'))) {
  throw new Error(`STORMPHRAX_SKIP_GIT=1 requires an unpacked Stormphrax source tree at ${engineDir}`);
}

const netPath = path.join(netDir, netName);
if (!fs.existsSync(netPath)) run('curl', ['-L', '--fail', '--retry', '3', '-o', netPath, netUrl]);
const actualNetSha256 = sha256(netPath);
if (actualNetSha256 !== netSha256) throw new Error(`${netName} checksum mismatch: expected ${netSha256}, got ${actualNetSha256}`);
fs.copyFileSync(netPath, path.join(engineDir, netName));

run('git', ['apply', '--ignore-space-change', '--ignore-whitespace', patchPath], { cwd: engineDir });

const sources = [
  ...walkSources(path.join(engineDir, 'src')).sort(),
  '3rdparty/fmt/src/format.cc',
  '3rdparty/pyrrhic/tbprobe.cpp',
  '3rdparty/zstd/zstddeclib.c',
];
const emxxArgs = [
  '-std=c++20',
  process.env.STORMPHRAX_EMSCRIPTEN_OPT ?? '-O2',
  '-DNDEBUG',
  `-DSP_VERSION=${version}`,
  '-DSP_WASM_SIMD',
  ...(relaxedSimdAudit ? ['-DSP_RELAXED_SIMD_AUDIT'] : []),
  '-DSP_SYNC_SEARCH',
  `-DSP_NETWORK_FILE="/${netName}"`,
  '-I3rdparty/fmt/include',
  '-msimd128',
  '-mssse3',
  '-msse4.1',
  ...(relaxedSimd ? ['-mrelaxed-simd'] : []),
  '-s', 'USE_PTHREADS=0',
  '-s', 'MODULARIZE=1',
  '-s', 'EXPORT_NAME="Stormphrax"',
  '-s', 'ENVIRONMENT=node,web,worker',
  '-s', 'ALLOW_MEMORY_GROWTH=1',
  '-s', `INITIAL_MEMORY=${process.env.STORMPHRAX_INITIAL_MEMORY ?? '536870912'}`,
  '-s', `MAXIMUM_MEMORY=${process.env.STORMPHRAX_MAXIMUM_MEMORY ?? '2147483648'}`,
  '-s', `STACK_SIZE=${process.env.STORMPHRAX_STACK_SIZE ?? '67108864'}`,
  '-s', 'EXIT_RUNTIME=0',
  '-s', 'EXPORTED_RUNTIME_METHODS=ccall',
  '-s', 'EXPORTED_FUNCTIONS=["_main","_command","_isReady","_isSearching"]',
  '--preload-file', `${netName}@/${netName}`,
  ...sources,
  '-o', `${outBase}.js`,
];

if (process.env.STORMPHRAX_EMXX || canRun('em++')) {
  run(process.env.STORMPHRAX_EMXX ?? 'em++', emxxArgs, { cwd: engineDir });
} else {
  run('docker', ['run', '--rm', '-v', `${engineDir}:/src`, '-w', '/src', emsdkImage, 'em++', ...emxxArgs]);
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
publishSharedData(path.join(engineDir, `${outBase}.data`));
const staged = [];
for (const ext of ['js', 'wasm']) {
  const built = path.join(engineDir, `${outBase}.${ext}`);
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
