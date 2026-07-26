#!/usr/bin/env node
// PROTOTYPE ONLY -- research build for docs/threaded_emscripten_smp_prototype_2026-07-25.md
//
// This is a deliberate fork of scripts/build_stormphrax_emscripten.mjs whose only
// purpose is to answer one question: how much search throughput does upstream
// Stormphrax's native Lazy SMP actually buy us under Emscripten pthreads?
//
// Differences from the shipped single-thread build, and nothing else:
//   - does NOT define -DSP_SYNC_SEARCH, so patches/stormphrax-emscripten.patch
//     falls through to upstream's real barrier-driven threaded search
//   - -pthread / -sUSE_PTHREADS=1 / -sPTHREAD_POOL_SIZE=<N> instead of
//     -s USE_PTHREADS=0
//   - -sDEFAULT_PTHREAD_STACK_SIZE (Emscripten's -sSTACK_SIZE only sizes the
//     main thread's stack; helper threads would otherwise get the 64 KiB default
//     which cannot hold Stormphrax's recursive search frames)
//   - -sPTHREAD_POOL_SIZE_STRICT=0 so exceeding the pool degrades instead of
//     aborting
// SIMD flags, opt level, memory settings, exports and --preload-file are kept
// byte-identical to the shipped script so the A/B is apples-to-apples.
//
// Output defaults to a scratch path, NOT public/, because this build is not
// shippable as-is (see the doc for why).
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
const jsOut = path.resolve(
  process.env.STORMPHRAX_EMSCRIPTEN_JS_OUT
    ?? path.join(root, '.local_engines', 'stormphrax-threaded-out', 'stormphrax-emscripten-threaded.js'),
);
const outBase = path.basename(jsOut, '.js');
const emsdkImage = process.env.STORMPHRAX_EMSDK_IMAGE ?? 'emscripten/emsdk:6.0.2';
const skipGit = process.env.STORMPHRAX_SKIP_GIT === '1';
const relaxedSimd = process.env.STORMPHRAX_WASM_RELAXED_SIMD === '1';
const relaxedSimdAudit = process.env.STORMPHRAX_RELAXED_SIMD_AUDIT === '1';
const poolSize = Math.max(1, Math.floor(Number(process.env.STORMPHRAX_PTHREAD_POOL_SIZE ?? 12)));
const pthreadStackSize = process.env.STORMPHRAX_PTHREAD_STACK_SIZE ?? '8388608';

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

// --- post-patch fixup: over-align the browser network buffer -----------------
// patches/stormphrax-emscripten.patch reads the NNUE file into a plain
// std::vector<std::byte>. NetworkLoader::get() rejects any pointer that is not
// util::simd::kAlignment (16 B) aligned. In the single-threaded build dlmalloc
// happens to hand back a 16 B-aligned block for that ~53 MB allocation; under
// dlmalloc-mt the preceding pthread bookkeeping shifts it and the engine dies
// with "NetworkLoader: Unaligned pointer / No network loaded".
//
// This is an incidental fragility of the existing browser patch, NOT something
// intrinsic to threading, so instead of forking the tracked patch we swap the
// vector for an interface-compatible shim that guarantees 64 B alignment. Only
// the storage declaration changes; resize()/data()/size() call sites compile
// unchanged and no NNUE compute path is touched.
{
  const nnuePath = path.join(engineDir, 'src', 'eval', 'nnue.cpp');
  let src = fs.readFileSync(nnuePath, 'utf8');
  const declNeedle = '    std::vector<std::byte> g_defaultNetStorage{};';
  if (!src.includes(declNeedle)) {
    throw new Error(`Alignment fixup anchor not found in ${nnuePath}; the patch layout changed.`);
  }
  const shim = `    // [threaded prototype] 64 B-overaligned stand-in for std::vector<std::byte>.
    class SpAlignedNetStorage {
    public:
        void resize(std::size_t n) {
            m_raw.assign(n + 64, std::byte{});
            const auto addr = reinterpret_cast<std::uintptr_t>(m_raw.data());
            m_offset = static_cast<std::size_t>((64 - (addr % 64)) % 64);
            m_size = n;
        }
        [[nodiscard]] std::byte* data() { return m_raw.data() + m_offset; }
        [[nodiscard]] const std::byte* data() const { return m_raw.data() + m_offset; }
        [[nodiscard]] std::size_t size() const { return m_size; }

    private:
        std::vector<std::byte> m_raw{};
        std::size_t m_offset{};
        std::size_t m_size{};
    };
    SpAlignedNetStorage g_defaultNetStorage{};`;
  src = src.replace(declNeedle, shim);
  src = src.replace('#include <vector>', '#include <cstdint>\n#include <vector>');
  fs.writeFileSync(nnuePath, src);
  console.log('Applied threaded-prototype network-buffer alignment fixup to src/eval/nnue.cpp');
}

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
  // NOTE: -DSP_SYNC_SEARCH intentionally omitted -- this is the whole point.
  `-DSP_NETWORK_FILE="/${netName}"`,
  '-I3rdparty/fmt/include',
  '-msimd128',
  '-mssse3',
  '-msse4.1',
  ...(relaxedSimd ? ['-mrelaxed-simd'] : []),
  '-pthread',
  '-s', 'USE_PTHREADS=1',
  '-s', `PTHREAD_POOL_SIZE=${poolSize}`,
  '-s', 'PTHREAD_POOL_SIZE_STRICT=0',
  '-s', `DEFAULT_PTHREAD_STACK_SIZE=${pthreadStackSize}`,
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

for (const ext of ['js', 'wasm', 'data']) {
  const built = path.join(engineDir, `${outBase}.${ext}`);
  const out = path.join(path.dirname(jsOut), `${outBase}.${ext}`);
  fs.copyFileSync(built, out);
  console.log(`Wrote ${out} (${fs.statSync(out).size} bytes)`);
}
