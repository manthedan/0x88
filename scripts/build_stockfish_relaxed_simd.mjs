#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync, copyFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourceTar = resolve(repoRoot, 'public/stockfish/stockfish-18.0.7-corresponding-source.tar.gz');
const workRoot = resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/repro');
const sourceDir = resolve(workRoot, 'source');
const outDir = resolve(workRoot, 'dist');
const dockerImage = process.env.EMSDK_DOCKER_IMAGE || 'emscripten/emsdk:3.1.40';
const dockerPlatform = process.env.DOCKER_PLATFORM || 'linux/amd64';

function run(cmd, args, options = {}) {
  console.log(`+ ${cmd} ${args.join(' ')}`);
  const result = spawnSync(cmd, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function replaceOnce(path, oldText, newText) {
  const text = readFileSync(path, 'utf8');
  const count = text.split(oldText).length - 1;
  if (count !== 1) throw new Error(`${path}: expected exactly one match, found ${count}`);
  writeFileSync(path, text.replace(oldText, newText));
}

function replaceAll(path, oldText, newText) {
  const text = readFileSync(path, 'utf8');
  if (!text.includes(oldText)) throw new Error(`${path}: missing ${JSON.stringify(oldText)}`);
  writeFileSync(path, text.split(oldText).join(newText));
}

function ensureLiteNet() {
  const liteNets = readFileSync(resolve(sourceDir, 'src/lite_nets.h'), 'utf8');
  const net = liteNets.match(/#define EvalFileDefaultNameBig "([^"]+)"/)?.[1];
  if (!net) throw new Error('Could not find lite EvalFileDefaultNameBig');
  const netPath = resolve(sourceDir, 'src', net);
  if (existsSync(netPath)) return;

  const localCandidates = [
    process.env.STOCKFISH_NNUE_PATH,
    resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/source/upstream/src', net),
    resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/source/docker-3140-relaxed/src', net),
  ].filter(Boolean);
  for (const candidate of localCandidates) {
    if (!existsSync(candidate)) continue;
    const hash = createHash('sha256').update(readFileSync(candidate)).digest('hex').slice(0, 12);
    if (net !== `nn-${hash}.nnue`) throw new Error(`${candidate} sha prefix was ${hash}, expected ${net}`);
    copyFileSync(candidate, netPath);
    return;
  }

  for (const url of [
    `https://tests.stockfishchess.org/api/nn/${net}`,
    `https://github.com/official-stockfish/networks/raw/master/${net}`,
  ]) {
    const tmp = `${netPath}.tmp`;
    const result = spawnSync('curl', ['-fsSL', url, '-o', tmp], { stdio: 'inherit' });
    if (result.status !== 0) continue;
    const hash = createHash('sha256').update(readFileSync(tmp)).digest('hex').slice(0, 12);
    if (net !== `nn-${hash}.nnue`) {
      console.warn(`Downloaded ${url} but sha prefix was ${hash}, expected ${net}`);
      rmSync(tmp, { force: true });
      continue;
    }
    copyFileSync(tmp, netPath);
    rmSync(tmp, { force: true });
    return;
  }
  throw new Error(`Failed to download ${net}; set STOCKFISH_NNUE_PATH=/path/to/${net} and retry`);
}

function patchSource() {
  const wasmMk = resolve(sourceDir, 'src/emscripten/wasm-makefile.mk');
  replaceOnce(wasmMk, 'EM_CXXFLAGS += -msimd128', 'EM_CXXFLAGS += -msimd128 -mrelaxed-simd');
  replaceOnce(wasmMk, 'EM_LDFLAGS  += --closure 1', 'EM_LDFLAGS  += --closure 0');
  replaceOnce(
    wasmMk,
    'EM_LDFLAGS  += -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=134217728 -s MAXIMUM_MEMORY=2147483648 -Wno-pthreads-mem-growth\n',
    'EM_LDFLAGS  += -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=134217728 -s MAXIMUM_MEMORY=2147483648 -Wno-pthreads-mem-growth\nEM_LDFLAGS  += -s STACK_SIZE=1048576\n',
  );
  replaceOnce(
    wasmMk,
    '\tEM_LDFLAGS  += -s ASYNCIFY=1\n',
    '\tEM_LDFLAGS  += -s ASYNCIFY=1\n\tEM_LDFLAGS  += -s ASYNCIFY_IMPORTS=["emscripten_utils_getline_impl"]\n',
  );

  // Emscripten 3.1.40 clang rejects this obsolete flag that the stockfish.js
  // Makefile still adds for wasm clang-like builds.
  replaceAll(resolve(sourceDir, 'src/Makefile'), '\t\t\tCXXFLAGS += -fexperimental-new-pass-manager\n', '');

  const simd = resolve(sourceDir, 'src/nnue/simd.h');
  replaceOnce(
    simd,
    '#elif defined(USE_SSE2)\n    #include <emmintrin.h>\n\n#elif defined(USE_NEON)',
    '#elif defined(USE_SSE2)\n    #include <emmintrin.h>\n\n#endif\n\n#if defined(__wasm_relaxed_simd__)\n    #include <wasm_simd128.h>\n#endif\n\n#if defined(USE_NEON)',
  );
  replaceOnce(
    simd,
    `[[maybe_unused]] static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {\n\n    __m128i product0 = _mm_maddubs_epi16(a, b);\n    product0         = _mm_madd_epi16(product0, _mm_set1_epi16(1));\n    acc              = _mm_add_epi32(acc, product0);\n}\n`,
    `[[maybe_unused]] static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {\n\n    #if defined(__wasm_relaxed_simd__)\n    // Stockfish's SSSE3 path computes unsigned activations (a, clipped to 0..127)\n    // times signed weights (b). WebAssembly relaxed dot leaves lanes with the high\n    // bit set in the second operand implementation-defined, so keep activations as\n    // the second/i7 operand and weights as the fully signed first operand.\n    v128_t weights = reinterpret_cast<v128_t>(b);\n    v128_t act_i7  = reinterpret_cast<v128_t>(a);\n    v128_t sum     = reinterpret_cast<v128_t>(acc);\n    acc = reinterpret_cast<__m128i>(wasm_i32x4_relaxed_dot_i8x16_i7x16_add(weights, act_i7, sum));\n    #else\n    __m128i product0 = _mm_maddubs_epi16(a, b);\n    product0         = _mm_madd_epi16(product0, _mm_set1_epi16(1));\n    acc              = _mm_add_epi32(acc, product0);\n    #endif\n}\n`,
  );
}

if (!existsSync(sourceTar)) throw new Error(`missing ${sourceTar}`);
rmSync(workRoot, { recursive: true, force: true });
mkdirSync(sourceDir, { recursive: true });
mkdirSync(outDir, { recursive: true });
run('tar', [
  '-xzf', sourceTar,
  '-C', sourceDir,
  '--strip-components=3',
  'stockfish-stockfish-js-18.0.7-corresponding-source/upstream/stockfish-js-32d4b5ae40c01db88219bfbe2b82dbe6dec93832',
]);
patchSource();
ensureLiteNet();
run('docker', [
  'run', '--rm', '--platform', dockerPlatform,
  '--user', `${process.getuid?.() ?? 0}:${process.getgid?.() ?? 0}`,
  '-v', `${sourceDir}:/src`, '-w', '/src', dockerImage,
  'bash', '-lc', 'node build.js --lite --single-threaded --no-split --skip-em-check --force --no-minify',
]);
copyFileSync(resolve(sourceDir, 'src/stockfish-18-lite-single.js'), resolve(outDir, 'stockfish-18-lite-single-relaxed.js'));
copyFileSync(resolve(sourceDir, 'src/stockfish-18-lite-single.wasm'), resolve(outDir, 'stockfish-18-lite-single-relaxed.wasm'));
console.log(`Wrote ${outDir}/stockfish-18-lite-single-relaxed.{js,wasm}`);
