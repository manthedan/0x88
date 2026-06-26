#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync, copyFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, dirname, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourceTar = resolve(repoRoot, 'public/stockfish/stockfish-18.0.7-corresponding-source.tar.gz');
const workRoot = resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/repro');
const sourceDir = resolve(workRoot, 'source');
const outDir = resolve(workRoot, 'dist');
const dockerImage = process.env.EMSDK_DOCKER_IMAGE || 'emscripten/emsdk:3.1.40';
const dockerPlatform = process.env.DOCKER_PLATFORM || 'linux/amd64';
const sourceInputDir = process.env.STOCKFISH_RELAXED_SOURCE_DIR ? resolve(process.env.STOCKFISH_RELAXED_SOURCE_DIR) : null;

const VARIANTS = {
  'lite-single': {
    label: 'Lite single-threaded',
    netHeader: 'lite_nets.h',
    buildArgs: ['--lite', '--single-threaded', '--no-split'],
    inputBase: 'stockfish-18-lite-single',
    outputBase: 'stockfish-18-lite-single-relaxed',
  },
  single: {
    label: 'Full single-threaded',
    netHeader: 'evaluate.h',
    buildArgs: ['--single-threaded', '--no-split'],
    inputBase: 'stockfish-18-single',
    outputBase: 'stockfish-18-single-relaxed',
  },
  'lite-threaded': {
    label: 'Lite pthread',
    netHeader: 'lite_nets.h',
    buildArgs: ['--lite', '--no-split'],
    inputBase: 'stockfish-18-lite',
    outputBase: 'stockfish-18-lite-relaxed',
  },
  threaded: {
    label: 'Full pthread',
    netHeader: 'evaluate.h',
    buildArgs: ['--no-split'],
    inputBase: 'stockfish',
    outputBase: 'stockfish-18-relaxed',
  },
};

function argValue(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function usage() {
  console.error(`Usage: node scripts/build_stockfish_relaxed_simd.mjs [--variant ${Object.keys(VARIANTS).join('|')}]

Environment:
  EMSDK_DOCKER_IMAGE             Docker image to use (default emscripten/emsdk:3.1.40)
  DOCKER_PLATFORM                Docker platform (default linux/amd64)
  STOCKFISH_RELAXED_SOURCE_DIR   Existing stockfish.js source checkout/extract to patch instead of the corresponding-source tarball
  STOCKFISH_NNUE_PATH            One required NNUE file to copy before download attempts
  STOCKFISH_NNUE_DIR             Directory containing required nn-*.nnue files
  STOCKFISH_MAKE_JOBS            make -j value inside the upstream build (use 1-2 for full nets on memory-constrained hosts)
  STOCKFISH_RELAXED_DISABLE_LTO  Set to 1 to remove upstream LTO flags for exploratory full/pthread builds
`);
}

const variantName = argValue('--variant') || process.env.STOCKFISH_RELAXED_VARIANT || 'lite-single';
const variant = VARIANTS[variantName];
if (!variant || process.argv.includes('--help') || process.argv.includes('-h')) {
  usage();
  process.exit(variant ? 0 : 1);
}

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

function shaPrefix(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex').slice(0, 12);
}

function requiredNets() {
  const header = readFileSync(resolve(sourceDir, 'src', variant.netHeader), 'utf8');
  const nets = [];
  for (const type of ['Big', 'Small']) {
    const path = header.match(new RegExp(`#define EvalFileDefaultName${type} "([^"]*)"`))?.[1];
    if (path === undefined) throw new Error(`Could not find EvalFileDefaultName${type} in ${variant.netHeader}`);
    if (path) nets.push(path);
  }
  return nets;
}

function localNetCandidates(net) {
  return [
    process.env.STOCKFISH_NNUE_PATH,
    process.env.STOCKFISH_NNUE_DIR ? resolve(process.env.STOCKFISH_NNUE_DIR, net) : null,
    resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/source/upstream/src', net),
    resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/source/docker-3140-relaxed/src', net),
    resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/source/docker-3140-relaxed-stack/src', net),
    resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/repro/source/src', net),
    resolve(repoRoot, 'node_modules/stockfish/src', net),
  ].filter(Boolean);
}

function ensureRequiredNets() {
  for (const net of requiredNets()) {
    const netPath = resolve(sourceDir, 'src', net);
    if (existsSync(netPath)) continue;

    for (const candidate of localNetCandidates(net)) {
      if (!existsSync(candidate) || basename(candidate) !== net) continue;
      const hash = shaPrefix(candidate);
      if (net !== `nn-${hash}.nnue`) throw new Error(`${candidate} sha prefix was ${hash}, expected ${net}`);
      copyFileSync(candidate, netPath);
      console.log(`Copied ${net} from ${candidate}`);
      break;
    }
    if (existsSync(netPath)) continue;

    for (const url of [
      `https://tests.stockfishchess.org/api/nn/${net}`,
      `https://github.com/official-stockfish/networks/raw/master/${net}`,
    ]) {
      const tmp = `${netPath}.tmp`;
      const result = spawnSync('curl', ['-fL', '--retry', '3', '--retry-delay', '2', url, '-o', tmp], { stdio: 'inherit' });
      if (result.status !== 0) continue;
      const hash = shaPrefix(tmp);
      if (net !== `nn-${hash}.nnue`) {
        console.warn(`Downloaded ${url} but sha prefix was ${hash}, expected ${net}`);
        rmSync(tmp, { force: true });
        continue;
      }
      copyFileSync(tmp, netPath);
      rmSync(tmp, { force: true });
      break;
    }
    if (!existsSync(netPath)) throw new Error(`Failed to download ${net}; set STOCKFISH_NNUE_DIR=/dir or STOCKFISH_NNUE_PATH=/path/to/${net} and retry`);
  }
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
  if (variantName === 'lite-threaded' || variantName === 'threaded') {
    replaceOnce(
      wasmMk,
      '\tEM_LDFLAGS  += -s PROXY_TO_PTHREAD\n\tEM_LDFLAGS  += -s USE_PTHREADS=1\n',
      '\tEM_CXXFLAGS += -pthread\n\tEM_LDFLAGS  += -pthread\n\tEM_LDFLAGS  += -s PROXY_TO_PTHREAD\n\tEM_LDFLAGS  += -s USE_PTHREADS=1\n',
    );
  }

  const makefile = resolve(sourceDir, 'src/Makefile');
  // Emscripten 3.1.40 clang rejects this obsolete flag that the stockfish.js
  // Makefile still adds for wasm clang-like builds.
  replaceAll(makefile, '\t\t\tCXXFLAGS += -fexperimental-new-pass-manager\n', '');
  if (process.env.STOCKFISH_RELAXED_DISABLE_LTO === '1') {
    replaceAll(makefile, '\t\tCXXFLAGS += -flto=full\n', '');
  }

  const buildJs = resolve(sourceDir, 'build.js');
  replaceOnce(
    buildJs,
    'var args = ["-j", require("os").cpus().length];',
    'var args = ["-j", process.env.STOCKFISH_MAKE_JOBS || require("os").cpus().length];',
  );
  replaceOnce(
    buildJs,
    'var workerExternPostData = fs.readFileSync(workerExternPostPath, "utf8");',
    'var workerExternPostData = "";',
  );
  replaceOnce(
    buildJs,
    'workerData = fs.readFileSync(stockfishWorkerThreadPath, "utf8") + workerExternPostData;',
    'workerData = fs.readFileSync(stockfishWorkerThreadPath, "utf8").replace(\'"use strict";var Module={};\', \'"use strict";var startWorker;var Module={};\').replace(\'self.startWorker=instance=>\', \'startWorker=self.startWorker=instance=>\').replace(\'if(typeof e.data.urlOrBlob=="string"){importScripts(e.data.urlOrBlob)}else{var objectUrl=URL.createObjectURL(e.data.urlOrBlob);importScripts(objectUrl);URL.revokeObjectURL(objectUrl)}Stockfish(Module)\', \'Stockfish=INIT_ENGINE();Stockfish(Module)\') + workerExternPostData;',
  );
  replaceOnce(
    buildJs,
    'stockfishWASMLoaderData = fs.readFileSync(stockfishWASMLoaderPath, "utf8").replace(/\\/\\/\\/ Insert worker here/, workerData);',
    'stockfishWASMLoaderData = fs.readFileSync(stockfishWASMLoaderPath, "utf8").replace(/\\/\\/\\/ Insert worker here/, workerData).replace(\'startWorker(Module);\', \'self.startWorker(Module);\');',
  );

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

if (!sourceInputDir && !existsSync(sourceTar)) throw new Error(`missing ${sourceTar}; set STOCKFISH_RELAXED_SOURCE_DIR=/path/to/stockfish.js/source to build from an extracted source archive`);
rmSync(workRoot, { recursive: true, force: true });
mkdirSync(sourceDir, { recursive: true });
mkdirSync(outDir, { recursive: true });
if (sourceInputDir) {
  if (!existsSync(resolve(sourceInputDir, 'build.js')) || !existsSync(resolve(sourceInputDir, 'src/Makefile'))) {
    throw new Error(`STOCKFISH_RELAXED_SOURCE_DIR does not look like stockfish.js source: ${sourceInputDir}`);
  }
  cpSync(sourceInputDir, sourceDir, { recursive: true });
} else {
  run('tar', [
    '-xzf', sourceTar,
    '-C', sourceDir,
    '--strip-components=3',
    'stockfish-stockfish-js-18.0.7-corresponding-source/upstream/stockfish-js-32d4b5ae40c01db88219bfbe2b82dbe6dec93832',
  ]);
}
patchSource();
ensureRequiredNets();
console.log(`Building Stockfish.js relaxed SIMD variant: ${variantName} (${variant.label})`);
const dockerEnv = [];
if (process.env.STOCKFISH_MAKE_JOBS) dockerEnv.push('-e', `STOCKFISH_MAKE_JOBS=${process.env.STOCKFISH_MAKE_JOBS}`);
run('docker', [
  'run', '--rm', '--platform', dockerPlatform,
  '--user', `${process.getuid?.() ?? 0}:${process.getgid?.() ?? 0}`,
  ...dockerEnv,
  '-v', `${sourceDir}:/src`, '-w', '/src', dockerImage,
  'bash', '-lc', ['node', 'build.js', ...variant.buildArgs, '--skip-em-check', '--force', '--no-minify'].join(' '),
]);
copyFileSync(resolve(sourceDir, 'src', `${variant.inputBase}.js`), resolve(outDir, `${variant.outputBase}.js`));
copyFileSync(resolve(sourceDir, 'src', `${variant.inputBase}.wasm`), resolve(outDir, `${variant.outputBase}.wasm`));
console.log(`Wrote ${outDir}/${variant.outputBase}.{js,wasm}`);
