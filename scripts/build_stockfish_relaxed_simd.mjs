#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { STOCKFISH_RELEASE, stockfishSourceArchive, stockfishSourcePrefix } from './stockfish_release_config.mjs';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const VARIANTS = {
  'lite-single': { lite: true, args: ['--lite', '--single-threaded'] , suffix: 'lite-single' },
  single: { lite: false, args: ['--single-threaded'], suffix: 'single' },
  'lite-threaded': { lite: true, args: ['--lite'], suffix: 'lite' },
  threaded: { lite: false, args: [], suffix: '' },
};

function replaceOnce(path, oldText, newText) {
  const text = readFileSync(path, 'utf8');
  const count = text.split(oldText).length - 1;
  if (count !== 1) throw new Error(`${path}: expected exactly one match, found ${count}`);
  writeFileSync(path, text.replace(oldText, newText));
}

// SF19 already supplies -pthread, a 4 MiB single-thread search stack, and the
// Emscripten 6.x same-script pthread bootstrap. Do NOT apply the SF18 loader,
// stack or ASYNCIFY_IMPORTS patches to this source generation.
export function patchSource(sourceDir, { relaxed = true, opt = 'O3', lto = true } = {}) {
  if (!['O2', 'O3'].includes(opt)) throw new Error('opt must be O2 or O3');
  const wasmMk = resolve(sourceDir, 'src/emscripten/wasm-makefile.mk');
  if (relaxed) replaceOnce(wasmMk, 'EM_CXXFLAGS += -msimd128', 'EM_CXXFLAGS += -msimd128 -mrelaxed-simd');
  const makefile = resolve(sourceDir, 'src/Makefile');
  replaceOnce(makefile, 'CXXFLAGS += -O3 -funroll-loops', `CXXFLAGS += -${opt} -funroll-loops`);
  if (!lto) {
    const text = readFileSync(makefile, 'utf8');
    if (!text.includes('-flto=full')) throw new Error('Upstream LTO gate changed');
    writeFileSync(makefile, text.replaceAll('-flto=full', ''));
  }
  const buildJs = resolve(sourceDir, 'build.js');
  replaceOnce(buildJs, 'var args = ["-j", require("os").cpus().length];',
    'var args = ["-j", process.env.STOCKFISH_MAKE_JOBS || require("os").cpus().length];');
  if (!relaxed) return;
  const simd = resolve(sourceDir, 'src/nnue/simd.h');
  replaceOnce(simd, '#include "../types.h"', '#if defined(__wasm_relaxed_simd__)\n    #include <wasm_simd128.h>\n#endif\n\n#include "../types.h"');
  // SF19 upstream now carries the relaxed dot implementation itself. Match it
  // exactly so an upstream operand/gate change forces a new range/parity audit.
  replaceOnce(simd,
    '    acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(b, a, acc);',
    `    // SF19 FT: sat_u8(acc0)*sat_u8(acc1)/512 <= 127. Both squared and\n    // clipped hidden activations are also in [0,127]. Keep those activations\n    // in the SECOND/i7 operand; signed weights must be the first operand.\n    // The old SSSE3 pair sum cannot saturate: 2*127*128 = 32512 < 32768.\n    acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(b, a, acc);`);
}

function run(cmd, args, options = {}) {
  console.log(`+ ${cmd} ${args.join(' ')}`);
  const result = spawnSync(cmd, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) throw new Error(`${cmd} failed (${result.status ?? result.error})`);
}

function hash(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

export function requiredNet(sourceDir, variant) {
  const macro = variant.lite ? 'EvalFileLiteName' : 'EvalFileDefaultName';
  const header = readFileSync(resolve(sourceDir, 'src/evaluate.h'), 'utf8');
  const name = header.match(new RegExp(`#define ${macro} "([^"]+)"`))?.[1];
  const net = STOCKFISH_RELEASE.nets.find((n) => n.variant === (variant.lite ? 'lite' : 'full'));
  if (name !== net.name) throw new Error(`Upstream ${macro} changed: ${name}`);
  return net;
}

function ensureNet(sourceDir, net) {
  const dest = resolve(sourceDir, 'src', net.name);
  const candidates = [dest, process.env.STOCKFISH_NNUE_PATH,
    process.env.STOCKFISH_NNUE_DIR && resolve(process.env.STOCKFISH_NNUE_DIR, net.name),
    resolve(repoRoot, '.local_engines/stockfish-nets', net.name),
    resolve(repoRoot, 'assets', net.name)];
  for (const path of candidates.filter(Boolean)) {
    if (!existsSync(path)) continue;
    if (hash(path) !== net.sha256) throw new Error(`Invalid network: ${path}`);
    if (path !== dest) copyFileSync(path, dest);
    return;
  }
  for (const url of [`https://tests.stockfishchess.org/api/nn/${net.name}`,
    `https://media.githubusercontent.com/media/official-stockfish/networks/master/${net.name}`]) {
    const tmp = `${dest}.tmp`;
    const result = spawnSync('curl', ['-fL', '--retry', '3', '--max-time', '300', url, '-o', tmp], { stdio: 'inherit' });
    if (result.status === 0 && hash(tmp) === net.sha256) {
      copyFileSync(tmp, dest);
      rmSync(tmp, { force: true });
      return;
    }
    rmSync(tmp, { force: true });
  }
  throw new Error(`Failed to fetch ${net.name}; set STOCKFISH_NNUE_DIR`);
}

function argValue(name) {
  const i = process.argv.indexOf(name);
  if (i < 0) return undefined;
  const value = process.argv[i + 1];
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${name}`);
  return value;
}

function main() {
  if (process.argv.includes('--help')) {
    console.log(`Usage: node scripts/build_stockfish_relaxed_simd.mjs
  --variant ${Object.keys(VARIANTS).join('|')} (default lite-single)
  --simd baseline|relaxed (default relaxed)
  --opt O2|O3 (default O3) --lto on|off (default on)
  --work-dir path --out-dir path
Environment: EMSDK_DOCKER_IMAGE (default ${STOCKFISH_RELEASE.toolchain}),
  DOCKER_PLATFORM (default linux/amd64), STOCKFISH_MAKE_JOBS (default 2),
  STOCKFISH_RELAXED_SOURCE_DIR, STOCKFISH_NNUE_DIR, STOCKFISH_NNUE_PATH.
The source archive contains the nets. SF18-specific bootstrap patches are retired.`);
    return;
  }
  const variantName = argValue('--variant') || process.env.STOCKFISH_RELAXED_VARIANT || 'lite-single';
  const variant = VARIANTS[variantName];
  if (!variant) throw new Error(`Unknown variant ${variantName}`);
  const simd = argValue('--simd') || 'relaxed';
  const opt = argValue('--opt') || 'O3';
  const lto = argValue('--lto') || (process.env.STOCKFISH_RELAXED_DISABLE_LTO === '1' ? 'off' : 'on');
  if (!['baseline', 'relaxed'].includes(simd) || !['O2', 'O3'].includes(opt) || !['on', 'off'].includes(lto)) throw new Error('Invalid build flags');
  const workRoot = resolve(argValue('--work-dir') || resolve(repoRoot, '.local-dev-artifacts/stockfish-relaxed/repro'));
  // Only the private source subdirectory is cleared; other matrix outputs survive.
  const sourceDir = resolve(workRoot, 'source');
  const outDir = resolve(argValue('--out-dir') || resolve(workRoot, 'dist'));
  const sourceInput = process.env.STOCKFISH_RELAXED_SOURCE_DIR && resolve(process.env.STOCKFISH_RELAXED_SOURCE_DIR);
  if (sourceInput && (sourceInput === workRoot || sourceInput.startsWith(`${sourceDir}/`) || sourceInput === sourceDir)) throw new Error('Source input overlaps build directory');
  rmSync(sourceDir, { recursive: true, force: true });
  mkdirSync(sourceDir, { recursive: true });
  mkdirSync(outDir, { recursive: true });
  if (sourceInput) cpSync(sourceInput, sourceDir, { recursive: true, filter: (path) => !path.includes('/.git') });
  else {
    const archive = resolve(repoRoot, stockfishSourceArchive);
    const root = `stockfish-stockfish-js-${STOCKFISH_RELEASE.version}-corresponding-source`;
    run('tar', ['-xzf', archive, '-C', sourceDir, '--strip-components=3', `${root}/${stockfishSourcePrefix.replace(/\/$/, '')}`]);
    for (const net of STOCKFISH_RELEASE.nets) run('tar', ['-xzf', archive, '-C', resolve(sourceDir, 'src'), '--strip-components=2', `${root}/assets/${net.name}`]);
  }
  const version = JSON.parse(readFileSync(resolve(sourceDir, 'package.json'), 'utf8')).version;
  if (version !== STOCKFISH_RELEASE.version) throw new Error(`Expected Stockfish.js ${STOCKFISH_RELEASE.version}, got ${version}`);
  patchSource(sourceDir, { relaxed: simd === 'relaxed', opt, lto: lto === 'on' });
  ensureNet(sourceDir, requiredNet(sourceDir, variant));
  run('docker', ['run', '--rm', '--platform', process.env.DOCKER_PLATFORM || 'linux/amd64',
    '--user', `${process.getuid?.() ?? 0}:${process.getgid?.() ?? 0}`,
    '-e', `STOCKFISH_MAKE_JOBS=${process.env.STOCKFISH_MAKE_JOBS || 2}`,
    ...(process.env.STOCKFISH_BUILD_CPUS ? ['--cpuset-cpus', process.env.STOCKFISH_BUILD_CPUS] : []),
    '-v', `${sourceDir}:/src`, '-w', '/src', process.env.EMSDK_DOCKER_IMAGE || STOCKFISH_RELEASE.toolchain,
    'bash', '-lc', ['nice', '-n', '10', 'node', 'build.js', ...variant.args, '--no-split', '--skip-em-check', '--force', '--no-minify'].join(' ')]);
  const base = `stockfish-${STOCKFISH_RELEASE.major}${variant.suffix ? `-${variant.suffix}` : ''}`;
  const outputBase = `${base}${simd === 'relaxed' ? '-relaxed' : ''}`;
  // Upstream's default full pthread build is unversioned; --all/prepack adds
  // the versioned npm alias. The other standalone variants are versioned.
  const inputBase = variantName === 'threaded' ? 'stockfish' : base;
  for (const ext of ['js', 'wasm']) copyFileSync(resolve(sourceDir, 'src', `${inputBase}.${ext}`), resolve(outDir, `${outputBase}.${ext}`));
  writeFileSync(resolve(outDir, `${outputBase}.build.json`), `${JSON.stringify({ upstream: STOCKFISH_RELEASE, variant: variantName, simd, opt, lto,
    toolchain: process.env.EMSDK_DOCKER_IMAGE || STOCKFISH_RELEASE.toolchain,
    wasmSha256: hash(resolve(outDir, `${outputBase}.wasm`)) }, null, 2)}\n`);
  console.log(`Wrote ${outDir}/${outputBase}.{js,wasm}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
