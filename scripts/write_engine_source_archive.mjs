#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, relative, resolve } from 'node:path';

import { STOCKFISH_RELEASE, stockfishSourceArchive, stockfishSourcePrefix } from './stockfish_release_config.mjs';

const ROOT = process.cwd();

const CONFIGS = {
  berserk: {
    engine: 'berserk',
    flavor: 'emscripten-single-thread',
    upstreamRepo: 'https://github.com/jhonnold/berserk.git',
    upstreamCommit: '8ae895a6151695be4a50d4fb65b0c131659c513a',
    upstreamTag: '14',
    buildDir: '.local_engines/berserk-emscripten-src',
    sourcePrefix: 'upstream/berserk-8ae895a6151695be4a50d4fb65b0c131659c513a/',
    output: 'public/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz',
    patch: 'patches/berserk-emscripten.patch',
    buildScript: 'scripts/build_berserk_emscripten.mjs',
    smokeScript: 'scripts/berserk_emscripten_smoke.mjs',
    smokeHtml: 'lab/berserk-smoke.html',
    adapterFiles: [
      'src/lc0/berserkEngine.ts',
      'src/lc0/berserkSmoke.ts',
      'src/lc0/berserkVariants.ts',
      'src/lc0/emscriptenLocateFile.ts',
      'src/lc0/browserUciEngine.ts',
      'src/lc0/stockfishEngine.ts',
      'src/chess/board.ts',
    ],
    docs: [
      'docs/engine_artifact_distribution.md',
      'docs/browser_c_engine_porting.md',
      'docs/engine_catalog.md',
      'docs/berserk_browser_benchmarks.md',
      'docs/netlify_engine_artifacts.md',
      'public/berserk/README.md',
    ],
    assetDir: '.local_engines/berserk-nets',
    assets: [
      {
        name: 'berserk-9b84c340af7e.nn',
        sourceUrl: 'https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn',
        rawSha256: '9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084',
        licenseNote:
          'Covered by Berserk GPL-3.0 under this release policy: the net is named as EVALFILE in the engine Makefile, downloaded and hash-verified by the engine build, and required for the binary to run, so it is treated as part of the Corresponding Source. See docs/engine_artifact_distribution.md.',
      },
    ],
    envPrefix: 'BERSERK',
  },
  plentychess: {
    engine: 'plentychess',
    flavor: 'emscripten-single-thread',
    upstreamRepo: 'https://github.com/Yoshie2000/PlentyChess.git',
    upstreamCommit: '50ad9f1b523063cbbbf599fe0547f42870fe2906',
    upstreamTag: 'b-v8.0.0',
    upstreamVersion: '8.0.0',
    buildDir: '.local_engines/plentychess-emscripten-src',
    sourcePrefix: 'upstream/plentychess-50ad9f1b523063cbbbf599fe0547f42870fe2906/',
    output: 'public/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz',
    patch: 'patches/plentychess-emscripten.patch',
    buildScript: 'scripts/build_plentychess_emscripten.mjs',
    smokeScript: 'scripts/plentychess_emscripten_smoke.mjs',
    smokeHtml: 'lab/plentychess-smoke.html',
    adapterFiles: [
      'src/lc0/plentychessEngine.ts',
      'src/lc0/plentychessSmoke.ts',
      'src/lc0/plentychessVariants.ts',
      'src/lc0/emscriptenLocateFile.ts',
      'src/lc0/browserUciEngine.ts',
      'src/lc0/stockfishEngine.ts',
      'src/chess/board.ts',
    ],
    docs: [
      'docs/engine_artifact_distribution.md',
      'docs/browser_c_engine_porting.md',
      'docs/engine_catalog.md',
      'docs/plentychess_browser_port.md',
      'docs/plentychess_upgrade_v8.md',
      'docs/plentychess_upgrade_v8_evidence.json',
      'docs/plentychess_simd_audit.md',
      'docs/netlify_engine_artifacts.md',
      'public/plentychess/README.md',
    ],
    assetDir: '.local_engines/plentychess-nets',
    assets: [
      {
        name: '0178r.bin',
        sourceUrl: 'https://github.com/Yoshie2000/PlentyNetworks/releases/download/0178r/0178r.bin',
        rawSha256: '663e0531ac71c6c7c6d3bc4f1fac923bf69136f9786fa1da5c4ac25382aee54b',
        processedSha256: '5cedef19778f1bb784e181fc013543127d1fdcde227795710b1a48f6e53d79cd',
        processingCommand: 'tools/process_net false',
        licenseNote: 'Yoshie2000/PlentyNetworks is licensed under GPL-3.0; preserve GPL-3.0 notices with the network asset.',
      },
    ],
    envPrefix: 'PLENTYCHESS',
  },
  stormphrax: {
    engine: 'stormphrax',
    flavor: 'emscripten-single-thread',
    upstreamRepo: 'https://github.com/Ciekce/Stormphrax.git',
    upstreamCommit: '582965517ed2032d41a6b4cd6c2e66b1b934e2ad',
    upstreamTag: 'v8.0.0',
    upstreamVersion: '8.0.0',
    buildDir: '.local_engines/stormphrax-emscripten-src',
    sourcePrefix: 'upstream/stormphrax-582965517ed2032d41a6b4cd6c2e66b1b934e2ad/',
    output: 'public/stormphrax/stormphrax-emscripten-single-thread-corresponding-source.tar.gz',
    patch: 'patches/stormphrax-emscripten.patch',
    buildScript: 'scripts/build_stormphrax_emscripten.mjs',
    smokeScript: 'scripts/stormphrax_emscripten_smoke.mjs',
    adapterFiles: [
      'src/lc0/stormphraxEngine.ts',
      'src/lc0/stormphraxVariants.ts',
      'src/lc0/emscriptenLocateFile.ts',
      'src/lc0/browserUciEngine.ts',
      'src/lc0/stockfishEngine.ts',
      'src/chess/board.ts',
    ],
    docs: [
      'docs/engine_artifact_distribution.md',
      'docs/browser_c_engine_porting.md',
      'docs/engine_catalog.md',
      'docs/stormphrax_browser_port.md',
      'docs/netlify_engine_artifacts.md',
      'public/stormphrax/README.md',
    ],
    assetDir: '.local_engines/stormphrax-nets',
    assets: [
      {
        name: 'undertown.nnue',
        sourceUrl: 'https://github.com/Ciekce/stormphrax-nets/releases/download/undertown/undertown.nnue',
        rawSha256: '04d651e078b7c7334709dbd772d40a23c0a5480e93e19521a03020c7d633f2cf',
        licenseNote:
          'Stormphrax documents undertown as the self-trained release network for 8.0.0; preserve GPL source and provenance alongside public browser artifacts.',
      },
    ],
    envPrefix: 'STORMPHRAX',
  },
  viridithas: {
    engine: 'viridithas',
    flavor: 'wasip1-scalar-simd128',
    upstreamRepo: 'https://github.com/cosmobobak/viridithas.git',
    upstreamCommit: '0631113e2215714c54d397cce9a8caebc5e6f8ee',
    buildDir: '.local_engines/viridithas-wasi-src',
    sourcePrefix: 'upstream/viridithas-0631113e2215714c54d397cce9a8caebc5e6f8ee/',
    output: 'public/viridithas/viridithas-wasip1-corresponding-source.tar.gz',
    patch: 'patches/viridithas-wasip1.patch',
    buildScript: 'scripts/build_viridithas_wasi.mjs',
    smokeScript: 'scripts/viridithas_wasi_smoke.mjs',
    adapterFiles: ['src/lc0/viridithasEngine.ts', 'src/lc0/viridithasVariants.ts', 'src/lc0/stockfishEngine.ts', 'src/chess/board.ts'],
    docs: [
      'docs/engine_artifact_distribution.md',
      'docs/browser_c_engine_porting.md',
      'docs/engine_catalog.md',
      'docs/netlify_engine_artifacts.md',
      'public/viridithas/README.md',
    ],
    assetDir: '.local_engines/viridithas-nets',
    assets: [
      {
        name: 'sandhi-s2-b200.nnue.zst',
        sourceUrl: 'https://github.com/cosmobobak/viridithas-networks/releases/download/v109/sandhi-s2-b200.nnue.zst',
        rawSha256: '066c172e22c25ee3d3b4084ad44ff761426f906a9c536cabd0127d17413d0585',
        licenseNote: 'sandhi-s2-b200.nnue.zst is CC0-1.0 (cosmobobak/viridithas-networks LICENSE at 1040c3ee, 2026-06-08, before the v109 release). The engine source is MIT.',
      },
    ],
    envPrefix: 'VIRIDITHAS',
  },
  stockfish: {
    engine: 'stockfish',
    flavor: `stockfish-js-${STOCKFISH_RELEASE.version}`,
    upstreamRepo: STOCKFISH_RELEASE.repo,
    upstreamCommit: STOCKFISH_RELEASE.commit,
    upstreamTag: STOCKFISH_RELEASE.tag,
    upstreamVersion: STOCKFISH_RELEASE.version,
    buildDir: '.local_engines/stockfish-js-src',
    sourcePrefix: stockfishSourcePrefix,
    output: stockfishSourceArchive,
    buildScript: 'scripts/build_stockfish_relaxed_simd.mjs',
    smokeScript: 'scripts/uci_stockfish_js_wrapper.mjs',
    adapterFiles: ['src/lc0/stockfishEngine.ts', 'src/lc0/stockfishPthreadBootstrap.js',
      'src/lc0/wasmFeatures.ts', 'src/lc0/assetUrls.ts', 'src/lc0/browserUciEngine.ts',
      'src/lc0/engineCatalog.ts', 'src/chess/board.ts'],
    docs: [
      'docs/engine_artifact_distribution.md',
      'docs/engine_catalog.md',
      'docs/netlify_engine_artifacts.md',
      'docs/stockfish_relaxed_simd_experiment.md',
      'docs/stockfish_19_upgrade.md',
      'scripts/stockfish_upgrade_bench.mjs',
      'scripts/emscripten_uci_protocol.mjs',
      'patches/stockfish-js-native-audit.patch',
      'scripts/stockfish_upgrade_browser.mjs',
      'scripts/stockfish_upgrade_browser_matrix.mjs',
      'public/stockfish/README.md',
      'node_modules/stockfish/README.md',
      'node_modules/stockfish/Copying.txt',
      'node_modules/stockfish/package.json',
      'node_modules/stockfish/index.js',
      'node_modules/stockfish/scripts/findEngine.js',
      'node_modules/stockfish/scripts/cli.js',
    ],
    assetDir: '.local_engines/stockfish-nets',
    assets: STOCKFISH_RELEASE.nets.map((net) => ({
      name: net.name,
      rawSha256: net.sha256,
      sourceUrl: `https://tests.stockfishchess.org/api/nn/${net.name}`,
      licenseNote: 'Required embedded Stockfish GPL network; included in Corresponding Source.',
    })),
    envPrefix: 'STOCKFISH_JS',
  },
};

function usage() {
  console.error(
    'Usage: node scripts/write_engine_source_archive.mjs <berserk|plentychess|stormphrax|viridithas|stockfish> [--out path] [--allow-missing-assets]',
  );
}

function argValue(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: options.capture ? 'pipe' : 'inherit', encoding: 'utf8', ...options });
  if (result.status !== 0) {
    const detail = result.stderr || result.stdout || '';
    throw new Error(`${command} ${args.join(' ')} failed${detail ? `\n${detail}` : ''}`);
  }
  return result.stdout ?? '';
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

async function copyIfExists(path, archiveRoot) {
  const src = resolve(path);
  if (!existsSync(src)) throw new Error(`Missing required source file: ${path}`);
  const dest = join(archiveRoot, path);
  await mkdir(dirname(dest), { recursive: true });
  await cp(src, dest, { recursive: true, dereference: false });
}

async function copyFiles(paths, archiveRoot) {
  for (const path of paths) await copyIfExists(path, archiveRoot);
}

function ensureGitSource(config) {
  const dir = resolve(config.buildDir);
  if (!existsSync(join(dir, '.git'))) {
    mkdirSync(dirname(dir), { recursive: true });
    run('git', ['clone', config.upstreamRepo, dir]);
  }
  run('git', ['fetch', '--tags', 'origin'], { cwd: dir });
  run('git', ['checkout', config.upstreamCommit], { cwd: dir });
  run('git', ['reset', '--hard'], { cwd: dir });
  return dir;
}

async function exportUpstream(config, archiveRoot) {
  const sourceDir = ensureGitSource(config);
  const tarPath = join(archiveRoot, 'upstream-source.tar');
  run('git', ['archive', '--format=tar', `--prefix=${config.sourcePrefix}`, '--output', tarPath, config.upstreamCommit], { cwd: sourceDir });
  run('tar', ['-xf', tarPath, '-C', archiveRoot]);
  rmSync(tarPath, { force: true });
}

async function copyAssets(config, archiveRoot, allowMissingAssets) {
  const copied = [];
  const assetOut = join(archiveRoot, 'assets');
  await mkdir(assetOut, { recursive: true });
  for (const asset of config.assets) {
    const source = resolve(config.assetDir, asset.name);
    if (!existsSync(source) && asset.sourceUrl && !allowMissingAssets) {
      mkdirSync(dirname(source), { recursive: true });
      run('curl', ['-L', '--fail', '-o', source, asset.sourceUrl]);
    }
    if (!existsSync(source)) {
      if (allowMissingAssets) {
        copied.push({ ...asset, missing: true });
        continue;
      }
      throw new Error(`Missing required network/model asset: ${source}`);
    }
    const dest = join(assetOut, asset.name);
    copyFileSync(source, dest);
    const actual = sha256(dest);
    if (asset.rawSha256 && asset.rawSha256 !== actual) throw new Error(`${asset.name} checksum mismatch: expected ${asset.rawSha256}, got ${actual}`);
    copied.push({ ...asset, path: `assets/${asset.name}`, bytes: statSync(dest).size, sha256: actual });
  }
  await writeFile(join(archiveRoot, 'ASSET_PROVENANCE.json'), `${JSON.stringify(copied, null, 2)}\n`);
}

function rebuildCommand(config) {
  const sourceDir = `$PWD/${config.sourcePrefix.replace(/\/$/, '')}`;
  if (config.engine === 'stormphrax') {
    return `rm -rf "$PWD/.build/stormphrax-baseline" "$PWD/.build/stormphrax-relaxed"\nmkdir -p "$PWD/.build" "$PWD/out/public/stormphrax"\ncp -a "${sourceDir}" "$PWD/.build/stormphrax-baseline"\ncp -a "${sourceDir}" "$PWD/.build/stormphrax-relaxed"\n${config.envPrefix}_SKIP_GIT=1 ${config.envPrefix}_BUILD_DIR="$PWD/.build/stormphrax-baseline" ${config.envPrefix}_NET_DIR="$PWD/assets" ${config.envPrefix}_EMSCRIPTEN_JS_OUT="$PWD/out/public/stormphrax/stormphrax-emscripten.js" npm run stormphrax:build-emscripten\n${config.envPrefix}_SKIP_GIT=1 ${config.envPrefix}_BUILD_DIR="$PWD/.build/stormphrax-relaxed" ${config.envPrefix}_NET_DIR="$PWD/assets" ${config.envPrefix}_WASM_RELAXED_SIMD=1 ${config.envPrefix}_EMSCRIPTEN_JS_OUT="$PWD/out/public/stormphrax/stormphrax-emscripten-relaxed-simd128.js" node scripts/build_stormphrax_emscripten.mjs`;
  }
  if (config.engine === 'berserk') {
    return `rm -rf "$PWD/.build/berserk-baseline" "$PWD/.build/berserk-simd" "$PWD/.build/berserk-relaxed"
mkdir -p "$PWD/.build" "$PWD/out/public/berserk"
cp -a "${sourceDir}" "$PWD/.build/berserk-baseline"
cp -a "${sourceDir}" "$PWD/.build/berserk-simd"
cp -a "${sourceDir}" "$PWD/.build/berserk-relaxed"
BERSERK_SKIP_GIT=1 BERSERK_BUILD_DIR="$PWD/.build/berserk-baseline" BERSERK_NET_DIR="$PWD/assets" BERSERK_EMSCRIPTEN_JS_OUT="$PWD/out/public/berserk/berserk-emscripten.js" node scripts/build_berserk_emscripten.mjs
BERSERK_SKIP_GIT=1 BERSERK_BUILD_DIR="$PWD/.build/berserk-simd" BERSERK_NET_DIR="$PWD/assets" BERSERK_WASM_SIMD=1 BERSERK_EMSCRIPTEN_JS_OUT="$PWD/out/public/berserk/berserk-emscripten-simd128.js" node scripts/build_berserk_emscripten.mjs
BERSERK_SKIP_GIT=1 BERSERK_BUILD_DIR="$PWD/.build/berserk-relaxed" BERSERK_NET_DIR="$PWD/assets" BERSERK_WASM_RELAXED_SIMD=1 BERSERK_EMSCRIPTEN_JS_OUT="$PWD/out/public/berserk/berserk-emscripten-relaxed-simd128.js" node scripts/build_berserk_emscripten.mjs`;
  }
  if (config.engine === 'plentychess') {
    // Each build applies the patch, so use a fresh source copy for each tier.
    return `mkdir -p "$PWD/.build" "$PWD/out/public/plentychess"
for tier in base sse41 relaxed; do
  build="$PWD/.build/plentychess-$tier"
  rm -rf "$build"
  cp -a "${sourceDir}" "$build"
  suffix=""; sse41=0; relaxed=0
  case "$tier" in
    sse41) suffix="-sse41"; sse41=1 ;;
    relaxed) suffix="-relaxed-simd128"; relaxed=1 ;;
  esac
  PLENTYCHESS_SKIP_GIT=1 PLENTYCHESS_BUILD_DIR="$build" PLENTYCHESS_NET_DIR="$PWD/assets" PLENTYCHESS_WASM_SSE41="$sse41" PLENTYCHESS_WASM_RELAXED_SIMD="$relaxed" PLENTYCHESS_EMSCRIPTEN_JS_OUT="$PWD/out/public/plentychess/plentychess-emscripten$suffix.js" node scripts/build_plentychess_emscripten.mjs || exit 1
done`;
  }
  if (config.engine === 'viridithas') {
    return `${config.envPrefix}_SKIP_GIT=1 ${config.envPrefix}_BUILD_DIR="${sourceDir}" ${config.envPrefix}_NET_DIR="$PWD/assets" ${config.envPrefix}_WASM_OUT="$PWD/out/public/viridithas/viridithas.wasm" npm run viridithas:build-wasi && ${config.envPrefix}_SKIP_GIT=1 ${config.envPrefix}_BUILD_DIR="${sourceDir}" ${config.envPrefix}_NET_DIR="$PWD/assets" ${config.envPrefix}_WASM_SIMD=1 ${config.envPrefix}_WASM_OUT="$PWD/out/public/viridithas/viridithas-simd128.wasm" npm run viridithas:build-wasi && ${config.envPrefix}_SKIP_GIT=1 ${config.envPrefix}_BUILD_DIR="${sourceDir}" ${config.envPrefix}_NET_DIR="$PWD/assets" ${config.envPrefix}_WASM_RELAXED_SIMD=1 ${config.envPrefix}_WASM_OUT="$PWD/out/public/viridithas/viridithas-relaxed-simd128.wasm" npm run viridithas:build-wasi`;
  }
  if (config.engine === 'stockfish') {
    return `export STOCKFISH_RELAXED_SOURCE_DIR="${sourceDir}"\nexport STOCKFISH_NNUE_DIR="$PWD/assets"\n# Rebuild each WASM flavor with the pinned Docker toolchain (no ASM-JS release).\nfor variant in lite-single single lite-threaded threaded; do\n  node scripts/build_stockfish_relaxed_simd.mjs --variant "$variant" --simd baseline --out-dir "$PWD/out/public/stockfish"\ndone\nnode scripts/build_stockfish_relaxed_simd.mjs --variant lite-single --out-dir "$PWD/out/public/stockfish"\n# Optional unpromoted relaxed full/pthread candidates: use the same --variant values.`;
  }
  return `npm run ${config.engine}:build`;
}

async function writeArchiveDocs(config, archiveRoot) {
  const meta = {
    schema: 'lc0-webgpu.browser-engine-corresponding-source.v1',
    engine: config.engine,
    flavor: config.flavor,
    generatedAt: new Date().toISOString(),
    upstream: {
      repo: config.upstreamRepo,
      commit: config.upstreamCommit,
      tag: config.upstreamTag,
      version: config.upstreamVersion,
    },
    patch: config.patch ?? null,
    buildScript: config.buildScript,
    smokeScript: config.smokeScript,
    assets: config.assets.map((asset) => asset.name),
  };
  await writeFile(join(archiveRoot, 'SOURCE_ARCHIVE.json'), `${JSON.stringify(meta, null, 2)}\n`);
  const patchNote = config.engine === 'stockfish'
    ? 'The relaxed builder enables the upstream relaxed-dot gate, adds its audited operand-range comment and supports the O2/O3 × LTO matrix. SF19 supplies the stack and pthread bootstrap fixes upstream. The native-audit patch only repairs a pointer typo in the non-WASM CLI.'
    : config.patch
    ? `The build script applies ${config.patch} to the upstream snapshot in ${config.sourcePrefix}.`
    : `No local source patch is applied; the deployed artifacts come from the pinned upstream/package source in ${config.sourcePrefix}.`;
  // PlentyChess's builder/smoke use only Node built-ins, and Stockfish builds in
  // pinned Docker; npm ci would invoke the application's Svelte prepare script
  // in this deliberately partial tree.
  const setup = config.engine === 'stockfish'
    ? 'Install Node.js and Docker. The pinned Docker image supplies Emscripten and Node; the --no-minify builder needs no host npm dependencies. Do not run npm ci in this source subset (it is not a complete Svelte application).'
    : config.engine === 'plentychess'
      ? 'Install Node, Emscripten, a native C++ compiler, make, curl and git (no npm dependencies are needed for the engine build), then run:'
      : 'Install Node dependencies and the documented upstream toolchain, then run:';
  const install = config.engine === 'stockfish' || config.engine === 'plentychess' ? '' : 'npm ci\n';
  await writeFile(
    join(archiveRoot, 'BUILDING.md'),
    `# ${config.engine} ${config.flavor} corresponding source\n\nThis archive contains the upstream source snapshot, local browser integration scripts/adapters, docs, and required network/model asset provenance for the generated ${config.engine} browser artifacts.\n\n## Rebuild\n\n${setup}\n\n\`\`\`sh\n${install}${rebuildCommand(config)}\n\`\`\`\n\n${patchNote}\n\nThe build writes outputs under \`out/public/${config.engine}/\` or the upstream build directory as noted by the command above.\n\nSee \`ASSET_PROVENANCE.json\` for network/model source URLs and hashes.\n`,
  );
}

async function writeArchive(config, outPath, allowMissingAssets) {
  const baseName = `${config.engine}-${config.flavor}-corresponding-source`;
  const tmp = resolve(tmpdir(), `${baseName}-${process.pid}-${Date.now()}`);
  const archiveRoot = join(tmp, baseName);
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(archiveRoot, { recursive: true });
  try {
    await exportUpstream(config, archiveRoot);
    await copyFiles(
      [
        'package.json',
        'package-lock.json',
        'tsconfig.json',
        'vite.config.ts',
        config.patch,
        config.buildScript,
        config.smokeScript,
        'scripts/write_engine_artifact_manifest.mjs',
        // The shared manifest helper imports this even for non-Stockfish engines.
        'scripts/stockfish_release_config.mjs',
        'scripts/precompress_engine_artifacts.mjs',
        config.smokeHtml,
        ...config.adapterFiles,
        ...config.docs,
      ].filter(Boolean),
      archiveRoot,
    );
    await copyAssets(config, archiveRoot, allowMissingAssets);
    await writeArchiveDocs(config, archiveRoot);
    await mkdir(dirname(outPath), { recursive: true });
    run('tar', ['-czf', outPath, '-C', tmp, baseName]);
    return { outPath, sha256: sha256(outPath), bytes: statSync(outPath).size };
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

const engine = process.argv[2];
const config = CONFIGS[engine];
if (!config) {
  usage();
  process.exitCode = 1;
} else {
  const out = resolve(argValue('--out') ?? config.output);
  const allowMissingAssets = process.argv.includes('--allow-missing-assets');
  const result = await writeArchive(config, out, allowMissingAssets);
  console.log(`Wrote ${relative(ROOT, result.outPath)} (${result.bytes} bytes)`);
  console.log(`sha256 ${result.sha256}`);
}
