#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, copyFileSync } from 'node:fs';
import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';

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
    smokeHtml: 'berserk-smoke.html',
    adapterFiles: [
      'src/lc0/berserkEngine.ts',
      'src/lc0/berserkSmoke.ts',
      'src/lc0/berserkVariants.ts',
      'src/lc0/browserUciEngine.ts',
      'src/lc0/stockfishEngine.ts',
      'src/chess/board.ts',
    ],
    docs: ['docs/engine_artifact_distribution.md', 'docs/browser_c_engine_porting.md', 'docs/engine_catalog.md', 'docs/berserk_browser_benchmarks.md', 'docs/netlify_engine_artifacts.md', 'public/berserk/README.md'],
    assetDir: '.local_engines/berserk-nets',
    assets: [
      {
        name: 'berserk-9b84c340af7e.nn',
        sourceUrl: 'https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn',
        rawSha256: '9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084',
        licenseNote: 'No standalone license file found in jhonnold/berserk-networks during intake; do not publicly distribute this network until provenance/license is resolved or confirmed as covered by the engine release.',
      },
    ],
    envPrefix: 'BERSERK',
  },
  plentychess: {
    engine: 'plentychess',
    flavor: 'emscripten-single-thread',
    upstreamRepo: 'https://github.com/Yoshie2000/PlentyChess.git',
    upstreamCommit: '58d8ba2505ae2b49f48dd410d214a457d15c12c6',
    upstreamVersion: '7.0.66',
    buildDir: '.local_engines/plentychess-emscripten-src',
    sourcePrefix: 'upstream/plentychess-58d8ba2505ae2b49f48dd410d214a457d15c12c6/',
    output: 'public/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz',
    patch: 'patches/plentychess-emscripten.patch',
    buildScript: 'scripts/build_plentychess_emscripten.mjs',
    smokeScript: 'scripts/plentychess_emscripten_smoke.mjs',
    smokeHtml: 'plentychess-smoke.html',
    adapterFiles: [
      'src/lc0/plentychessEngine.ts',
      'src/lc0/plentychessSmoke.ts',
      'src/lc0/plentychessVariants.ts',
      'src/lc0/browserUciEngine.ts',
      'src/lc0/stockfishEngine.ts',
      'src/chess/board.ts',
    ],
    docs: ['docs/engine_artifact_distribution.md', 'docs/browser_c_engine_porting.md', 'docs/engine_catalog.md', 'docs/plentychess_browser_port.md', 'docs/netlify_engine_artifacts.md', 'public/plentychess/README.md'],
    assetDir: '.local_engines/plentychess-nets',
    assets: [
      {
        name: '0134-2r24-s0.bin',
        sourceUrl: 'https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin',
        rawSha256: '550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061',
        processedSha256: '691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4',
        processingCommand: 'tools/process_net false',
        licenseNote: 'Yoshie2000/PlentyNetworks is licensed under GPL-3.0; preserve GPL-3.0 notices with the network asset.',
      },
    ],
    envPrefix: 'PLENTYCHESS',
  },
};

function usage() {
  console.error('Usage: node scripts/write_engine_source_archive.mjs <berserk|plentychess> [--out path] [--allow-missing-assets]');
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
  if (existsSync(join(dir, '.git'))) return dir;
  throw new Error(`Missing local git checkout for ${config.engine}: ${dir}. Run ${config.buildScript} once or set up the source checkout before writing a source archive.`);
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
  const outDir = config.engine === 'berserk' ? '$PWD/out/public/berserk/berserk-emscripten.js' : '$PWD/out/public/plentychess/plentychess-emscripten.js';
  return `${config.envPrefix}_SKIP_GIT=1 ${config.envPrefix}_BUILD_DIR="${sourceDir}" ${config.envPrefix}_NET_DIR="$PWD/assets" ${config.envPrefix}_EMSCRIPTEN_JS_OUT="${outDir}" npm run ${config.engine}:build-emscripten`;
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
    patch: config.patch,
    buildScript: config.buildScript,
    smokeScript: config.smokeScript,
    assets: config.assets.map((asset) => asset.name),
  };
  await writeFile(join(archiveRoot, 'SOURCE_ARCHIVE.json'), `${JSON.stringify(meta, null, 2)}\n`);
  await writeFile(join(archiveRoot, 'BUILDING.md'), `# ${config.engine} ${config.flavor} corresponding source\n\nThis archive contains the upstream source snapshot, local browser-port patch, build scripts, browser adapter source, docs, and required network/model asset provenance for the generated ${config.engine} Emscripten artifacts.\n\n## Rebuild\n\nInstall Node dependencies and provide Emscripten either on PATH or through Docker, then run:\n\n\`\`\`sh\nnpm ci\n${rebuildCommand(config)}\n\`\`\`\n\nThe build script applies ${config.patch} to the upstream snapshot in ${config.sourcePrefix}. It writes JS/WASM/data outputs under \`out/public/${config.engine}/\`.\n\nTo smoke-test the rebuilt artifact, point the smoke script at the rebuilt JS output, for example:\n\n\`\`\`sh\nnode ${config.smokeScript} --js out/public/${config.engine}/${config.engine}-emscripten.js --depth 1\n\`\`\`\n\nSee \`ASSET_PROVENANCE.json\` for network/model source URLs and hashes.\n`);
}

async function writeArchive(config, outPath, allowMissingAssets) {
  const baseName = `${config.engine}-${config.flavor}-corresponding-source`;
  const tmp = resolve(tmpdir(), `${baseName}-${process.pid}-${Date.now()}`);
  const archiveRoot = join(tmp, baseName);
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(archiveRoot, { recursive: true });
  try {
    await exportUpstream(config, archiveRoot);
    await copyFiles([
      'package.json',
      'package-lock.json',
      'tsconfig.json',
      'vite.config.ts',
      config.patch,
      config.buildScript,
      config.smokeScript,
      'scripts/write_engine_artifact_manifest.mjs',
      'scripts/precompress_engine_artifacts.mjs',
      config.smokeHtml,
      ...config.adapterFiles,
      ...config.docs,
    ], archiveRoot);
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
