import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, relative } from 'node:path';
import { spawnSync } from 'node:child_process';
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';

const ROOT = process.cwd();

const CONFIGS = {
  berserk: {
    engine: 'berserk',
    flavor: 'emscripten-single-thread',
    status: 'experimental',
    upstream: {
      repo: 'https://github.com/jhonnold/berserk.git',
      tag: '14',
      commit: '8ae895a6151695be4a50d4fb65b0c131659c513a',
      license: 'GPL-3.0-or-compatible upstream; verify exact release notices before publishing',
    },
    build: {
      script: 'scripts/build_berserk_emscripten.mjs',
      command: 'npm run berserk:build-emscripten',
      patches: ['patches/berserk-emscripten.patch'],
    },
    artifacts: [
      'public/berserk/berserk-emscripten.js',
      'public/berserk/berserk-emscripten.wasm',
      'public/berserk/berserk-emscripten.data',
    ],
    assets: [
      {
        name: 'berserk-9b84c340af7e.nn',
        sourceUrl: 'https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn',
        rawSha256: '9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084',
        licenseNote: 'No standalone license file found in jhonnold/berserk-networks during intake; do not publicly distribute this network until provenance/license is resolved or confirmed as covered by the engine release.',
        embeddedIn: 'public/berserk/berserk-emscripten.data',
      },
    ],
  },
  plentychess: {
    engine: 'plentychess',
    flavor: 'emscripten-single-thread',
    status: 'experimental',
    upstream: {
      repo: 'https://github.com/Yoshie2000/PlentyChess.git',
      commit: '58d8ba2505ae2b49f48dd410d214a457d15c12c6',
      version: '7.0.66',
      license: 'GPL-3.0',
    },
    build: {
      script: 'scripts/build_plentychess_emscripten.mjs',
      command: 'npm run plentychess:build-emscripten',
      patches: ['patches/plentychess-emscripten.patch'],
    },
    artifacts: [
      'public/plentychess/plentychess-emscripten.js',
      'public/plentychess/plentychess-emscripten.wasm',
      'public/plentychess/plentychess-emscripten.data',
    ],
    assets: [
      {
        name: '0134-2r24-s0.bin',
        sourceUrl: 'https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin',
        rawSha256: '550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061',
        processedPath: '/processed.bin',
        processedSha256: '691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4',
        processingCommand: 'tools/process_net false',
        licenseNote: 'Yoshie2000/PlentyNetworks is licensed under GPL-3.0; preserve GPL-3.0 notices with the network asset.',
        embeddedIn: 'public/plentychess/plentychess-emscripten.data',
      },
    ],
  },
  reckless: {
    engine: 'reckless',
    flavor: 'wasip1-browser',
    status: 'experimental',
    upstream: {
      repo: 'https://github.com/codedeliveryservice/Reckless.git',
      commit: '0010617448bdef4c8cd7d4f4825b7e42c8bc262a',
      license: 'AGPL-3.0',
    },
    build: {
      script: 'scripts/build_reckless_release_assets.mjs',
      command: 'npm run reckless:build-production && npm run reckless:build-browser-api && npm run reckless:build-browser-api-simd && npm run reckless:build-browser-api-simd-external',
      patches: [],
      toolchain: 'Rust cargo with wasm32-wasip1 target; scalar, SIMD, relaxed-SIMD, and browser-API builds from the pinned Reckless source.',
    },
    artifacts: [
      'public/reckless/reckless.wasm',
      'public/reckless/reckless-simd128.wasm',
      'public/reckless/reckless-relaxed-simd128.wasm',
      'public/reckless/reckless-browser-api.wasm',
      'public/reckless/reckless-browser-api-simd128.wasm',
      'public/reckless/reckless-browser-api-simd128-external.wasm',
      'public/reckless/reckless-v60-7f587dfb.nnue',
      'public/reckless/reckless-scalar-corresponding-source.tar.gz',
      'public/reckless/reckless-simd128-corresponding-source.tar.gz',
      'public/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz',
    ],
    assets: [
      {
        name: 'v60-7f587dfb.nnue',
        sourceUrl: 'https://github.com/codedeliveryservice/Reckless.git',
        rawSha256: '7f587dfb1fe5d74d53909328afa6fd51650c8c7f45907602db7fbb1e52948c61',
        licenseNote: 'Reckless embeds and distributes this NNUE with the AGPL-3.0 engine source; preserve corresponding source archives with public WASM distribution.',
        embeddedIn: 'public/reckless/reckless.wasm, public/reckless/reckless-simd128.wasm, public/reckless/reckless-relaxed-simd128.wasm, public/reckless/reckless-browser-api.wasm, public/reckless/reckless-browser-api-simd128.wasm, and public/reckless/reckless-v60-7f587dfb.nnue for the external-NNUE browser API build',
      },
    ],
  },
  viridithas: {
    engine: 'viridithas',
    flavor: 'wasip1-scalar-simd128',
    status: 'experimental',
    upstream: {
      repo: 'https://github.com/cosmobobak/viridithas.git',
      commit: '20d7402065cae084715183e019fdd18089e2dfac',
      license: 'MIT',
    },
    build: {
      script: 'scripts/build_viridithas_wasi.mjs',
      command: 'npm run viridithas:build-wasi && npm run viridithas:build-simd-wasi',
      patches: ['patches/viridithas-wasip1.patch'],
      toolchain: 'Rust cargo with wasm32-wasip1 target; scalar build uses +bulk-memory, SIMD build also uses +simd128.',
    },
    artifacts: [
      'public/viridithas/viridithas.wasm',
      'public/viridithas/viridithas-simd128.wasm',
    ],
    assets: [
      {
        name: 'atlantis-b800.nnue.zst',
        sourceUrl: 'https://github.com/cosmobobak/viridithas-networks/releases/download/v106/atlantis-b800.nnue.zst',
        rawSha256: '2d387407b926df4dbda441cdc3e2288fee2e6a2afa8e1bd22262309ec0fb668a',
        licenseNote: 'Viridithas network required by the WASI build; preserve upstream notices with the MIT engine source and network provider provenance.',
        embeddedIn: 'public/viridithas/viridithas.wasm and public/viridithas/viridithas-simd128.wasm',
      },
    ],
  },
  stockfish: {
    engine: 'stockfish',
    flavor: 'stockfish-js-18.0.7',
    status: 'release',
    upstream: {
      repo: 'https://github.com/nmrugg/stockfish.js.git',
      commit: '32d4b5ae40c01db88219bfbe2b82dbe6dec93832',
      version: '18.0.7',
      license: 'GPL-3.0',
    },
    build: {
      script: 'upstream build.js',
      command: 'cd upstream/stockfish-js-32d4b5ae40c01db88219bfbe2b82dbe6dec93832 && npm install && node build.js --all -f',
      patches: [],
      toolchain: 'Emscripten 3.1.7 as required by Stockfish.js 18 upstream README; Node/npm to install the upstream build dependencies.',
    },
    artifacts: [
      'public/stockfish/stockfish-18-lite-single.js',
      'public/stockfish/stockfish-18-lite-single.wasm',
      'public/stockfish/stockfish-18-lite.js',
      'public/stockfish/stockfish-18-lite.wasm',
      'public/stockfish/stockfish-18-single.js',
      'public/stockfish/stockfish-18-single.wasm',
      'public/stockfish/stockfish-18.js',
      'public/stockfish/stockfish-18.wasm',
    ],
    assets: [],
  },
};

function usage() {
  console.error('Usage: node scripts/write_engine_artifact_manifest.mjs <berserk|plentychess|reckless|viridithas|stockfish> [--out path] [--allow-missing]');
}

function argValue(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function toolchainSummary(config) {
  const explicit = argValue('--toolchain') ?? process.env.ENGINE_ARTIFACT_TOOLCHAIN;
  if (explicit) return explicit;
  if (config.build.toolchain) return config.build.toolchain;
  const emcc = spawnSync('emcc', ['--version'], { encoding: 'utf8' });
  if (emcc.status === 0) return emcc.stdout.split('\n')[0]?.trim() || 'emcc available';
  return 'emcc not found on PATH while writing manifest; pass --toolchain or ENGINE_ARTIFACT_TOOLCHAIN for release manifests';
}

function compressionSummary(buf) {
  const gzip = gzipSync(buf, { level: 9 });
  const brotli = brotliCompressSync(buf, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 } });
  const ratio = (bytes) => Number((bytes / buf.byteLength).toFixed(4));
  return {
    note: 'Estimated precompressed transfer sizes using Node zlib gzip level 9 and brotli quality 11; actual CDN/server settings may differ.',
    gzip: { bytes: gzip.byteLength, ratio: ratio(gzip.byteLength) },
    brotli: { bytes: brotli.byteLength, ratio: ratio(brotli.byteLength) },
  };
}

function sumArtifactBytes(artifacts, field) {
  return artifacts.reduce((sum, artifact) => {
    if (artifact.missing) return sum;
    if (field === 'bytes') return sum + artifact.bytes;
    return sum + artifact.compression[field].bytes;
  }, 0);
}

async function fileMetadata(path, allowMissing, label = 'artifact') {
  if (!existsSync(path)) {
    if (allowMissing) return { path, missing: true };
    throw new Error(`Missing ${label}: ${path}`);
  }
  const buf = await readFile(path);
  return {
    path,
    bytes: buf.byteLength,
    sha256: createHash('sha256').update(buf).digest('hex'),
  };
}

async function fileEntry(path, allowMissing) {
  const entry = await fileMetadata(path, allowMissing, 'artifact');
  if (entry.missing) return entry;
  const buf = await readFile(path);
  return {
    ...entry,
    compression: compressionSummary(buf),
  };
}

const engine = process.argv[2];
const config = CONFIGS[engine];
if (!config) {
  usage();
  process.exitCode = 1;
} else {
  const allowMissing = process.argv.includes('--allow-missing');
  const out = argValue('--out') ?? `artifacts/engine-manifests/${engine}-${config.flavor}.manifest.json`;
  const sourceArchivePath = argValue('--source-archive');
  const sourceArchiveUrl = argValue('--source-url');
  const artifacts = await Promise.all(config.artifacts.map((p) => fileEntry(p, allowMissing)));
  const totalBytes = sumArtifactBytes(artifacts, 'bytes');
  const totalGzipBytes = sumArtifactBytes(artifacts, 'gzip');
  const totalBrotliBytes = sumArtifactBytes(artifacts, 'brotli');
  const manifest = {
    schema: 'lc0-webgpu.browser-engine-artifact-manifest.v1',
    generatedAt: new Date().toISOString(),
    distributionPolicy: 'docs/engine_artifact_distribution.md',
    ...config,
    build: { ...config.build, toolchain: toolchainSummary(config) },
    artifacts,
    totals: {
      bytes: totalBytes,
      gzipBytes: totalGzipBytes,
      brotliBytes: totalBrotliBytes,
      gzipRatio: totalBytes ? Number((totalGzipBytes / totalBytes).toFixed(4)) : null,
      brotliRatio: totalBytes ? Number((totalBrotliBytes / totalBytes).toFixed(4)) : null,
    },
    sourceArchive: sourceArchivePath
      ? {
          required: true,
          ...(await fileMetadata(sourceArchivePath, false, 'source archive')),
          url: sourceArchiveUrl ?? null,
          note: sourceArchiveUrl ? 'Source archive recorded for public distribution.' : 'Source archive hash recorded; add --source-url before public distribution.',
        }
      : {
          required: true,
          url: null,
          sha256: null,
          note: 'Fill before public distribution; source archive must match this manifest.',
        },
  };
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Wrote ${relative(ROOT, out)}`);
}
