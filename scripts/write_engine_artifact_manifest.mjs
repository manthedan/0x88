import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, relative } from 'node:path';
import { spawnSync } from 'node:child_process';

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
        embeddedIn: 'public/plentychess/plentychess-emscripten.data',
      },
    ],
  },
};

function usage() {
  console.error('Usage: node scripts/write_engine_artifact_manifest.mjs <berserk|plentychess> [--out path] [--allow-missing]');
}

function argValue(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function toolchainSummary() {
  const emcc = spawnSync('emcc', ['--version'], { encoding: 'utf8' });
  if (emcc.status === 0) return emcc.stdout.split('\n')[0]?.trim() || 'emcc available';
  return 'emcc not found on PATH while writing manifest';
}

async function fileEntry(path, allowMissing) {
  if (!existsSync(path)) {
    if (allowMissing) return { path, missing: true };
    throw new Error(`Missing artifact: ${path}`);
  }
  const buf = await readFile(path);
  return {
    path,
    bytes: buf.byteLength,
    sha256: createHash('sha256').update(buf).digest('hex'),
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
  const artifacts = await Promise.all(config.artifacts.map((p) => fileEntry(p, allowMissing)));
  const manifest = {
    schema: 'lc0-webgpu.browser-engine-artifact-manifest.v1',
    generatedAt: new Date().toISOString(),
    distributionPolicy: 'docs/engine_artifact_distribution.md',
    ...config,
    build: { ...config.build, toolchain: toolchainSummary() },
    artifacts,
    sourceArchive: {
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
