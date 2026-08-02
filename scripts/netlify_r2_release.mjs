#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { checkDeployCachePolicy } from './check_deploy_cache_policy.mjs';
import { checkOrtRuntimeAssets } from './check_ort_runtime_assets.mjs';
import { EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES, isExternalArtifactName } from './engine_artifact_registry.mjs';
import { parseScriptArgs } from './lib/cli.mjs';
import { isSameOriginThreadedStockfishScript } from './prepare_netlify_r2_public_assets.mjs';

const DEFAULT_ASSET_BASE_URL = 'https://assets.0x88.app';
const DEFAULT_CHANNEL_URL = `${DEFAULT_ASSET_BASE_URL}/channels/stable.json`;
const STAMP_FILE = 'release-build.json';

const USAGE = `Usage: node scripts/netlify_r2_release.mjs [options]\n\nOptions:\n  --dist DIR          Built dist directory (default dist-client)\n  --channel-url URL   Artifact channel URL baked into the app shell\n  --asset-base URL    R2/Worker origin for engine/model asset URLs (default https://assets.0x88.app)\n  --build-if-needed   Run the R2/pruned build when the dist stamp is missing/stale\n  --check             Verify the current dist is stamped and pruned; do not build/deploy\n  --deploy            Deploy the verified dist with netlify deploy --no-build\n  --prod              Pass --prod to netlify deploy\n  --message TEXT      Netlify deploy message\n  --npm-bin BIN       npm executable (default npm)\n  --netlify-bin BIN   netlify executable (default netlify)\n  --json              Print machine-readable summary\n  -h, --help          Show help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      dist: { type: 'string', default: 'dist-client' },
      'channel-url': { type: 'string', default: process.env.VITE_LC0_ARTIFACT_CHANNEL_URL || DEFAULT_CHANNEL_URL },
      'asset-base': { type: 'string' },
      'build-if-needed': { type: 'boolean', default: false },
      check: { type: 'boolean', default: false },
      deploy: { type: 'boolean', default: false },
      prod: { type: 'boolean', default: false },
      message: { type: 'string' },
      'npm-bin': { type: 'string', default: 'npm' },
      'netlify-bin': { type: 'string', default: 'netlify' },
      json: { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.assetBase = args.assetBase !== undefined ? args.assetBase.replace(/\/+$/, '') : process.env.VITE_LC0_BROWSER_ASSET_BASE_URL || DEFAULT_ASSET_BASE_URL;
  if (args.check && (args.buildIfNeeded || args.deploy)) {
    throw new Error('--check is verification-only and cannot be combined with --build-if-needed or --deploy');
  }
  return args;
}

function run(command, args, options = {}) {
  const child = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (child.status !== 0) throw new Error(`${command} ${args.join(' ')} failed with status ${child.status}`);
}

function formatMs(ms) {
  return `${(ms / 1000).toFixed(2)}s`;
}

async function timed(name, timings, fn) {
  const started = performance.now();
  try {
    return await fn();
  } finally {
    const ms = performance.now() - started;
    timings.push({ name, ms });
    console.error(`[netlify-r2-release] ${name}: ${formatMs(ms)}`);
  }
}

function capture(command, args) {
  const child = spawnSync(command, args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (child.status !== 0) return undefined;
  return child.stdout.trim();
}

function captureBuffer(command, args) {
  const child = spawnSync(command, args, { maxBuffer: 256 * 1024 * 1024 });
  if (child.status !== 0) return Buffer.alloc(0);
  return child.stdout;
}

async function sha256Path(path) {
  if (!existsSync(path)) return undefined;
  return createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
}

function gitBuildState() {
  const commit = capture('git', ['rev-parse', 'HEAD']) ?? 'unknown';
  const trackedDirty = capture('git', ['status', '--porcelain', '--untracked-files=no']) ?? '';
  const unstagedDiff = captureBuffer('git', ['diff', '--binary']);
  const stagedDiff = captureBuffer('git', ['diff', '--cached', '--binary']);
  const trackedDiffSha256 = createHash('sha256').update(unstagedDiff).update(stagedDiff).digest('hex');
  return { commit, trackedDirty, trackedDiffSha256 };
}

async function desiredStamp(args) {
  const git = gitBuildState();
  return {
    schema: 'lc0_browser.netlify_r2_release_build.v1',
    gitCommit: git.commit,
    trackedDirtySha256: createHash('sha256').update(git.trackedDirty).digest('hex'),
    trackedDiffSha256: git.trackedDiffSha256,
    trackedDirty: git.trackedDirty ? git.trackedDirty.split('\n') : [],
    nodeVersion: process.version,
    buildScope: 'product',
    artifactChannelUrl: args.channelUrl,
    viteEnv: {
      VITE_LC0_BROWSER_ASSET_BASE_URL: args.assetBase ?? '',
      VITE_LC0_MODEL_BASE_URL: process.env.VITE_LC0_MODEL_BASE_URL ?? '',
      VITE_BROWSER_CHESS_DEPLOY_PROFILE: process.env.VITE_BROWSER_CHESS_DEPLOY_PROFILE ?? 'v0',
    },
    inputs: {
      packageJsonSha256: await sha256Path('package.json'),
      packageLockSha256: await sha256Path('package-lock.json'),
      svelteConfigSha256: await sha256Path('svelte.config.js'),
      viteConfigSha256: await sha256Path('vite.config.ts'),
      tsconfigSha256: await sha256Path('tsconfig.json'),
      netlifyTomlSha256: await sha256Path('netlify.toml'),
      // public/_headers is generated, so a renderer change can alter the
      // published artifact while netlify.toml is untouched. Without these, an
      // existing stamp still matches and `--deploy --no-build` ships the old
      // dist copy of a file that should have changed.
      publicHeadersSha256: await sha256Path('public/_headers'),
      netlifyHeadersModelSha256: await sha256Path('scripts/netlify_headers.mjs'),
      generateNetlifyHeadersSha256: await sha256Path('scripts/generate_netlify_headers_file.mjs'),
      checkDeployCachePolicySha256: await sha256Path('scripts/check_deploy_cache_policy.mjs'),
      buildNetlifyR2Sha256: await sha256Path('scripts/build_netlify_r2.mjs'),
      prepareNetlifyR2PublicAssetsSha256: await sha256Path('scripts/prepare_netlify_r2_public_assets.mjs'),
      pruneExternalModelAssetsSha256: await sha256Path('scripts/prune_external_model_assets.mjs'),
      precompressEngineArtifactsSha256: await sha256Path('scripts/precompress_engine_artifacts.mjs'),
    },
  };
}

function comparableStamp(stamp) {
  if (!stamp) return undefined;
  const { generatedAt: _generatedAt, ...rest } = stamp;
  return rest;
}

async function readStamp(dist) {
  const path = join(dist, STAMP_FILE);
  if (!existsSync(path)) return undefined;
  return JSON.parse(await readFile(path, 'utf8'));
}

async function writeStamp(dist, stamp) {
  const path = join(dist, STAMP_FILE);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify({ generatedAt: new Date().toISOString(), ...stamp }, null, 2)}\n`);
}

function findForbiddenExternalAssets(root) {
  const forbidden = [];
  function push(path, kind) {
    forbidden.push({ path, kind, ...(kind === 'file' ? { bytes: statSync(path).size } : {}) });
  }
  function walk(dir, predicate) {
    if (!existsSync(dir)) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (predicate(entry.name, path, true)) push(path, 'directory');
        else walk(path, predicate);
      } else if (predicate(entry.name, path, false)) {
        push(path, 'file');
      }
    }
  }
  walk(join(root, 'models', 'lc0'), (name, _path, isDir) => (isDir ? name.endsWith('.lc0web') : name.endsWith('.onnx')));
  walk(join(root, 'models', 'maia3'), (name, _path, isDir) => !isDir && name.endsWith('.onnx'));
  walk(join(root, 'models'), (name, _path, isDir) => !isDir && name === 'bt4_soap_rem_c19000_final.onnx');
  walk(join(root, 'models', 'monty'), (_name, _path, _isDir) => true);
  walk(join(root, 'monty'), (_name, _path, _isDir) => true);
  for (const dir of EXTERNAL_ENGINE_ARTIFACT_DIRECTORIES) {
    walk(join(root, dir), (name, path, isDir) => !isDir && !isSameOriginThreadedStockfishScript(relative(root, path)) && isExternalArtifactName(name));
  }
  return forbidden.map((item) => ({ ...item, path: relative(process.cwd(), item.path) }));
}

function verifyPrunedDist(dist) {
  if (!existsSync(dist)) throw new Error(`Dist directory does not exist: ${dist}`);
  const forbidden = findForbiddenExternalAssets(dist);
  if (forbidden.length) {
    throw new Error(`R2 Netlify dist contains pruned external artifacts: ${forbidden.map((item) => item.path).join(', ')}`);
  }
  const ortRuntimeAssets = checkOrtRuntimeAssets(dist).runtimeFiles;
  return { forbiddenExternalAssets: forbidden, ortRuntimeAssets };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dist = resolve(args.dist);
  const timings = [];
  // Deploy header policy is a property of the source config, so check it before
  // spending a build on a tree that must not ship.
  const cachePolicy = await timed('check deploy cache policy', timings, () => checkDeployCachePolicy());
  const desired = await timed('compute desired build stamp', timings, () => desiredStamp(args));
  const existing = await timed('read existing build stamp', timings, () => readStamp(dist));
  const stampMatches = JSON.stringify(comparableStamp(existing)) === JSON.stringify(comparableStamp(desired));
  let built = false;

  if (!stampMatches) {
    if (!args.buildIfNeeded) {
      throw new Error(`Dist build stamp is ${existing ? 'stale' : 'missing'}; rerun with --build-if-needed to rebuild once`);
    }
    await timed('build R2 Netlify dist', timings, () => {
      run(args.npmBin, ['run', 'build:netlify:r2'], {
        env: {
          ...process.env,
          BUILD_SCOPE: 'product',
          VITE_LC0_ARTIFACT_CHANNEL_URL: args.channelUrl,
          VITE_LC0_BROWSER_ASSET_BASE_URL: args.assetBase,
          NETLIFY_R2_RELEASE_DIST: dist,
        },
      });
    });
    await timed('verify rebuilt pruned dist', timings, () => verifyPrunedDist(dist));
    await timed('write build stamp', timings, () => writeStamp(dist, desired));
    built = true;
  }

  const verification = await timed('verify pruned dist', timings, () => verifyPrunedDist(dist));

  if (args.deploy) {
    const deployArgs = ['deploy', '--no-build', '--dir', dist];
    if (args.prod) deployArgs.push('--prod');
    if (args.message) deployArgs.push('--message', args.message);
    await timed('netlify deploy', timings, () => run(args.netlifyBin, deployArgs));
  }

  const summary = {
    ok: true,
    dist: relative(process.cwd(), dist) || '.',
    stampMatches,
    built,
    deployed: args.deploy,
    artifactChannelUrl: args.channelUrl,
    assetBaseUrl: args.assetBase,
    cachePolicy,
    verification,
    timings,
  };
  if (args.json) console.log(JSON.stringify(summary, null, 2));
  else console.log(`R2 Netlify release ${args.deploy ? 'deployed' : 'verified'}: built=${built} dist=${summary.dist}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
