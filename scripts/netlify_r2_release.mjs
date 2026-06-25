#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, relative, resolve } from 'node:path';

const DEFAULT_ASSET_BASE_URL = 'https://assets.0x88.app';
const DEFAULT_CHANNEL_URL = `${DEFAULT_ASSET_BASE_URL}/channels/stable.json`;
const STAMP_FILE = 'release-build.json';

function usage() {
  console.log(`Usage: node scripts/netlify_r2_release.mjs [options]\n\nOptions:\n  --dist DIR          Built dist directory (default dist-client)\n  --channel-url URL   Artifact channel URL baked into the app shell\n  --asset-base URL    R2/Worker origin for engine/model asset URLs (default https://assets.0x88.app)\n  --build-if-needed   Run the R2/pruned build when the dist stamp is missing/stale\n  --check             Verify the current dist is stamped and pruned; do not build/deploy\n  --deploy            Deploy the verified dist with netlify deploy --no-build\n  --prod              Pass --prod to netlify deploy\n  --message TEXT      Netlify deploy message\n  --npm-bin BIN       npm executable (default npm)\n  --netlify-bin BIN   netlify executable (default netlify)\n  --json              Print machine-readable summary\n  -h, --help          Show help\n`);
}

function parseArgs(argv) {
  const args = {
    dist: 'dist-client',
    channelUrl: process.env.VITE_LC0_ARTIFACT_CHANNEL_URL || DEFAULT_CHANNEL_URL,
    assetBase: process.env.VITE_LC0_BROWSER_ASSET_BASE_URL || DEFAULT_ASSET_BASE_URL,
    buildIfNeeded: false,
    check: false,
    deploy: false,
    prod: false,
    message: undefined,
    npmBin: 'npm',
    netlifyBin: 'netlify',
    json: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--dist' && next) { args.dist = next; i += 1; continue; }
    if (arg === '--channel-url' && next) { args.channelUrl = next; i += 1; continue; }
    if (arg === '--asset-base' && next) { args.assetBase = next.replace(/\/+$/, ''); i += 1; continue; }
    if (arg === '--message' && next) { args.message = next; i += 1; continue; }
    if (arg === '--npm-bin' && next) { args.npmBin = next; i += 1; continue; }
    if (arg === '--netlify-bin' && next) { args.netlifyBin = next; i += 1; continue; }
    if (arg === '--build-if-needed') { args.buildIfNeeded = true; continue; }
    if (arg === '--check') { args.check = true; continue; }
    if (arg === '--deploy') { args.deploy = true; continue; }
    if (arg === '--prod') { args.prod = true; continue; }
    if (arg === '--json') { args.json = true; continue; }
    if (arg === '-h' || arg === '--help') { usage(); process.exit(0); }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (args.check && (args.buildIfNeeded || args.deploy)) {
    throw new Error('--check is verification-only and cannot be combined with --build-if-needed or --deploy');
  }
  return args;
}

function run(command, args, options = {}) {
  const child = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (child.status !== 0) throw new Error(`${command} ${args.join(' ')} failed with status ${child.status}`);
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
  return createHash('sha256').update(await readFile(path)).digest('hex');
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
    },
    inputs: {
      packageJsonSha256: await sha256Path('package.json'),
      packageLockSha256: await sha256Path('package-lock.json'),
      viteConfigSha256: await sha256Path('vite.config.ts'),
      tsconfigSha256: await sha256Path('tsconfig.json'),
      netlifyTomlSha256: await sha256Path('netlify.toml'),
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

function isForbiddenExternalArtifact(name) {
  return name.endsWith('.onnx')
    || name.endsWith('.lc0web')
    || name.endsWith('.wasm')
    || name.endsWith('.data')
    || name.endsWith('.nn')
    || name.endsWith('.nnue')
    || name.endsWith('.bin')
    || name.endsWith('.tar.gz')
    || name.endsWith('.gz')
    || name.endsWith('.br')
    || name.endsWith('.js')
    || name.endsWith('.mjs');
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
  walk(join(root, 'models', 'lc0'), (name, _path, isDir) => isDir ? name.endsWith('.lc0web') : name.endsWith('.onnx'));
  walk(join(root, 'models', 'maia3'), (name, _path, isDir) => !isDir && name.endsWith('.onnx'));
  walk(join(root, 'monty'), (_name, _path, _isDir) => true);
  for (const dir of ['berserk', 'plentychess', 'reckless', 'stockfish', 'viridithas', 'runtimes']) {
    walk(join(root, dir), (name, _path, isDir) => !isDir && isForbiddenExternalArtifact(name));
  }
  return forbidden.map((item) => ({ ...item, path: relative(process.cwd(), item.path) }));
}

function verifyPrunedDist(dist) {
  if (!existsSync(dist)) throw new Error(`Dist directory does not exist: ${dist}`);
  const forbidden = findForbiddenExternalAssets(dist);
  if (forbidden.length) {
    throw new Error(`R2 Netlify dist contains pruned external artifacts: ${forbidden.map((item) => item.path).join(', ')}`);
  }
  return { forbiddenExternalAssets: forbidden };
}

async function main() {
  const args = parseArgs(process.argv);
  const dist = resolve(args.dist);
  const desired = await desiredStamp(args);
  const existing = await readStamp(dist);
  const stampMatches = JSON.stringify(comparableStamp(existing)) === JSON.stringify(comparableStamp(desired));
  let built = false;

  if (!stampMatches) {
    if (!args.buildIfNeeded) {
      throw new Error(`Dist build stamp is ${existing ? 'stale' : 'missing'}; rerun with --build-if-needed to rebuild once`);
    }
    run(args.npmBin, ['run', 'build:netlify:r2'], {
      env: { ...process.env, BUILD_SCOPE: 'product', VITE_LC0_ARTIFACT_CHANNEL_URL: args.channelUrl, VITE_LC0_BROWSER_ASSET_BASE_URL: args.assetBase, NETLIFY_R2_RELEASE_DIST: dist },
    });
    verifyPrunedDist(dist);
    await writeStamp(dist, desired);
    built = true;
  }

  const verification = verifyPrunedDist(dist);

  if (args.deploy) {
    const deployArgs = ['deploy', '--no-build', '--dir', dist];
    if (args.prod) deployArgs.push('--prod');
    if (args.message) deployArgs.push('--message', args.message);
    run(args.netlifyBin, deployArgs);
  }

  const summary = {
    ok: true,
    dist: relative(process.cwd(), dist) || '.',
    stampMatches,
    built,
    deployed: args.deploy,
    artifactChannelUrl: args.channelUrl,
    assetBaseUrl: args.assetBase,
    verification,
  };
  if (args.json) console.log(JSON.stringify(summary, null, 2));
  else console.log(`R2 Netlify release ${args.deploy ? 'deployed' : 'verified'}: built=${built} dist=${summary.dist}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
