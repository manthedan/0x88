import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { chmod, mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

async function fakeBin(path, body) {
  await writeFile(path, `#!/bin/sh\n${body}\n`);
  await chmod(path, 0o755);
}

test('netlify_r2_release builds once, stamps dist, then deploys with no-build without rebuilding', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-netlify-r2-release-'));
  const dist = join(root, 'dist-client');
  const npmLog = join(root, 'npm.log');
  const netlifyLog = join(root, 'netlify.log');
  const npm = join(root, 'fake-npm.sh');
  const netlify = join(root, 'fake-netlify.sh');
  await fakeBin(npm, 'printf "%s\\n" "$*" >> "$NPM_LOG"\nmkdir -p "$NETLIFY_R2_RELEASE_DIST/models/lc0"\nprintf "{}\\n" > "$NETLIFY_R2_RELEASE_DIST/models/lc0/manifest.json"\nexit 0');
  await fakeBin(netlify, 'printf "%s\\n" "$*" >> "$NETLIFY_LOG"\nexit 0');

  const first = spawnSync(process.execPath, [
    'scripts/netlify_r2_release.mjs',
    '--dist', dist,
    '--build-if-needed',
    '--npm-bin', npm,
    '--json',
  ], { cwd: process.cwd(), encoding: 'utf8', env: { ...process.env, NPM_LOG: npmLog, NETLIFY_LOG: netlifyLog } });
  assert.equal(first.status, 0, first.stderr);
  const firstSummary = JSON.parse(first.stdout);
  assert.equal(firstSummary.built, true);
  const stamp = JSON.parse(await readFile(join(dist, 'release-build.json'), 'utf8'));
  assert.equal(stamp.schema, 'lc0_browser.netlify_r2_release_build.v1');
  assert.equal(stamp.artifactChannelUrl, 'https://assets.0x88.app/channels/stable.json');
  assert.deepEqual(stamp.viteEnv, {
    VITE_LC0_BROWSER_ASSET_BASE_URL: 'https://assets.0x88.app',
    VITE_LC0_MODEL_BASE_URL: '',
  });

  const second = spawnSync(process.execPath, [
    'scripts/netlify_r2_release.mjs',
    '--dist', dist,
    '--build-if-needed',
    '--deploy',
    '--prod',
    '--message', 'test deploy',
    '--npm-bin', npm,
    '--netlify-bin', netlify,
    '--json',
  ], { cwd: process.cwd(), encoding: 'utf8', env: { ...process.env, NPM_LOG: npmLog, NETLIFY_LOG: netlifyLog } });
  assert.equal(second.status, 0, second.stderr);
  const secondSummary = JSON.parse(second.stdout);
  assert.equal(secondSummary.built, false);
  assert.equal(secondSummary.deployed, true);
  assert.equal((await readFile(npmLog, 'utf8')).trim().split('\n').length, 1);
  const deployLog = await readFile(netlifyLog, 'utf8');
  assert.match(deployLog, /deploy --no-build --dir .* --prod --message test deploy/);
});

test('netlify_r2_release check mode rejects side-effect flags', async () => {
  const result = spawnSync(process.execPath, [
    'scripts/netlify_r2_release.mjs',
    '--check',
    '--build-if-needed',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /--check is verification-only/);
});

test('prune_external_model_assets removes Monty from R2 Netlify dist', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-netlify-r2-prune-monty-'));
  const dist = join(root, 'dist-client');
  await mkdir(join(dist, 'models', 'monty'), { recursive: true });
  await mkdir(join(dist, 'monty'), { recursive: true });
  await writeFile(join(dist, 'models', 'monty', 'nn.network'), 'abc');
  await writeFile(join(dist, 'monty', 'monty.wasm'), 'abc');

  const result = spawnSync(process.execPath, [
    'scripts/prune_external_model_assets.mjs',
    dist,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(existsSync(join(dist, 'models', 'monty')), false);
  assert.equal(existsSync(join(dist, 'monty')), false);
  assert.match(result.stdout, /monty/);
});

test('precompress_engine_artifacts skips Monty artifacts', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-precompress-skip-monty-'));
  await mkdir(join(root, 'monty'), { recursive: true });
  await mkdir(join(root, 'models'), { recursive: true });
  await writeFile(join(root, 'monty', 'monty.wasm'), 'abc');
  await writeFile(join(root, 'models', 'tiny.wasm'), 'abc');

  const result = spawnSync(process.execPath, [
    'scripts/precompress_engine_artifacts.mjs',
    root,
    '--allow-missing',
    '--exclude', 'monty',
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(existsSync(join(root, 'monty', 'monty.wasm.gz')), false);
  assert.equal(existsSync(join(root, 'monty', 'monty.wasm.br')), false);
  assert.equal(existsSync(join(root, 'models', 'tiny.wasm.gz')), true);
  assert.equal(existsSync(join(root, 'models', 'tiny.wasm.br')), true);
});

test('netlify_r2_release rejects a dist that still contains pruned external blobs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lc0-netlify-r2-bad-dist-'));
  const dist = join(root, 'dist-client');
  await mkdir(join(dist, 'models/lc0'), { recursive: true });
  await mkdir(join(dist, 'models/monty'), { recursive: true });
  await mkdir(join(dist, 'monty'), { recursive: true });
  await mkdir(join(dist, 'stockfish'), { recursive: true });
  await writeFile(join(dist, 'models/lc0/test.onnx'), 'abc');
  await writeFile(join(dist, 'models/monty/nn.network'), 'abc');
  await writeFile(join(dist, 'monty/monty.wasm'), 'abc');
  await writeFile(join(dist, 'stockfish/stockfish-18-lite.js'), 'abc');
  const npm = join(root, 'fake-npm.sh');
  await fakeBin(npm, 'exit 0');
  const result = spawnSync(process.execPath, [
    'scripts/netlify_r2_release.mjs',
    '--dist', dist,
    '--build-if-needed',
    '--npm-bin', npm,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /R2 Netlify dist contains pruned external artifacts/);
  assert.match(result.stderr, /models\/lc0\/test\.onnx/);
  assert.match(result.stderr, /models\/monty\/nn\.network/);
  assert.match(result.stderr, /monty\/monty\.wasm/);
  assert.match(result.stderr, /stockfish\/stockfish-18-lite\.js/);
});

test('netlify.toml and package scripts use the R2-pruned build path', async () => {
  const netlifyToml = await readFile('netlify.toml', 'utf8');
  assert.match(netlifyToml, /build:netlify:r2/);
  assert.match(netlifyToml, /VITE_LC0_ARTIFACT_CHANNEL_URL=https:\/\/assets\.0x88\.app\/channels\/stable\.json/);
  assert.match(netlifyToml, /VITE_LC0_BROWSER_ASSET_BASE_URL=https:\/\/assets\.0x88\.app/);
  const packageJson = JSON.parse(await readFile('package.json', 'utf8'));
  assert.match(packageJson.scripts['build:netlify:r2'], /NETLIFY_R2_RELEASE_DIST:-dist-client/);
  assert.match(packageJson.scripts['build:netlify:r2'], /prune_external_model_assets/);
  assert.match(packageJson.scripts['build:netlify:r2'], /--exclude monty/);
});
