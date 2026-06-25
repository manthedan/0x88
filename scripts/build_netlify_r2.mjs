#!/usr/bin/env node
import { mkdtemp, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { prepareNetlifyR2PublicAssets } from './prepare_netlify_r2_public_assets.mjs';

const dist = resolve(process.env.NETLIFY_R2_RELEASE_DIST || process.argv[2] || 'dist-client');
const precompressCacheDir = resolve(process.env.NETLIFY_R2_PRECOMPRESS_CACHE_DIR || '.local-dev-artifacts/precompress-r2');

function formatMs(ms) {
  return `${(ms / 1000).toFixed(2)}s`;
}

async function timed(name, fn, timings) {
  const started = performance.now();
  try {
    return await fn();
  } finally {
    const ms = performance.now() - started;
    timings.push({ name, ms });
    console.error(`[netlify-r2-build] ${name}: ${formatMs(ms)}`);
  }
}

function run(command, args, options = {}) {
  const child = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (child.status !== 0) throw new Error(`${command} ${args.join(' ')} failed with status ${child.status}`);
}

async function main() {
  const timings = [];
  const publicDir = await mkdtemp(join(tmpdir(), 'lc0-netlify-r2-public-'));
  try {
    const prep = await timed('prepare pruned public assets', () => prepareNetlifyR2PublicAssets('public', publicDir), timings);
    console.error(`[netlify-r2-build] public assets: copied ${prep.copiedFiles} files, skipped ${prep.skippedFiles} files and ${prep.skippedDirs} dirs (${(prep.skippedBytes / 1_000_000).toFixed(1)} MB)`);
    await timed('vite build', () => {
      run('vite', ['build', '--outDir', dist], {
        env: {
          ...process.env,
          BUILD_SCOPE: process.env.BUILD_SCOPE || 'product',
          NETLIFY_R2_RELEASE_DIST: dist,
          NETLIFY_R2_PUBLIC_ASSETS: publicDir,
        },
      });
    }, timings);
    await timed('prune external assets', () => {
      run(process.execPath, ['scripts/prune_external_model_assets.mjs', dist]);
    }, timings);
    await timed('precompress artifacts', () => {
      run(process.execPath, ['scripts/precompress_engine_artifacts.mjs', dist, '--allow-missing', '--exclude', 'monty', '--cache-dir', precompressCacheDir]);
    }, timings);
    const total = timings.reduce((sum, entry) => sum + entry.ms, 0);
    console.error(`[netlify-r2-build] total: ${formatMs(total)} dist=${relative(process.cwd(), dist) || '.'}`);
  } finally {
    await rm(publicDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
