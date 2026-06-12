#!/usr/bin/env node
// Build Monty (official-monty/Monty) for wasm32-wasip1.
//
// Unlike the other Rust engines, Monty's networks are NOT embedded: the wasm
// is built without the `embed` feature and reads the raw nets from WASI
// preopened files at runtime (nn-<sha12>.network in the preopened cwd).
// The ~950MB of raw nets ship as separate cacheable assets; see
// scripts/monty_wasi_smoke.mjs for the loading contract.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const engineDir = process.env.MONTY_BUILD_DIR ?? path.join(root, '.local_engines', 'monty-wasi-src');
const out = process.env.MONTY_WASM_OUT ?? path.join(root, 'public', 'monty', 'monty.wasm');
const repo = process.env.MONTY_REPO ?? 'https://github.com/official-monty/Monty.git';
const ref = process.env.MONTY_REF ?? '0950aff1604024fbd1469aedff12cc9460903b43';
const patchPath = path.join(root, 'patches', 'monty-wasip1.patch');

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function canRun(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: 'ignore', ...options });
  return result.status === 0;
}

function applyPatchOnce() {
  if (canRun('git', ['apply', '--check', patchPath], { cwd: engineDir })) {
    run('git', ['apply', patchPath], { cwd: engineDir });
    return;
  }
  if (canRun('git', ['apply', '--reverse', '--check', patchPath], { cwd: engineDir })) {
    console.log(`Patch already applied: ${patchPath}`);
    return;
  }
  run('git', ['apply', patchPath], { cwd: engineDir });
}

fs.mkdirSync(path.dirname(engineDir), { recursive: true });
fs.mkdirSync(path.dirname(out), { recursive: true });

if (process.env.MONTY_SKIP_GIT !== '1') {
  if (!fs.existsSync(path.join(engineDir, '.git'))) {
    run('git', ['clone', repo, engineDir]);
  }
  run('git', ['fetch', '--tags', 'origin'], { cwd: engineDir });
  run('git', ['checkout', ref], { cwd: engineDir });
  run('git', ['reset', '--hard'], { cwd: engineDir });
  run('git', ['clean', '-fdx', '-e', 'target'], { cwd: engineDir });
}

applyPatchOnce();
run('rustup', ['target', 'add', 'wasm32-wasip1']);
const targetFeatures = [
  '+bulk-memory',
  '+simd128',
  ...(process.env.MONTY_WASM_RELAXED_SIMD === '1' ? ['+relaxed-simd'] : []),
].join(',');
run('cargo', ['build', '--release', '--target', 'wasm32-wasip1'], {
  cwd: engineDir,
  env: { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} -C target-feature=${targetFeatures}`.trim() },
});

const built = path.join(engineDir, 'target', 'wasm32-wasip1', 'release', 'monty.wasm');
fs.copyFileSync(built, out);
console.log(`Wrote ${out}`);
