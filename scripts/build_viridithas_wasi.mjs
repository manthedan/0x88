#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const engineDir = process.env.VIRIDITHAS_BUILD_DIR ?? path.join(root, '.local_engines', 'viridithas-wasi-src');
const netDir = process.env.VIRIDITHAS_NET_DIR ?? path.join(root, '.local_engines', 'viridithas-nets');
const out = process.env.VIRIDITHAS_WASM_OUT ?? path.join(root, 'public', 'viridithas', 'viridithas.wasm');
const repo = process.env.VIRIDITHAS_REPO ?? 'https://github.com/cosmobobak/viridithas.git';
const ref = process.env.VIRIDITHAS_REF ?? '20d7402065cae084715183e019fdd18089e2dfac';
const netUrl = process.env.VIRIDITHAS_NET_URL ?? 'https://github.com/cosmobobak/viridithas-networks/releases/download/v106/atlantis-b800.nnue.zst';
const netSha256 = process.env.VIRIDITHAS_NET_SHA256 ?? '2d387407b926df4dbda441cdc3e2288fee2e6a2afa8e1bd22262309ec0fb668a';
const netPath = path.join(netDir, path.basename(new URL(netUrl).pathname));
const patchPath = path.join(root, 'patches', 'viridithas-wasip1.patch');

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function canRun(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: 'ignore', ...options });
  return result.status === 0;
}

function sha256(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
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
fs.mkdirSync(netDir, { recursive: true });
fs.mkdirSync(path.dirname(out), { recursive: true });

if (process.env.VIRIDITHAS_SKIP_GIT !== '1') {
  if (!fs.existsSync(path.join(engineDir, '.git'))) {
    run('git', ['clone', repo, engineDir]);
  }
  run('git', ['fetch', '--tags', 'origin'], { cwd: engineDir });
  run('git', ['checkout', ref], { cwd: engineDir });
  run('git', ['reset', '--hard'], { cwd: engineDir });
  run('git', ['clean', '-fdx', '-e', 'target'], { cwd: engineDir });
}

if (!fs.existsSync(netPath)) {
  run('curl', ['-L', '--fail', '-o', netPath, netUrl]);
}
const actualNetSha256 = sha256(netPath);
if (actualNetSha256 !== netSha256) {
  throw new Error(`Viridithas network checksum mismatch for ${netPath}: expected ${netSha256}, got ${actualNetSha256}`);
}
fs.copyFileSync(netPath, path.join(engineDir, 'viridithas.nnue.zst'));

applyPatchOnce();
run('rustup', ['target', 'add', 'wasm32-wasip1']);
const targetFeatures = ['+bulk-memory', ...(process.env.VIRIDITHAS_WASM_SIMD === '1' ? ['+simd128'] : [])].join(',');
run('cargo', ['build', '--release', '--target', 'wasm32-wasip1', '--no-default-features'], {
  cwd: engineDir,
  env: { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} -C target-feature=${targetFeatures}`.trim() },
});

const built = path.join(engineDir, 'target', 'wasm32-wasip1', 'release', 'viridithas.wasm');
fs.copyFileSync(built, out);
console.log(`Wrote ${out}`);
