#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const engineDir = path.join(root, '.local_engines', 'viridithas-wasi-src');
const netDir = path.join(root, '.local_engines', 'viridithas-nets');
const out = process.env.VIRIDITHAS_WASM_OUT ?? path.join(root, 'public', 'viridithas', 'viridithas.wasm');
const repo = process.env.VIRIDITHAS_REPO ?? 'https://github.com/cosmobobak/viridithas.git';
const ref = process.env.VIRIDITHAS_REF ?? '20d7402065cae084715183e019fdd18089e2dfac';
const netUrl = process.env.VIRIDITHAS_NET_URL ?? 'https://github.com/cosmobobak/viridithas-networks/releases/download/v106/atlantis-b800.nnue.zst';
const netPath = path.join(netDir, path.basename(new URL(netUrl).pathname));
const patchPath = path.join(root, 'patches', 'viridithas-wasip1.patch');

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

fs.mkdirSync(path.dirname(engineDir), { recursive: true });
fs.mkdirSync(netDir, { recursive: true });
fs.mkdirSync(path.dirname(out), { recursive: true });

if (!fs.existsSync(path.join(engineDir, '.git'))) {
  run('git', ['clone', repo, engineDir]);
}
run('git', ['fetch', '--tags', 'origin'], { cwd: engineDir });
run('git', ['checkout', ref], { cwd: engineDir });
run('git', ['reset', '--hard'], { cwd: engineDir });
run('git', ['clean', '-fdx', '-e', 'target'], { cwd: engineDir });

if (!fs.existsSync(netPath)) {
  run('curl', ['-L', '--fail', '-o', netPath, netUrl]);
}
fs.copyFileSync(netPath, path.join(engineDir, 'viridithas.nnue.zst'));

run('git', ['apply', patchPath], { cwd: engineDir });
run('rustup', ['target', 'add', 'wasm32-wasip1']);
run('cargo', ['build', '--release', '--target', 'wasm32-wasip1', '--no-default-features'], {
  cwd: engineDir,
  env: { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} -C target-feature=+bulk-memory`.trim() },
});

const built = path.join(engineDir, 'target', 'wasm32-wasip1', 'release', 'viridithas.wasm');
fs.copyFileSync(built, out);
console.log(`Wrote ${out}`);
