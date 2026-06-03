#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync, copyFileSync } from 'node:fs';
import { basename, dirname, relative, resolve } from 'node:path';

const mode = process.argv.includes('--copy') ? 'copy' : 'symlink';
const repoRoot = resolve(new URL('..', import.meta.url).pathname);
const workspaceRoot = resolve(repoRoot, '..');
const sourceDir = resolve(workspaceRoot, 'models/lc0-bestnets/onnx');
const publicDir = resolve(repoRoot, 'public/models/lc0');
const files = [
  't1-256x10-distilled-swa-2432500.batch1.f32.onnx',
  't1-256x10-distilled-swa-2432500.batch1.f16.onnx',
];

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

mkdirSync(publicDir, { recursive: true });
const models = [];
for (const file of files) {
  const source = resolve(sourceDir, file);
  if (!existsSync(source)) throw new Error(`Missing LC0 ONNX source model: ${source}`);
  const target = resolve(publicDir, file);
  rmSync(target, { force: true });
  if (mode === 'copy') {
    copyFileSync(source, target);
  } else {
    symlinkSync(relative(dirname(target), source), target);
  }
  const stat = lstatSync(target);
  models.push({
    file,
    url: `/models/lc0/${file}`,
    mode: stat.isSymbolicLink() ? 'symlink' : 'copy',
    source: relative(repoRoot, source),
    bytes: lstatSync(source).size,
    sha256: sha256(source),
  });
}

const manifest = {
  generatedBy: 'scripts/lc0_prepare_model_assets.mjs',
  note: 'Local LC0 browser model assets. The large ONNX files are exposed as symlinks by default so they are not committed as blobs.',
  models,
};
writeFileSync(resolve(publicDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));
