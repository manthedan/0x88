#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync, copyFileSync } from 'node:fs';
import { basename, dirname, relative, resolve } from 'node:path';

const mode = process.argv.includes('--copy') ? 'copy' : 'symlink';
const repoRoot = resolve(new URL('..', import.meta.url).pathname);
const workspaceRoot = resolve(repoRoot, '..');
const sourceDir = resolve(workspaceRoot, 'models/lc0-bestnets/onnx');
const packSourceDir = resolve(workspaceRoot, 'models/lc0-bestnets/lc0web');
const publicDir = resolve(repoRoot, 'public/models/lc0');
const files = [
  't1-256x10-distilled-swa-2432500.batch1.f32.onnx',
  't1-256x10-distilled-swa-2432500.batch1.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch4.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch8.f16.onnx',
  // batch16/32 exported 2026-06-09 for the TVMJS batch-scaling lane:
  // lc0 leela2onnx --onnx-batch-size=N --onnx-data-type=f16
  't1-256x10-distilled-swa-2432500.batch16.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch32.f16.onnx',
  // Lc0 BT4 (1024x15x32h attention net, ~353MB f16). Heavy: browser use is
  // WebGPU-gated and lazy-loaded; see src/lc0/bt4Engine.ts.
  'BT4-1024x15x32h-swa-6147500.batch1.f16.onnx',
];

const packDirs = [
  't1-256x10-distilled-swa-2432500.batch8.f16.lc0web',
];

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function exposeAsset(source, target) {
  rmSync(target, { force: true, recursive: true });
  if (mode === 'copy') {
    copyFileSync(source, target);
  } else {
    symlinkSync(relative(dirname(target), source), target);
  }
}

mkdirSync(publicDir, { recursive: true });
const models = [];
for (const file of files) {
  const source = resolve(sourceDir, file);
  if (!existsSync(source)) throw new Error(`Missing LC0 ONNX source model: ${source}`);
  const target = resolve(publicDir, file);
  exposeAsset(source, target);
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

const packs = [];
for (const packDir of packDirs) {
  const source = resolve(packSourceDir, packDir);
  if (!existsSync(source)) continue;
  const target = resolve(publicDir, packDir);
  rmSync(target, { force: true, recursive: true });
  mkdirSync(target, { recursive: true });
  const packFiles = readdirSync(source).filter((file) => !file.startsWith('.')).sort();
  for (const file of packFiles) {
    exposeAsset(resolve(source, file), resolve(target, file));
  }
  const packManifestPath = resolve(source, 'model.lc0web.json');
  const packManifest = JSON.parse(readFileSync(packManifestPath, 'utf8'));
  const metadataBytes = lstatSync(packManifestPath).size;
  const shardBytes = packManifest.weights.shards.reduce((sum, shard) => sum + shard.bytes, 0);
  packs.push({
    id: packDir,
    url: `/models/lc0/${packDir}/model.lc0web.json`,
    mode,
    source: relative(repoRoot, source),
    format: packManifest.format,
    version: packManifest.version,
    sourceSha256: packManifest.model.sourceSha256,
    packSha256: packManifest.packSha256,
    metadataBytes,
    shardBytes,
    tensorCount: packManifest.weights.tensorCount,
    shards: packManifest.weights.shards,
    recommendedRuntime: packManifest.model.recommendedRuntime,
    layout: packManifest.model.layout,
  });
}

const manifest = {
  generatedBy: 'scripts/lc0_prepare_model_assets.mjs',
  note: 'Local LC0 browser model assets. The large ONNX/model-pack files are exposed as symlinks by default so they are not committed as blobs.',
  models,
  packs,
};
writeFileSync(resolve(publicDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));
