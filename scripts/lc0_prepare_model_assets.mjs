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
  // Default Play/analysis small-net model: weight-only int8 QDQ of the f16
  // batch1 export (81MB f32 -> 20.6MB; parity 38/40 top-1, drift <=0.026;
  // ~15-20% slower on WebGPU, ~2x on the wasm fallback — a download trade).
  't1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx',
  't1-256x10-distilled-swa-2432500.batch1.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch4.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch8.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx',
  // batch16/32 exported 2026-06-09 for the TVMJS batch-scaling lane:
  // lc0 leela2onnx --onnx-batch-size=N --onnx-data-type=f16
  't1-256x10-distilled-swa-2432500.batch16.f16.onnx',
  't1-256x10-distilled-swa-2432500.batch32.f16.onnx',
  // Lc0 BT4 (1024x15x32h attention net, ~353MB f16). Heavy: browser use is
  // WebGPU-gated and lazy-loaded; see src/lc0/bt4Engine.ts.
  'BT4-1024x15x32h-swa-6147500.batch1.f16.onnx',
  // BT4-it332 (policytune-332) fixed-batch f16 exports for the TVMJS/WebGPU
  // lane, 2026-06-09: lc0 leela2onnx --onnx-batch-size={1,4,8} --onnx-data-type=f16
  'BT4-1024x15x32h-swa-6147500-policytune-332.batch1.f16.onnx',
  'BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx',
  'BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.onnx',
  // t3-512x15x16h-distill, 2026-06-10: the mid-rung of the t1 -> t3 -> BT4
  // progressive ladder. lc0 leela2onnx --onnx-batch-size={1,4,8,16} f16.
  't3-512x15x16h-distill-swa-2767500.batch1.f16.onnx',
  't3-512x15x16h-distill-swa-2767500.batch4.f16.onnx',
  't3-512x15x16h-distill-swa-2767500.batch8.f16.onnx',
  't3-512x15x16h-distill-swa-2767500.batch16.f16.onnx',
];

const packDirs = [
  't1-256x10-distilled-swa-2432500.batch8.f16.lc0web',
];

// LeelaQueenOdds v2 (notune/LeelaQueenOdds, the public net behind the Lichess
// queen-odds bot). T-era attention net; converted 2026-06-11 with:
// lc0 leela2onnx --onnx-data-type=f16. WebGPU-gated big-net (bt4Engine.ts).
const oddsSourceDir = resolve(workspaceRoot, 'models/odds/onnx');
const oddsFiles = [
  'lqo_v2.f16.qdq8.onnx',
  'lqo_v2.f16.onnx',
];

// The Maia-1 per-level nets (maia-NNNN.f32.onnx) were retired 2026-06-12:
// the Play page's fixed-rating human opponents now run on Maia3 conditioned
// at each level (one 28MB model instead of five 3.5MB nets; head-to-head
// Maia3(1500) beat sampled Maia-1500 +7 =1 -0). The source nets remain in
// models/maia/onnx for the calibration scripts.
const maiaSourceDir = resolve(workspaceRoot, 'models/maia/onnx');
const maiaFiles = [];

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

for (const file of oddsFiles) {
  const source = resolve(oddsSourceDir, file);
  if (!existsSync(source)) throw new Error(`Missing odds ONNX source model: ${source}`);
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
    ...(file === 'lqo_v2.f16.qdq8.onnx' ? { derivedFrom: 'lqo_v2.f16.onnx' } : {}),
  });
}

for (const file of maiaFiles) {
  const source = resolve(maiaSourceDir, file);
  if (!existsSync(source)) throw new Error(`Missing Maia ONNX source model: ${source}`);
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
