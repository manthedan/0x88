#!/usr/bin/env node
import { existsSync, lstatSync, readdirSync, statSync } from 'node:fs';
import { copyFile, link, mkdir, rm } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';

function isExternalArtifact(name) {
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

export function shouldSkipR2PublicAsset(relPath, isDir) {
  const normalized = relPath.replace(/\\/g, '/');
  const parts = normalized.split('/');
  const name = parts[parts.length - 1] ?? '';
  if (normalized === 'monty' || normalized.startsWith('monty/')) return true;
  if (normalized === 'models/monty' || normalized.startsWith('models/monty/')) return true;
  if (parts[0] === 'models' && parts[1] === 'lc0') return isDir ? name.endsWith('.lc0web') : name.endsWith('.onnx');
  if (parts[0] === 'models' && parts[1] === 'maia3') return !isDir && name.endsWith('.onnx');
  if (['berserk', 'plentychess', 'reckless', 'stockfish', 'viridithas', 'runtimes'].includes(parts[0])) return !isDir && isExternalArtifact(name);
  return false;
}

async function linkOrCopy(source, target) {
  await mkdir(dirname(target), { recursive: true });
  try {
    await link(source, target);
  } catch {
    await copyFile(source, target);
  }
}

export async function prepareNetlifyR2PublicAssets(sourceRoot, targetRoot) {
  const source = resolve(sourceRoot);
  const target = resolve(targetRoot);
  if (!existsSync(source)) throw new Error(`Missing public asset root: ${source}`);
  await rm(target, { recursive: true, force: true });
  await mkdir(target, { recursive: true });
  const summary = { copiedFiles: 0, copiedBytes: 0, skippedFiles: 0, skippedBytes: 0, skippedDirs: 0 };

  async function walk(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const sourcePath = join(dir, entry.name);
      const rel = relative(source, sourcePath);
      const isRealDirectory = entry.isDirectory() || (entry.isSymbolicLink() && existsSync(sourcePath) && statSync(sourcePath).isDirectory());
      const isRealFile = entry.isFile() || (entry.isSymbolicLink() && existsSync(sourcePath) && statSync(sourcePath).isFile());
      if (shouldSkipR2PublicAsset(rel, isRealDirectory)) {
        if (isRealDirectory) summary.skippedDirs += 1;
        else {
          summary.skippedFiles += 1;
          summary.skippedBytes += lstatSync(sourcePath).size;
        }
        continue;
      }
      if (isRealDirectory) {
        await walk(sourcePath);
      } else if (isRealFile) {
        await linkOrCopy(sourcePath, join(target, rel));
        summary.copiedFiles += 1;
        summary.copiedBytes += statSync(sourcePath).size;
      }
    }
  }

  await walk(source);
  return { source: relative(process.cwd(), source) || '.', target: relative(process.cwd(), target) || '.', ...summary };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const source = process.argv[2] ?? 'public';
  const target = process.argv[3];
  if (!target) {
    console.error('Usage: node scripts/prepare_netlify_r2_public_assets.mjs SOURCE TARGET');
    process.exit(1);
  }
  prepareNetlifyR2PublicAssets(source, target)
    .then((summary) => console.log(JSON.stringify({ status: 'R2_PUBLIC_ASSETS_PREPARED', ...summary }, null, 2)))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}
