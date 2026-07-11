#!/usr/bin/env node
import { existsSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { ORT_RUNTIME_ASSET_FILES, isRequiredOrtRuntimeAsset, uncompressedOrtRuntimeAsset } from './ort_runtime_assets.mjs';

export function checkOrtRuntimeAssets(rootPath) {
  const root = resolve(rootPath);
  const ortDir = join(root, 'ort');
  if (!existsSync(ortDir)) throw new Error(`Missing staged ORT directory: ${ortDir}`);
  const files = readdirSync(ortDir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();
  const baseFiles = new Set(files.map(uncompressedOrtRuntimeAsset));
  const missing = ORT_RUNTIME_ASSET_FILES.filter((name) => !files.includes(name));
  const unexpected = [...baseFiles].filter((name) => !isRequiredOrtRuntimeAsset(name)).sort();
  if (missing.length || unexpected.length) {
    throw new Error(`ORT runtime asset allowlist mismatch${missing.length ? `; missing: ${missing.join(', ')}` : ''}${unexpected.length ? `; unexpected: ${unexpected.join(', ')}` : ''}`);
  }
  return { root, files, runtimeFiles: [...baseFiles].sort() };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const result = checkOrtRuntimeAssets(process.argv[2] ?? 'dist-client');
    console.log(`ORT runtime assets verified: ${result.runtimeFiles.join(', ')}`);
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}
