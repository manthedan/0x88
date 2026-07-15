#!/usr/bin/env node
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { ORT_PTHREAD_BOOTSTRAP_FILE, ORT_RUNTIME_ASSET_FILES, isRequiredOrtRuntimeAsset, uncompressedOrtRuntimeAsset } from './ort_runtime_assets.mjs';

function verifyPthreadBootstrap(ortDir) {
  const path = join(ortDir, ORT_PTHREAD_BOOTSTRAP_FILE);
  if (!existsSync(path)) return;
  const source = readFileSync(path, 'utf8');
  const createsSelfWorker = source.includes('new Worker(new URL(import.meta.url)');
  const namesPthreadWorker = /name\s*:\s*["']em-pthread["']/.test(source);
  if (!createsSelfWorker || !namesPthreadWorker) {
    throw new Error(`ORT pthread bootstrap markers missing from ${ORT_PTHREAD_BOOTSTRAP_FILE}`);
  }
}

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
  verifyPthreadBootstrap(ortDir);
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
