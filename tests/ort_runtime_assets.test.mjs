import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { checkOrtRuntimeAssets } from '../scripts/check_ort_runtime_assets.mjs';
import { ORT_RUNTIME_ASSET_FILES } from '../scripts/ort_runtime_assets.mjs';

async function fixture(files) {
  const root = await mkdtemp(join(tmpdir(), 'ort-runtime-assets-'));
  await mkdir(join(root, 'ort'));
  await Promise.all(files.map((name) => writeFile(
    join(root, 'ort', name),
    name === 'ort-wasm-simd-threaded.asyncify.mjs'
      ? 'new Worker(new URL(import.meta.url), {type:"module", name:"em-pthread"});'
      : name,
  )));
  return root;
}

test('ORT runtime allowlist contains only the WebGPU entrypoint runtime pair', () => {
  assert.deepEqual(ORT_RUNTIME_ASSET_FILES, [
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
  ]);
});

test('ORT runtime asset check accepts required files and compressed sidecars', async () => {
  const root = await fixture([...ORT_RUNTIME_ASSET_FILES, ...ORT_RUNTIME_ASSET_FILES.flatMap((name) => [`${name}.br`, `${name}.gz`])]);
  assert.deepEqual(checkOrtRuntimeAssets(root).runtimeFiles, [...ORT_RUNTIME_ASSET_FILES].sort());
});

test('ORT runtime asset check rejects missing and unexpected variants', async () => {
  const compressedOnlyRoot = await fixture(ORT_RUNTIME_ASSET_FILES.map((name) => `${name}.br`));
  assert.throws(() => checkOrtRuntimeAssets(compressedOnlyRoot), /missing: ort-wasm-simd-threaded\.asyncify\.mjs/);
  const missingRoot = await fixture(ORT_RUNTIME_ASSET_FILES.slice(1));
  assert.throws(() => checkOrtRuntimeAssets(missingRoot), /missing: ort-wasm-simd-threaded\.asyncify\.mjs/);
  const extraRoot = await fixture([...ORT_RUNTIME_ASSET_FILES, 'ort-wasm-simd-threaded.jsep.wasm']);
  assert.throws(() => checkOrtRuntimeAssets(extraRoot), /unexpected: ort-wasm-simd-threaded\.jsep\.wasm/);
});

test('ORT runtime asset check rejects glue without the pthread bootstrap', async () => {
  const root = await fixture(ORT_RUNTIME_ASSET_FILES);
  await writeFile(join(root, 'ort', 'ort-wasm-simd-threaded.asyncify.mjs'), 'export default function ort() {}');
  assert.throws(() => checkOrtRuntimeAssets(root), /pthread bootstrap markers missing/);
});
