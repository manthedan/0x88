import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));

test('ORT Relaxed SIMD build config pins matched fixed and relaxed artifacts', () => {
  const result = spawnSync(process.execPath, ['scripts/build_lc0_ort_wasm_relaxed_simd.mjs', '--print-config'], {
    cwd: root,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  const config = JSON.parse(result.stdout);
  assert.equal(config.tag, 'v1.27.0');
  assert.equal(config.commit, '8f0278c77bf44b0cc83c098c6c722b92a36ac4b5');
  assert.equal(config.emscriptenVersion, '4.0.23');
  assert.deepEqual(config.artifacts.fixed, [
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
  ]);
  assert.deepEqual(config.artifacts.relaxed, [
    'ort-wasm-relaxedsimd-threaded.asyncify.mjs',
    'ort-wasm-relaxedsimd-threaded.asyncify.wasm',
  ]);
  assert.equal(config.variants.fixed.includes('--enable_wasm_simd'), true);
  assert.equal(config.variants.fixed.includes('--enable_wasm_relaxed_simd'), false);
  assert.equal(config.variants.fixed.includes('--use_webgpu'), true);
  assert.equal(config.variants.fixed.includes('--use_jsep'), false);
  assert.equal(config.variants.fixed.includes('--enable_wasm_asyncify'), false);
  assert.equal(config.variants.relaxed.includes('--enable_wasm_simd'), true);
  assert.equal(config.variants.relaxed.includes('--enable_wasm_relaxed_simd'), true);
});
