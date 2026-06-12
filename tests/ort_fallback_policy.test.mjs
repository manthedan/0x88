import test from 'node:test';
import assert from 'node:assert/strict';
import { shouldFallbackToWasmAfterOrtFailure } from '../src/nn/ortRuntime.ts';

test('strict webgpu sessions do not silently fall back to wasm', () => {
  assert.equal(shouldFallbackToWasmAfterOrtFailure('webgpu', ['webgpu']), false);
  assert.equal(shouldFallbackToWasmAfterOrtFailure('webgpu', ['webgpu', 'wasm']), false);
});

test('auto and explicit webgpu,wasm sessions may fall back to wasm', () => {
  assert.equal(shouldFallbackToWasmAfterOrtFailure('auto', ['webgpu', 'wasm']), true);
  assert.equal(shouldFallbackToWasmAfterOrtFailure('webgpu,wasm', ['webgpu', 'wasm']), true);
  assert.equal(shouldFallbackToWasmAfterOrtFailure('wasm', ['wasm']), false);
});
