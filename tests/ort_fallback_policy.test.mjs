import test from 'node:test';
import assert from 'node:assert/strict';
import {
  requestedOrtWasmArtifact,
  resolvedOrtExecutionProviders,
  setRequestedOrtExecutionProviderForCurrentThread,
  setRequestedOrtWasmArtifactForCurrentThread,
  setRequestedOrtWasmThreadsForCurrentThread,
  shouldFallbackToWasmAfterOrtFailure,
  validateOrtWasmArtifactSelection,
} from '../src/nn/ortRuntime.ts';

test('strict webgpu sessions do not silently fall back to wasm', () => {
  assert.equal(shouldFallbackToWasmAfterOrtFailure('webgpu', ['webgpu']), false);
  assert.equal(shouldFallbackToWasmAfterOrtFailure('webgpu', ['webgpu', 'wasm']), false);
});

test('auto and explicit webgpu,wasm sessions may fall back to wasm', () => {
  assert.equal(shouldFallbackToWasmAfterOrtFailure('auto', ['webgpu', 'wasm']), true);
  assert.equal(shouldFallbackToWasmAfterOrtFailure('webgpu,wasm', ['webgpu', 'wasm']), true);
  assert.equal(shouldFallbackToWasmAfterOrtFailure('wasm', ['wasm']), false);
});

test('strict webgpu provider selection never resolves directly to wasm', () => {
  setRequestedOrtExecutionProviderForCurrentThread('webgpu');
  try {
    assert.deepEqual(resolvedOrtExecutionProviders(), ['webgpu']);
  } finally {
    setRequestedOrtExecutionProviderForCurrentThread(null);
  }
});

test('custom ORT WASM artifacts require matched glue and binary URLs', () => {
  assert.throws(
    () => validateOrtWasmArtifactSelection({ variant: 'relaxed', wasmUrl: '/ort/relaxed.wasm' }),
    /matching mjsUrl and wasmUrl/,
  );
  assert.throws(
    () => validateOrtWasmArtifactSelection({ variant: 'fixed' }),
    /requires matching mjsUrl and wasmUrl/,
  );
  assert.throws(
    () => validateOrtWasmArtifactSelection({ variant: 'bundled', mjsUrl: '/ort/fixed.mjs', wasmUrl: '/ort/fixed.wasm' }),
    /bundled ORT WASM variant cannot specify/,
  );
});

test('ORT WASM artifact selection preserves explicit runtime identity', () => {
  const selection = {
    variant: 'relaxed',
    mjsUrl: '/ort/relaxed.mjs',
    wasmUrl: '/ort/relaxed.wasm',
    artifactId: 'ort-relaxed-test',
  };
  setRequestedOrtWasmArtifactForCurrentThread(selection);
  try {
    assert.deepEqual(requestedOrtWasmArtifact(), selection);
    assert.notEqual(requestedOrtWasmArtifact(), selection);
  } finally {
    setRequestedOrtWasmArtifactForCurrentThread(null);
  }
});

test('ORT WASM thread pinning rejects invalid values', () => {
  assert.throws(() => setRequestedOrtWasmThreadsForCurrentThread(0), /positive integer/);
  assert.throws(() => setRequestedOrtWasmThreadsForCurrentThread(1.5), /positive integer/);
  setRequestedOrtWasmThreadsForCurrentThread(1);
  setRequestedOrtWasmThreadsForCurrentThread(null);
});
