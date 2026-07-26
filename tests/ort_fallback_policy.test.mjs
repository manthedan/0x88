import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ORT_RUNTIME_ARTIFACT_FILES,
  ortRuntimeArtifactKindForCurrentThread,
  requestedOrtWasmArtifact,
  resolveOrtPthreadRuntimeUrls,
  resolvedOrtExecutionProviders,
  setOrtRuntimeArtifactKindForCurrentThread,
  setRequestedOrtExecutionProviderForCurrentThread,
  setRequestedOrtWasmArtifactForCurrentThread,
  setRequestedOrtWasmThreadsForCurrentThread,
  shouldFallbackToWasmAfterOrtFailure,
  validateOrtWasmArtifactSelection,
} from '../src/nn/ortRuntime.ts';
import { ORT_RUNTIME_ASSET_FILES } from '../scripts/ort_runtime_assets.mjs';

/** Simulate a browser thread (Node has no `location` and no `navigator.gpu`). */
function withBrowserEnv(webgpuPresent, fn) {
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const previousLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'navigator', {
    value: webgpuPresent ? { gpu: {}, hardwareConcurrency: 8 } : { hardwareConcurrency: 8 },
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, 'location', {
    value: { href: 'https://0x88.app/app/analysis', search: '' },
    configurable: true,
    writable: true,
  });
  try {
    return fn();
  } finally {
    if (previousNavigator) Object.defineProperty(globalThis, 'navigator', previousNavigator);
    else delete globalThis.navigator;
    if (previousLocation) Object.defineProperty(globalThis, 'location', previousLocation);
    else delete globalThis.location;
  }
}

function withEp(ep, fn) {
  setRequestedOrtExecutionProviderForCurrentThread(ep);
  try {
    return fn();
  } finally {
    setRequestedOrtExecutionProviderForCurrentThread(null);
  }
}

test('ORT pthread sidecar URLs resolve from root and deployed subpath bases', () => {
  assert.deepEqual(resolveOrtPthreadRuntimeUrls('/ort/', 'https://0x88.app/app/analysis/'), {
    mjs: 'https://0x88.app/ort/ort-wasm-simd-threaded.asyncify.mjs',
    wasm: 'https://0x88.app/ort/ort-wasm-simd-threaded.asyncify.wasm',
  });
  assert.deepEqual(resolveOrtPthreadRuntimeUrls('/chess/ort', 'https://0x88.app/chess/app/analysis/'), {
    mjs: 'https://0x88.app/chess/ort/ort-wasm-simd-threaded.asyncify.mjs',
    wasm: 'https://0x88.app/chess/ort/ort-wasm-simd-threaded.asyncify.wasm',
  });
});

test('CPU-only ORT runtime URLs resolve from the same staged base', () => {
  assert.deepEqual(resolveOrtPthreadRuntimeUrls('/ort/', 'https://0x88.app/app/analysis/', 'wasm'), {
    mjs: 'https://0x88.app/ort/ort-wasm-simd-threaded.mjs',
    wasm: 'https://0x88.app/ort/ort-wasm-simd-threaded.wasm',
  });
});

test('runtime artifact filenames match the deploy staging allowlist', () => {
  assert.deepEqual([
    ORT_RUNTIME_ARTIFACT_FILES.asyncify.mjs,
    ORT_RUNTIME_ARTIFACT_FILES.asyncify.wasm,
    ORT_RUNTIME_ARTIFACT_FILES.wasm.mjs,
    ORT_RUNTIME_ARTIFACT_FILES.wasm.wasm,
  ], [...ORT_RUNTIME_ASSET_FILES]);
});

test('browsers without WebGPU load the CPU-only ORT runtime for every non-strict EP', () => {
  withBrowserEnv(false, () => {
    for (const ep of [null, 'wasm', 'auto', 'webgpu,wasm']) {
      withEp(ep, () => assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'wasm', `ep=${ep}`));
    }
    // Strict WebGPU keeps the GPU build so the failure stays "WebGPU unavailable".
    withEp('webgpu', () => assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'asyncify'));
  });
});

test('WebGPU-capable threads keep the asyncify runtime unless wasm is pinned up front', () => {
  withBrowserEnv(true, () => {
    for (const ep of ['auto', 'webgpu', 'webgpu,wasm']) {
      withEp(ep, () => assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'asyncify', `ep=${ep}`));
    }
    // A per-message programmatic wasm request must not swap the worker-global
    // binary: the same worker may be asked for WebGPU on the next message.
    withEp('wasm', () => assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'asyncify'));
    // An explicit env/URL pin is a whole-thread decision, so it does select the
    // CPU-only build.
    const previous = process.env.ORT_EXECUTION_PROVIDERS;
    process.env.ORT_EXECUTION_PROVIDERS = 'wasm';
    try {
      assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'wasm');
    } finally {
      if (previous === undefined) delete process.env.ORT_EXECUTION_PROVIDERS;
      else process.env.ORT_EXECUTION_PROVIDERS = previous;
    }
    // Callers that own a dedicated wasm-only worker can opt in explicitly.
    setOrtRuntimeArtifactKindForCurrentThread('wasm');
    try {
      withEp('wasm', () => assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'wasm'));
    } finally {
      setOrtRuntimeArtifactKindForCurrentThread(null);
    }
  });
});

test('Node keeps the asyncify runtime because its ORT glue is inlined', () => {
  withEp('wasm', () => assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'asyncify'));
});

test('experimental ORT WASM artifacts keep the asyncify runtime identity', () => {
  withBrowserEnv(false, () => {
    setRequestedOrtWasmArtifactForCurrentThread({
      variant: 'fixed',
      mjsUrl: '/ort-experimental/fixed/ort-wasm-simd-threaded.asyncify.mjs',
      wasmUrl: '/ort-experimental/fixed/ort-wasm-simd-threaded.asyncify.wasm',
    });
    try {
      assert.equal(ortRuntimeArtifactKindForCurrentThread(), 'asyncify');
    } finally {
      setRequestedOrtWasmArtifactForCurrentThread(null);
    }
  });
});

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
