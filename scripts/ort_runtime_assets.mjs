// Mirrors the runtime constants in src/nn/ortRuntime.ts (kept in sync by
// tests/ort_runtime_assets.test.mjs).
//
// The asyncify pair is the WebGPU/JSEP entrypoint: instrumented for stack
// unwinding so wasm can suspend across GPU work (~24MB). The plain pair is the
// CPU-only entrypoint used by executionProviders:['wasm'] sessions (~13.5MB).
// Both glue modules double as their own Emscripten pthread bootstrap.
export const ORT_PTHREAD_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.asyncify.mjs';
export const ORT_PTHREAD_WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';
export const ORT_WASM_EP_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.mjs';
export const ORT_WASM_EP_WASM_FILE = 'ort-wasm-simd-threaded.wasm';

/** Every staged glue module must carry the Emscripten pthread bootstrap. */
export const ORT_PTHREAD_BOOTSTRAP_FILES = Object.freeze([ORT_PTHREAD_BOOTSTRAP_FILE, ORT_WASM_EP_BOOTSTRAP_FILE]);

/** Both staged wasm binaries; each must exist exactly once under /ort/. */
export const ORT_RUNTIME_WASM_FILES = Object.freeze([ORT_PTHREAD_WASM_FILE, ORT_WASM_EP_WASM_FILE]);

export const ORT_RUNTIME_ASSET_FILES = Object.freeze([ORT_PTHREAD_BOOTSTRAP_FILE, ORT_PTHREAD_WASM_FILE, ORT_WASM_EP_BOOTSTRAP_FILE, ORT_WASM_EP_WASM_FILE]);

const ortRuntimeAssetFiles = new Set(ORT_RUNTIME_ASSET_FILES);

export function isRequiredOrtRuntimeAsset(filename) {
  return ortRuntimeAssetFiles.has(filename);
}

export function uncompressedOrtRuntimeAsset(filename) {
  return filename.endsWith('.br') ? filename.slice(0, -3) : filename.endsWith('.gz') ? filename.slice(0, -3) : filename;
}
