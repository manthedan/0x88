// Mirrors the runtime constants in src/nn/ortRuntime.ts (kept in sync by
// tests/ort_fallback_policy.test.mjs and tests/ort_runtime_assets.test.mjs).
//
// Two GPU pairs and one CPU pair. Asyncify (~24MB) is the WebGPU fallback when
// JSPI is unavailable. JSPI (~15MB) is the production WebGPU pair when
// WebAssembly.Suspending and WebAssembly.promising exist. The plain pair is
// the CPU-only entrypoint used by executionProviders:['wasm'] (~13.5MB).
// Each glue module is its own Emscripten pthread bootstrap. A publish that
// omits the JSPI pair 404s production GPU sessions; selection does not fall
// back to asyncify on a missing file.
export const ORT_PTHREAD_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.asyncify.mjs';
export const ORT_PTHREAD_WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';
export const ORT_JSPI_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.jspi.mjs';
export const ORT_JSPI_WASM_FILE = 'ort-wasm-simd-threaded.jspi.wasm';
export const ORT_WASM_EP_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.mjs';
export const ORT_WASM_EP_WASM_FILE = 'ort-wasm-simd-threaded.wasm';

/** Every staged glue module must carry the Emscripten pthread bootstrap. */
export const ORT_PTHREAD_BOOTSTRAP_FILES = Object.freeze([ORT_PTHREAD_BOOTSTRAP_FILE, ORT_JSPI_BOOTSTRAP_FILE, ORT_WASM_EP_BOOTSTRAP_FILE]);

/** Staged wasm binaries; each must exist exactly once under /ort/. */
export const ORT_RUNTIME_WASM_FILES = Object.freeze([ORT_PTHREAD_WASM_FILE, ORT_JSPI_WASM_FILE, ORT_WASM_EP_WASM_FILE]);

export const ORT_RUNTIME_ASSET_FILES = Object.freeze([
  ORT_PTHREAD_BOOTSTRAP_FILE,
  ORT_PTHREAD_WASM_FILE,
  ORT_JSPI_BOOTSTRAP_FILE,
  ORT_JSPI_WASM_FILE,
  ORT_WASM_EP_BOOTSTRAP_FILE,
  ORT_WASM_EP_WASM_FILE,
]);

const ortRuntimeAssetFiles = new Set(ORT_RUNTIME_ASSET_FILES);

export function isRequiredOrtRuntimeAsset(filename) {
  return ortRuntimeAssetFiles.has(filename);
}

export function uncompressedOrtRuntimeAsset(filename) {
  return filename.endsWith('.br') ? filename.slice(0, -3) : filename.endsWith('.gz') ? filename.slice(0, -3) : filename;
}
