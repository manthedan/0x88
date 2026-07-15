export const ORT_PTHREAD_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.asyncify.mjs';
export const ORT_PTHREAD_WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';

export const ORT_RUNTIME_ASSET_FILES = Object.freeze([
  ORT_PTHREAD_BOOTSTRAP_FILE,
  ORT_PTHREAD_WASM_FILE,
]);

const ortRuntimeAssetFiles = new Set(ORT_RUNTIME_ASSET_FILES);

export function isRequiredOrtRuntimeAsset(filename) {
  return ortRuntimeAssetFiles.has(filename);
}

export function uncompressedOrtRuntimeAsset(filename) {
  return filename.endsWith('.br') ? filename.slice(0, -3) : filename.endsWith('.gz') ? filename.slice(0, -3) : filename;
}
