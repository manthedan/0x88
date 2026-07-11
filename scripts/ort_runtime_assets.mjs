export const ORT_RUNTIME_ASSET_FILES = Object.freeze([
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]);

const ortRuntimeAssetFiles = new Set(ORT_RUNTIME_ASSET_FILES);

export function isRequiredOrtRuntimeAsset(filename) {
  return ortRuntimeAssetFiles.has(filename);
}

export function uncompressedOrtRuntimeAsset(filename) {
  return filename.endsWith('.br') ? filename.slice(0, -3) : filename.endsWith('.gz') ? filename.slice(0, -3) : filename;
}
