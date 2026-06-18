// Lightweight WebAssembly feature probes, kept local to avoid a runtime package
// dependency while matching the wasm-feature-detect style of validating tiny
// feature-specific modules.

const WASM_SIMD_PROBE = new Uint8Array([
  // (module (func (result v128) i32.const 0 i8x16.splat))
  0, 97, 115, 109, 1, 0, 0, 0, 1,
  5, 1, 96, 0, 1, 123, 3, 2, 1,
  0, 10, 8, 1, 6, 0, 65, 0, 253,
  15, 11,
]);

const WASM_RELAXED_SIMD_PROBE = new Uint8Array([
  // (module (func (result v128)
  //   (f32x4.relaxed_madd
  //     (f32x4.splat 1) (f32x4.splat 2) (f32x4.splat 3))))
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96,
  0, 1, 123, 3, 2, 1, 0, 10, 28, 1, 26, 0,
  67, 0, 0, 128, 63, 253, 19, 67, 0, 0, 0, 64,
  253, 19, 67, 0, 0, 64, 64, 253, 19, 253, 133,
  2, 11,
]);

function validateProbe(bytes: Uint8Array): boolean {
  if (typeof WebAssembly === 'undefined' || typeof WebAssembly.validate !== 'function') return false;
  try {
    return WebAssembly.validate(bytes as unknown as BufferSource);
  } catch {
    return false;
  }
}

export function supportsWasmSimd(): boolean {
  return validateProbe(WASM_SIMD_PROBE);
}

export function supportsWasmRelaxedSimd(): boolean {
  return validateProbe(WASM_RELAXED_SIMD_PROBE);
}
