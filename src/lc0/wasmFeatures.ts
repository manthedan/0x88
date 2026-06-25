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
  //   (i32x4.relaxed_dot_i8x16_i7x16_add
  //     (v128.const i32x4 0 0 0 0)
  //     (v128.const i32x4 0 0 0 0)
  //     (v128.const i32x4 0 0 0 0))))
  // This probes the exact relaxed dot-product opcode used by the Stockfish.js
  // lite-single relaxed artifact rather than a generic relaxed SIMD opcode.
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0, 10, 61, 1, 59, 0,
  253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  253, 147, 2, 11,
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
