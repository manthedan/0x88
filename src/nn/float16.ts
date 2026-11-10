// Shared IEEE 754 binary16 conversions. Encoding matches
// DataView.prototype.setFloat16 for every float32 input (round to nearest,
// ties to even; overflow -> +/-Infinity; NaN stays NaN), see tests/float16.test.mjs.

/** f16 bits -> number (exact; subnormals, signed zero, Infinity and NaN preserved). */
export function f16BitsToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/**
 * Round-to-nearest-even f32 -> f16 bits. The input is first rounded to
 * float32, so pass float32 values (e.g. Float32Array elements); a float64 that
 * is not a float32 can double-round differently from setFloat16 near a tie.
 * Finite values beyond the f16 range become +/-Infinity, as IEEE 754 and
 * setFloat16 do, rather than saturating to +/-65504.
 */
export function f32ToF16Bits(x: number): number {
  f32Scratch[0] = x;
  const b = u32Scratch[0];
  const sign = (b >>> 16) & 0x8000;
  const exp = (b >>> 23) & 0xff;
  let mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h += 1;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h += 1; // carry into the exponent is the correct rounding
  return sign | h;
}
