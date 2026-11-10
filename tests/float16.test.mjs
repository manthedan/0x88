import assert from 'node:assert/strict';
import test from 'node:test';
import { f16BitsToF32, f32ToF16Bits } from '../src/nn/float16.ts';
import * as dequantFold from '../src/nn/onnxDequantFold.ts';

const hasNative = typeof DataView.prototype.setFloat16 === 'function';
const view = new DataView(new ArrayBuffer(2));
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

function nativeBits(x) {
  view.setFloat16(0, x, true);
  return view.getUint16(0, true);
}

function nativeValue(bits) {
  view.setUint16(0, bits, true);
  return view.getFloat16(0, true);
}

/** The float32 one ulp above/below `x` (x finite, nonzero magnitude direction). */
function f32Neighbour(x, direction) {
  f32[0] = x;
  u32[0] += x > 0 === direction > 0 ? 1 : -1;
  return f32[0];
}

function assertEncodes(x, label) {
  const got = f32ToF16Bits(x);
  const want = nativeBits(x);
  if (got !== want) assert.fail(`${label}: f32ToF16Bits(${x}) = 0x${got.toString(16)}, setFloat16 gives 0x${want.toString(16)}`);
}

test('f16BitsToF32 matches DataView.getFloat16 for all 65536 bit patterns', { skip: !hasNative && 'no native float16' }, () => {
  for (let h = 0; h < 0x10000; h++) {
    const got = f16BitsToF32(h);
    const want = nativeValue(h);
    if (Number.isNaN(want)) assert.ok(Number.isNaN(got), `0x${h.toString(16)} should decode to NaN`);
    else if (!Object.is(got, want)) assert.fail(`f16BitsToF32(0x${h.toString(16)}) = ${got}, getFloat16 gives ${want}`);
  }
});

test('f32ToF16Bits matches DataView.setFloat16 at every f16 value, rounding midpoint and its neighbours', { skip: !hasNative && 'no native float16' }, () => {
  for (let h = 0; h < 0x10000; h++) {
    const value = f16BitsToF32(h);
    if (!Number.isFinite(value)) continue;
    assertEncodes(value, 'exact');
    // Midpoint to the next larger magnitude (for 0x7bff that is the overflow
    // threshold 65520): exactly representable in float32, so this hits every tie.
    const next = (h & 0x7fff) === 0x7bff ? Math.sign(value || 1) * 65520 : f16BitsToF32(h + 1);
    const mid = (value + next) / 2;
    f32[0] = mid;
    assert.equal(f32[0], mid, `midpoint after 0x${h.toString(16)} is a float32`);
    assertEncodes(mid, 'tie');
    if (mid !== 0) {
      assertEncodes(f32Neighbour(mid, 1), 'just above a tie');
      assertEncodes(f32Neighbour(mid, -1), 'just below a tie');
    }
  }
});

test('f32ToF16Bits matches DataView.setFloat16 on 2M random float32 bit patterns and edge values', { skip: !hasNative && 'no native float16' }, () => {
  let seed = 0x9e3779b9;
  for (let i = 0; i < 2_000_000; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    u32[0] = seed;
    if (!Number.isNaN(f32[0])) assertEncodes(f32[0], 'random');
  }
  for (const x of [0, -0, Infinity, -Infinity, 65504, -65504, 65519.99, 65520, -65520, 1e6, 3.4e38, 2 ** -14, 2 ** -24, 2 ** -25, 1.5 * 2 ** -25, 2 ** -26]) {
    f32[0] = x;
    assertEncodes(f32[0], 'edge');
  }
  assert.equal(f32ToF16Bits(Number.NaN) & 0x7c00, 0x7c00);
  assert.notEqual(f32ToF16Bits(Number.NaN) & 0x3ff, 0, 'NaN stays NaN');
});

test('f16 conversion semantics callers rely on', () => {
  assert.equal(f32ToF16Bits(1 + 2 ** -11), 0x3c00, 'tie rounds to even (down)');
  assert.equal(f32ToF16Bits(1 + 3 * 2 ** -11), 0x3c02, 'tie rounds to even (up)');
  assert.equal(f32ToF16Bits(768.25), 0x6200, 'tie rounds to even, not half-up');
  assert.equal(f32ToF16Bits(65520), 0x7c00, 'overflow is Infinity, not saturated to 65504');
  assert.equal(f32ToF16Bits(-1e6), 0xfc00);
  // LC0 112-plane inputs are 0/1 masks and the integer halfmove clock: all exact in f16.
  for (let n = 0; n <= 2048; n++) assert.equal(f16BitsToF32(f32ToF16Bits(n)), n);
  assert.equal(dequantFold.f32ToF16Bits, f32ToF16Bits, 'onnxDequantFold re-exports the shared encoder');
  assert.equal(dequantFold.f16BitsToF32, f16BitsToF32);
});
