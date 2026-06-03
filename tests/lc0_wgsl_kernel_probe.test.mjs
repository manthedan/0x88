import assert from 'node:assert/strict';
import { test } from 'node:test';
import { f16BitsToF32 } from '../src/lc0/wgslMatmulAddProbe.ts';

test('f16BitsToF32 decodes representative IEEE half values used by lc0web kernels', () => {
  assert.equal(f16BitsToF32(0x0000), 0);
  assert.equal(Object.is(f16BitsToF32(0x8000), -0), true);
  assert.equal(f16BitsToF32(0x3c00), 1);
  assert.equal(f16BitsToF32(0xc000), -2);
  assert.equal(f16BitsToF32(0x3800), 0.5);
  assert.equal(f16BitsToF32(0x7c00), Infinity);
  assert.equal(f16BitsToF32(0xfc00), -Infinity);
  assert.ok(Number.isNaN(f16BitsToF32(0x7e00)));
  assert.ok(Math.abs(f16BitsToF32(0x0001) - 5.960464477539063e-8) < 1e-15);
});
