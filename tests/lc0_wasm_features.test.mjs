import assert from 'node:assert/strict';
import test from 'node:test';
import { supportsWasmRelaxedSimd, supportsWasmRelaxedSimdIntegerDot, supportsWasmSimd } from '../src/lc0/wasmFeatures.ts';

test('LC0 WebAssembly feature probes return capability booleans', () => {
  assert.equal(typeof supportsWasmSimd(), 'boolean');
  assert.equal(typeof supportsWasmRelaxedSimd(), 'boolean');
  assert.equal(typeof supportsWasmRelaxedSimdIntegerDot(), 'boolean');
});

test('relaxed integer dot support implies the required SIMD features', () => {
  if (!supportsWasmRelaxedSimdIntegerDot()) return;
  assert.equal(supportsWasmSimd(), true);
  assert.equal(supportsWasmRelaxedSimd(), true);
});
