import test from 'node:test';
import assert from 'node:assert/strict';
import { softmax } from '../src/nn/numerics.ts';

function legacyStudentSoftmax(xs) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / total);
}

function legacyOnnxSoftmax(xs) {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  const out = Array.from(xs, (x) => Math.exp(Number(x) - m));
  const total = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((x) => x / total);
}

function legacySquareFormerSoftmax(xs) {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (Number(xs[i]) > m) m = Number(xs[i]);
  const out = Array.from(xs, (x) => Math.exp(Number(x) - m));
  const total = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((x) => x / total);
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

test('shared softmax matches legacy evaluator implementations bit-for-bit', () => {
  const rand = rng(0x51f7_1ee1);
  const cases = [
    [-1000, -1000, -1000],
    [0],
    [-20, 0, 20],
  ];
  for (let i = 0; i < 100; i++) {
    const n = 1 + Math.floor(rand() * 80);
    cases.push(Array.from({ length: n }, () => (rand() - 0.5) * 80));
  }

  for (const xs of cases) {
    const expected = legacyOnnxSoftmax(xs);
    assert.deepEqual(legacyStudentSoftmax(xs), expected, `student legacy drift for ${JSON.stringify(xs)}`);
    assert.deepEqual(legacySquareFormerSoftmax(xs), expected, `squareformer legacy drift for ${JSON.stringify(xs)}`);
    assert.deepEqual(softmax(xs), expected, `shared number[] drift for ${JSON.stringify(xs)}`);
    assert.deepEqual(softmax(Float32Array.from(xs)), legacyOnnxSoftmax(Float32Array.from(xs)), `shared Float32Array drift for ${JSON.stringify(xs)}`);
  }
});
