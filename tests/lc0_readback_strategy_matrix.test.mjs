import assert from 'node:assert/strict';
import test from 'node:test';
import { numericStats } from '../scripts/lc0_browser_readback_strategy_matrix.mjs';

test('readback matrix numeric stats handle one million samples', () => {
  const values = Array.from({ length: 1_000_000 }, (_, index) => index + 1);
  const stats = numericStats(values);

  assert.equal(stats.samples, values.length);
  assert.equal(stats.mean, 500_000.5);
  assert.equal(stats.median, 500_000.5);
  assert.equal(stats.min, 1);
  assert.equal(stats.max, 1_000_000);
  assert.ok(Number.isFinite(stats.variance));
  assert.ok(Number.isFinite(stats.standardDeviation));
  assert.ok(Number.isFinite(stats.coefficientOfVariation));
});

test('readback matrix numeric stats preserve signed zero extrema', () => {
  const stats = numericStats([0, -0]);

  assert.equal(Object.is(stats.min, -0), true);
  assert.equal(Object.is(stats.max, 0), true);
});
