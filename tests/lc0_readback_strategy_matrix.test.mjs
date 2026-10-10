import assert from 'node:assert/strict';
import test from 'node:test';
import { compactResult, numericStats, summarize } from '../scripts/lc0_browser_readback_strategy_matrix.mjs';

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

test('readback matrix retains unavailable native kernel profiling in cells and the printed summary', () => {
  const summary = compactResult('ort-gpu-outputs', {
    lastBackendTiming: {
      ortKernelCount: 0,
      ortKernelGpuMs: null,
      ortKernelProfileStatus: 'unavailable',
      ortKernelProfileReason: 'unavailable on native WebGPU EP',
    },
  });
  assert.equal(summary.ortKernelProfileStatus, 'unavailable');
  assert.equal(summary.ortKernelProfileReason, 'unavailable on native WebGPU EP');
  assert.equal(summary.ortKernelGpuMsLast, null);
  const aggregate = summarize([{ strategy: 'ort-gpu-outputs', summary }])['ort-gpu-outputs'];
  assert.deepEqual(aggregate.ortKernelProfileStatuses, ['unavailable']);
  assert.deepEqual(aggregate.ortKernelProfileReasons, ['unavailable on native WebGPU EP']);
});

test('readback matrix retains disabled and real-event kernel profiling without losing measured zero', () => {
  for (const [status, gpuMs] of [['disabled', null], ['available', 1.5], ['available', 0]]) {
    const summary = compactResult('ort-cpu-outputs', {
      lastBackendTiming: { ortKernelProfileStatus: status, ortKernelGpuMs: gpuMs },
    });
    assert.equal(summary.ortKernelProfileStatus, status);
    assert.equal(summary.ortKernelGpuMsLast, gpuMs);
  }
});

test('readback matrix numeric stats preserve signed zero extrema', () => {
  const stats = numericStats([0, -0]);

  assert.equal(Object.is(stats.min, -0), true);
  assert.equal(Object.is(stats.max, 0), true);
});
