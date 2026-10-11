import assert from 'node:assert/strict';
import test from 'node:test';
import { roundedBackendTimingRecord, roundedNumericRecord } from '../src/lc0/timingReport.ts';

test('backend timing reports retain native profiler unavailability through JSON without numeric zero samples', () => {
  const timing = {
    backend: 'ort-onnx',
    ortRunMs: 1.234567,
    ortKernelCount: 0,
    ortKernelGpuMs: null,
    ortKernelProfileStatus: 'unavailable',
    ortKernelProfileReason: 'unavailable on native WebGPU EP',
  };
  assert.deepEqual(JSON.parse(JSON.stringify(roundedBackendTimingRecord(timing))), {
    ortRunMs: 1.2346,
    ortKernelCount: 0,
    ortKernelGpuMs: null,
    ortKernelProfileStatus: 'unavailable',
    ortKernelProfileReason: 'unavailable on native WebGPU EP',
  });
  assert.deepEqual(roundedNumericRecord(timing), { ortRunMs: 1.2346, ortKernelCount: 0 });
});

test('backend timing reports preserve disabled and real-event profiles, including measured zero', () => {
  assert.deepEqual(roundedBackendTimingRecord({ ortKernelProfileStatus: 'disabled', ortKernelGpuMs: null }), {
    ortKernelProfileStatus: 'disabled',
    ortKernelGpuMs: null,
  });
  for (const gpuMs of [2.123456, 0]) {
    const timing = { ortKernelProfileStatus: 'available', ortKernelCount: 1, ortKernelGpuMs: gpuMs };
    assert.deepEqual(roundedBackendTimingRecord(timing), { ...timing, ortKernelGpuMs: Number(gpuMs.toFixed(4)) });
    assert.deepEqual(roundedNumericRecord(timing), { ortKernelCount: 1, ortKernelGpuMs: Number(gpuMs.toFixed(4)) });
  }
});

test('timing rounding still rejects non-finite, nonnumeric, and missing inputs', () => {
  assert.equal(roundedBackendTimingRecord(undefined), undefined);
  assert.equal(roundedBackendTimingRecord({ backend: 'hybrid', bad: NaN, infinite: Infinity }), undefined);
  assert.deepEqual(roundedNumericRecord({ cost: 1.234567, text: '3', missing: null }, 2), { cost: 1.23 });
});
