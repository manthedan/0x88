import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import {
  collectOrtRuntimeDiagnostics,
  env,
  getOrtWebGpuDiagnosticsSnapshot,
  setOrtRuntimeDiagnosticOptionsForCurrentThread,
  subtractOrtWebGpuDiagnosticsSnapshot,
} from '../src/nn/ortRuntime.ts';

function assertNoProfile(profile, status) {
  assert.equal(profile.status, status);
  assert.equal(profile.eventCount, 0);
  assert.equal(profile.kernelGpuMsTotal, null);
  assert.equal(profile.topPrograms, null);
  assert.match(profile.reason, status === 'disabled' ? /disabled/ : /unavailable on native WebGPU EP/);
  assert.deepEqual(JSON.parse(JSON.stringify(profile)), profile);
}

function emit(programName, durationNs) {
  const data = {
    version: 1,
    inputsMetadata: [],
    outputsMetadata: [],
    kernelId: 1,
    kernelType: 'MatMul',
    kernelName: 'test-kernel',
    programName,
    startTime: 1_000_000,
    endTime: 1_000_000 + durationNs,
  };
  env.webgpu.profiling.ondata(data);
  return data;
}

test('ORT kernel diagnostics distinguish disabled, unavailable intervals, and real JSEP events', async (t) => {
  const originalProfiling = env.webgpu.profiling;
  env.webgpu.profiling ??= {};
  const originalCallback = env.webgpu.profiling.ondata;
  const chainedEvents = [];
  env.webgpu.profiling.ondata = (data) => chainedEvents.push(data);
  try {
    await t.test('disabled snapshots and deltas carry no GPU cost', () => {
      setOrtRuntimeDiagnosticOptionsForCurrentThread({ webgpuProfiling: false });
      const snapshot = getOrtWebGpuDiagnosticsSnapshot();
      assert.equal(snapshot.profiling.enabled, false);
      assertNoProfile(snapshot.profiling, 'disabled');
      assertNoProfile(subtractOrtWebGpuDiagnosticsSnapshot(snapshot, snapshot).profiling, 'disabled');
    });

    setOrtRuntimeDiagnosticOptionsForCurrentThread({ webgpuProfiling: true });
    const diagnostics = await collectOrtRuntimeDiagnostics();
    const before = diagnostics.webgpuDiagnostics;
    await t.test('requested profiling without events is unavailable, independently of GPU API activity', () => {
      assert.equal(before.profiling.enabled, true);
      assert.equal(env.webgpu.profiling.mode, 'default');
      assertNoProfile(before.profiling, 'unavailable');
      const after = { ...before, api: { ...before.api, submitCount: 4, mapAsyncCount: 2, mapAsyncMsTotal: 1.25 } };
      const delta = subtractOrtWebGpuDiagnosticsSnapshot(after, before);
      assertNoProfile(delta.profiling, 'unavailable');
      assert.equal(delta.api.submitCount, 4);
      assert.equal(delta.api.mapAsyncCount, 2);
      assert.equal(delta.api.mapAsyncMsTotal, 1.25);
    });

    await t.test('first and subsequent JSEP events retain totals, sorting, units, and callback chaining', () => {
      const firstData = emit('MatMul', 2_000_000);
      emit('Add', 1_000_000);
      const after = getOrtWebGpuDiagnosticsSnapshot();
      assert.deepEqual(after.profiling, {
        enabled: true,
        status: 'available',
        eventCount: 2,
        kernelGpuMsTotal: 3,
        topPrograms: [
          { programName: 'MatMul', count: 1, gpuMs: 2 },
          { programName: 'Add', count: 1, gpuMs: 1 },
        ],
      });
      assert.deepEqual(subtractOrtWebGpuDiagnosticsSnapshot(after, before).profiling, after.profiling);
      assert.equal(chainedEvents[0], firstData);
      assert.equal(chainedEvents.length, 2);
      emit('MatMul', 500_000);
      assert.deepEqual(subtractOrtWebGpuDiagnosticsSnapshot(getOrtWebGpuDiagnosticsSnapshot(), after).profiling, {
        enabled: true,
        status: 'available',
        eventCount: 1,
        kernelGpuMsTotal: 0.5,
        topPrograms: [{ programName: 'MatMul', count: 1, gpuMs: 0.5 }],
      });
    });

    await t.test('an empty interval is unavailable even after earlier events; measured zero remains available', () => {
      const snapshot = getOrtWebGpuDiagnosticsSnapshot();
      assertNoProfile(subtractOrtWebGpuDiagnosticsSnapshot(snapshot, snapshot).profiling, 'unavailable');
      emit('ZeroDuration', 0);
      const delta = subtractOrtWebGpuDiagnosticsSnapshot(getOrtWebGpuDiagnosticsSnapshot(), snapshot);
      assert.equal(delta.profiling.status, 'available');
      assert.equal(delta.profiling.kernelGpuMsTotal, 0);
      assert.deepEqual(delta.profiling.topPrograms, [{ programName: 'ZeroDuration', count: 1, gpuMs: 0 }]);
    });

    await t.test('evaluator timing propagates unavailable and real-event states without changing evaluation outputs', async () => {
      let withEvents = false;
      const evaluator = new Lc0OnnxEvaluator({
        inputMetadata: [{ name: '/input/planes', type: 'float32', shape: [1, 112, 8, 8] }],
        async run() {
          if (withEvents) emit('MatMul', 1_500_000);
          return {
            '/output/policy': { type: 'float32', data: new Float32Array(1858) },
            '/output/wdl': { type: 'float32', data: new Float32Array([0.25, 0.5, 0.25]) },
          };
        },
      });
      const unavailable = await evaluator.evaluate(START_FEN);
      assert.equal(unavailable.timing.ortKernelProfileStatus, 'unavailable');
      assert.match(unavailable.timing.ortKernelProfileReason, /unavailable on native WebGPU EP/);
      assert.equal(unavailable.timing.ortKernelCount, 0);
      assert.equal(unavailable.timing.ortKernelGpuMs, null);
      withEvents = true;
      const available = await evaluator.evaluate(START_FEN);
      assert.equal(available.timing.ortKernelProfileStatus, 'available');
      assert.equal(available.timing.ortKernelProfileReason, undefined);
      assert.equal(available.timing.ortKernelCount, 1);
      assert.equal(available.timing.ortKernelGpuMs, 1.5);
      const { timing: _unavailableTiming, ...unavailableOutput } = unavailable;
      const { timing: _availableTiming, ...availableOutput } = available;
      assert.deepEqual(availableOutput, unavailableOutput);
      setOrtRuntimeDiagnosticOptionsForCurrentThread({ webgpuProfiling: false });
      const disabled = await evaluator.evaluate(START_FEN);
      // The callback stays installed, as before; test an interval with no events.
      withEvents = false;
      const disabledNoEvents = await evaluator.evaluate(START_FEN);
      assert.equal(disabled.timing.ortKernelProfileStatus, 'available');
      assert.equal(disabledNoEvents.timing.ortKernelProfileStatus, 'disabled');
      assert.equal(disabledNoEvents.timing.ortKernelGpuMs, null);
    });
  } finally {
    env.webgpu.profiling.ondata = originalCallback;
    env.webgpu.profiling = originalProfiling;
    setOrtRuntimeDiagnosticOptionsForCurrentThread(null);
  }
});
