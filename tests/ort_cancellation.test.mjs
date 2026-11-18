import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { awaitEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';
import { untilAborted } from '../src/lc0/modelCache.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { encodeTensorProto, ONNX_FLOAT16, ONNX_INT8, ProtoWriter } from '../src/nn/onnxDequantFold.ts';
import * as ort from '../src/nn/ortRuntime.ts';

function configure(t, provider = 'wasm', fold = false) {
  ort.setRequestedOrtExecutionProviderForCurrentThread(provider);
  ort.setOrtDequantFoldForCurrentThread(fold);
  ort.resetWebGpuAdapterProbeForTests();
  t.after(() => {
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
    ort.resetWebGpuAdapterProbeForTests();
  });
}

function quantizedModel() {
  const node = new ProtoWriter().stringField(1, 'Wq').stringField(1, 'Ws').stringField(2, 'W').stringField(4, 'DequantizeLinear').finish();
  const graph = new ProtoWriter()
    .bytesField(1, node)
    .bytesField(5, encodeTensorProto('Wq', ONNX_INT8, [2], Uint8Array.of(1, 2)))
    .bytesField(5, encodeTensorProto('Ws', ONNX_FLOAT16, [], Uint8Array.of(0, 56)))
    .finish();
  return new ProtoWriter().bytesField(7, graph).finish().buffer;
}

test('canceling a pending WebGPU probe rejects promptly without admitting any native ORT session', async (t) => {
  configure(t, 'auto');
  const probe = Promise.withResolvers(),
    started = Promise.withResolvers(),
    controller = new AbortController();
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        requestAdapter: () => {
          started.resolve();
          return probe.promise;
        },
      },
    },
  });
  t.after(() => (previous ? Object.defineProperty(globalThis, 'navigator', previous) : delete globalThis.navigator));
  let calls = 0;
  t.mock.method(ort.InferenceSession, 'create', () => {
    calls++;
    assert.fail('canceled probe must not allocate');
  });
  const creating = Lc0OnnxEvaluator.create(new Uint8Array([1]), { signal: controller.signal });
  await started.promise;
  controller.abort();
  await assert.rejects(creating, { name: 'AbortError' });
  assert.equal(calls, 0);
  probe.resolve(null);
  await new Promise(setImmediate);
  assert.equal(calls, 0, 'a late adapter result must not revive the canceled caller');
  const source = readFileSync(new URL('../src/lc0/onnxEvaluator.ts', import.meta.url), 'utf8');
  assert.equal(
    (source.match(/ort\.createOrtSession\s*\(/g) ?? []).length,
    (source.match(/ort\.createOrtSession\(\s*modelPath,\s*\{\s*signal:\s*options\.signal/g) ?? []).length,
    'every Lc0 native-session path must carry its initialization owner',
  );
});

test('an already-stopped consumer observes a shared operation rejection without dropping it', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(untilAborted(Promise.reject(new Error('shared retirement failed')), controller.signal), { name: 'AbortError' });
  await new Promise(setImmediate);
});

test('GPU-IO initialization cancellation retires its owned session before device wait completes', async () => {
  const text = readFileSync(new URL('../src/lc0/onnxEvaluator.ts', import.meta.url), 'utf8'),
    start = text.indexOf('async function createLc0GpuIoSession(');
  const body = stripTypeScriptTypes(text.slice(start, text.indexOf('export class Lc0OnnxEvaluator', start)));
  const device = Promise.withResolvers(),
    release = Promise.withResolvers(),
    started = Promise.withResolvers(),
    controller = new AbortController();
  let releases = 0,
    bindings = 0;
  const specs = { input: { name: 'input', type: 'float32', dims: [1] }, outputs: [{ name: 'policy', type: 'float32', dims: [1] }] };
  const deps = {
    untilAborted,
    trackEngineRetirement,
    ort: {
      env: { webgpu: { device: device.promise } },
      createOrtSession: async (_model, options) => {
        assert.equal(options.signal, controller.signal);
        started.resolve();
        return {};
      },
      releaseOrtSession: () => {
        releases++;
        return release.promise;
      },
    },
    lc0GpuIoSpecs: () => specs,
    applyOnnxFreeDimensionOverrides: (value) => value,
    OrtGpuIoBinding: class {
      constructor() {
        bindings++;
        throw new Error('late binding allocation');
      }
    },
  };
  const create = new Function(...Object.keys(deps), `${body};return createLc0GpuIoSession;`)(...Object.values(deps));
  const creating = create(() => new Uint8Array([1]), 'gpu', { signal: controller.signal }, specs);
  const canceled = assert.rejects(creating, { name: 'AbortError' });
  await started.promise;
  controller.abort();
  await new Promise(setImmediate);
  const releasedBeforeDevice = releases === 1;
  let admitted = false;
  const waiting = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  await new Promise(setImmediate);
  const blocked = !admitted;
  device.resolve({});
  release.resolve();
  await canceled;
  await waiting;
  assert.equal(releasedBeforeDevice, true);
  assert.equal(blocked, true);
  assert.equal(releases, 1);
  assert.equal(bindings, 0);
});

test('canceling an admitted native creation still waits for and tracks the actual session release', async (t) => {
  configure(t);
  const native = Promise.withResolvers(),
    started = Promise.withResolvers(),
    release = Promise.withResolvers(),
    controller = new AbortController();
  let releases = 0,
    calls = 0;
  t.mock.method(ort.InferenceSession, 'create', async () => {
    calls++;
    started.resolve();
    return native.promise;
  });
  const creating = ort.createOrtSession(new Uint8Array([1]), { signal: controller.signal });
  const canceled = assert.rejects(creating, { name: 'AbortError' });
  await started.promise;
  controller.abort();
  native.resolve({
    release: () => {
      releases++;
      return release.promise;
    },
  });
  await new Promise(setImmediate);
  let retired = false;
  const waiting = awaitEngineRetirements().then(() => {
    retired = true;
  });
  await new Promise(setImmediate);
  assert.equal(releases, 1);
  assert.equal(retired, false);
  release.resolve();
  await canceled;
  await waiting;
  assert.equal(retired, true);
  assert.equal(calls, 1);
  assert.equal(releases, 1);
});

test('cancellation after a failed folded attempt cannot admit unfolded or provider fallback attempts', async (t) => {
  configure(t, 'wasm', true);
  const controller = new AbortController();
  let calls = 0;
  const model = quantizedModel();
  t.mock.method(ort.InferenceSession, 'create', async () => {
    calls++;
    controller.abort();
    throw new Error('folded session failed');
  });
  await assert.rejects(ort.createOrtSession(model, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(calls, 1);
});
