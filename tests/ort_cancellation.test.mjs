import assert from 'node:assert/strict';
import test from 'node:test';
import { awaitEngineRetirements } from '../src/lc0/engineRetirement.ts';
import { encodeTensorProto, ONNX_FLOAT16, ONNX_INT8, ProtoWriter } from '../src/nn/onnxDequantFold.ts';
import * as ort from '../src/nn/ortRuntime.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

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
  const creating = ort.createOrtSession(new Uint8Array([1]), { signal: controller.signal });
  await started.promise;
  controller.abort();
  await assert.rejects(creating, { name: 'AbortError' });
  assert.equal(calls, 0);
  probe.resolve(null);
  await new Promise(setImmediate);
  assert.equal(calls, 0, 'a late adapter result must not revive the canceled caller');
});

test('research SquareFormer URL ingestion receives cancellation even with dequant folding disabled', async (t) => {
  configure(t);
  const started = Promise.withResolvers(),
    controller = new AbortController();
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    started.resolve();
    assert.equal(options.signal, controller.signal);
    return new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true }));
  });
  t.mock.method(ort.InferenceSession, 'create', () => assert.fail('canceled research download must not allocate'));
  const creating = SquareFormerEvaluator.create('https://research.test/custom.onnx', {}, {}, { signal: controller.signal });
  await started.promise;
  controller.abort();
  await assert.rejects(creating, { name: 'AbortError' });
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
  const node = new ProtoWriter().stringField(1, 'Wq').stringField(1, 'Ws').stringField(2, 'W').stringField(4, 'DequantizeLinear').finish();
  const graph = new ProtoWriter()
    .bytesField(1, node)
    .bytesField(5, encodeTensorProto('Wq', ONNX_INT8, [2], Uint8Array.of(1, 2)))
    .bytesField(5, encodeTensorProto('Ws', ONNX_FLOAT16, [], Uint8Array.of(0, 56)))
    .finish();
  const model = new ProtoWriter().bytesField(7, graph).finish().buffer;
  t.mock.method(ort.InferenceSession, 'create', async () => {
    calls++;
    controller.abort();
    throw new Error('folded session failed');
  });
  await assert.rejects(ort.createOrtSession(model, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(calls, 1);
});
