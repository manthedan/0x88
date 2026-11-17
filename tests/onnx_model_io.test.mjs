import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultLc0OrtIoMode, Lc0OnnxEvaluator, lc0GpuIoSpecs } from '../src/lc0/onnxEvaluator.ts';
import { applyOnnxFreeDimensionOverrides, readOnnxModelIo } from '../src/lc0/onnxModelIo.ts';
import { encodeTensorProto, ONNX_FLOAT16, ProtoWriter } from '../src/nn/onnxDequantFold.ts';
import { OrtGpuIoBinding } from '../src/nn/ortGpuIo.ts';
import * as ort from '../src/nn/ortRuntime.ts';

function value(name, shape, type = ONNX_FLOAT16) {
  const tensor = new ProtoWriter().varintField(1, type);
  if (shape) {
    const dims = new ProtoWriter();
    for (const size of shape) {
      const dim = new ProtoWriter();
      if (typeof size === 'number') dim.varintField(1, size);
      else if (typeof size === 'string') dim.stringField(2, size);
      dims.bytesField(1, dim.finish());
    }
    tensor.bytesField(2, dims.finish());
  }
  return new ProtoWriter().stringField(1, name).bytesField(2, new ProtoWriter().bytesField(1, tensor.finish()).finish()).finish();
}
function model({ input = [1, 112, 8, 8], policy = [1, 1858], outputType = ONNX_FLOAT16, omitWdl = false, initializerInput = false, wdl = [1, 3] } = {}) {
  const graph = new ProtoWriter().bytesField(11, value('/input/planes', input));
  if (initializerInput) {
    graph.bytesField(5, encodeTensorProto('weights', ONNX_FLOAT16, [1], new Uint8Array(2)));
    graph.bytesField(11, value('weights', [1]));
  }
  graph.bytesField(12, value('/output/policy', policy, outputType));
  if (!omitWdl) graph.bytesField(12, value('/output/wdl', wdl));
  return new ProtoWriter().varintField(1, 9).bytesField(7, graph.finish()).finish();
}

test('protobuf preflight reads fixed LC0 float IO without parsing initializer data', () => {
  const bytes = model({ initializerInput: true });
  const padded = new Uint8Array(bytes.length + 19);
  padded.set(bytes, 7);
  const io = readOnnxModelIo(padded.subarray(7, 7 + bytes.length));
  assert.deepEqual(io.inputNames, ['/input/planes']);
  assert.deepEqual(lc0GpuIoSpecs(io), {
    input: { name: '/input/planes', type: 'float16', dims: [1, 112, 8, 8] },
    outputs: [
      { name: '/output/policy', type: 'float16', dims: [1, 1858] },
      { name: '/output/wdl', type: 'float16', dims: [1, 3] },
    ],
  });
});

for (const [label, options, reason] of [
  ['dynamic batch', { input: ['batch', 112, 8, 8] }, /dynamic shape/],
  ['unknown dimension', { input: [null, 112, 8, 8] }, /dynamic shape/],
  ['zero dimension', { input: [0, 112, 8, 8] }, /dynamic shape/],
  ['unknown rank', { input: null }, /dynamic shape/],
  ['dynamic output width', { policy: [1, 'policy'] }, /dynamic shape/],
  ['non-float output', { outputType: 7 }, /type tensor\(7\)/],
  ['missing WDL', { omitWdl: true }, /missing policy\/wdl/],
]) {
  test(`preflight rejects ${label} before the only CPU-IO session is created`, async (t) => {
    const bytes = model(options);
    assert.match(lc0GpuIoSpecs(readOnnxModelIo(bytes)).reason, reason);
    ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
    ort.setOrtDequantFoldForCurrentThread(false);
    t.after(() => {
      ort.setRequestedOrtExecutionProviderForCurrentThread(null);
      ort.setOrtDequantFoldForCurrentThread(null);
    });
    t.mock.method(console, 'warn', () => {});
    const creates = [];
    t.mock.method(ort.InferenceSession, 'create', async (input, options) => {
      creates.push({ input, options });
      return { ...readOnnxModelIo(bytes), async release() {} };
    });
    const evaluator = await Lc0OnnxEvaluator.create(bytes.buffer, { ioMode: 'gpu-capture' });
    assert.equal(evaluator.ioMode, 'cpu');
    assert.equal(creates.length, 1, 'no rejected GPU session, second fold, or second load');
    assert.equal(creates[0].options.enableGraphCapture, undefined);
    assert.equal(creates[0].options.preferredOutputLocation, undefined);
    await evaluator.dispose();
  });
}

test('Small retains its gpu-capture default and fixed-shape models still attempt captured IO', async (t) => {
  assert.equal(defaultLc0OrtIoMode(true), 'gpu-capture');
  ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
  ort.setOrtDequantFoldForCurrentThread(false);
  t.after(() => {
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
  });
  t.mock.method(console, 'warn', () => {});
  const calls = [];
  t.mock.method(ort.InferenceSession, 'create', async (_bytes, options) => {
    calls.push(options);
    if (options.enableGraphCapture) throw new Error('injected GPU failure');
    return { ...readOnnxModelIo(model()), async release() {} };
  });
  const small = model({ policy: ['batch', 1858] });
  assert.match(lc0GpuIoSpecs(readOnnxModelIo(small)).reason, /dynamic shape/, 'literal declaration is symbolic');
  assert.equal(lc0GpuIoSpecs(readOnnxModelIo(small), { inferOutputBatch: true }).reason, undefined, 'Small head batch can be inferred');
  const evaluator = await Lc0OnnxEvaluator.create(small, { ioMode: defaultLc0OrtIoMode(true) });
  assert.equal(calls.length, 2);
  assert.equal(calls[0].enableGraphCapture, true);
  assert.equal(calls[0].preferredOutputLocation, 'gpu-buffer');
  assert.equal(calls[1].enableGraphCapture, undefined);
  await evaluator.dispose();
});

for (const [net, input, expected] of [
  ['LQO', ['batch', 112, 8, 8], 'cpu'],
  ['t3', [8, 112, 8, 8], 'gpu-capture'],
  ['BT4', [8, 112, 8, 8], 'gpu-capture'],
]) {
  test(`${net} uses the device-selected IO default with exactly one native session`, async (t) => {
    const bytes = model({ input, policy: [8, 1858] });
    ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
    ort.setOrtDequantFoldForCurrentThread(false);
    const deviceDescriptor = Object.getOwnPropertyDescriptor(ort.env.webgpu, 'device');
    Object.defineProperty(ort.env.webgpu, 'device', { configurable: true, value: { createBuffer: () => ({ destroy() {} }) } });
    t.after(() => {
      ort.setRequestedOrtExecutionProviderForCurrentThread(null);
      ort.setOrtDequantFoldForCurrentThread(null);
      if (deviceDescriptor) Object.defineProperty(ort.env.webgpu, 'device', deviceDescriptor);
      else delete ort.env.webgpu.device;
    });
    t.mock.method(console, 'warn', () => {});
    t.mock.method(ort.Tensor, 'fromGpuBuffer', () => ({ dispose() {} }));
    const warmup = t.mock.method(OrtGpuIoBinding.prototype, 'run', async () => ({}));
    const calls = [];
    t.mock.method(ort.InferenceSession, 'create', async (_input, options) => {
      calls.push(options);
      return { ...readOnnxModelIo(bytes), async release() {} };
    });
    const evaluator = await Lc0OnnxEvaluator.create(bytes, { ioMode: defaultLc0OrtIoMode(true) });
    assert.equal(evaluator.ioMode, expected);
    assert.equal(calls.length, 1, 'LQO must not build then reject a probe; fixed nets must not fall back');
    assert.equal(calls[0].enableGraphCapture, expected === 'gpu-capture' ? true : undefined);
    assert.equal(warmup.mock.callCount(), expected === 'gpu-capture' ? 1 : 0);
    await evaluator.dispose();
  });
}

test('free-dimension resolution uses prototype IO getters, preserves original metadata and validates sizes', () => {
  const io = readOnnxModelIo(model({ input: ['batch', 112, 8, 8], policy: ['batch', 1858] }));
  const source = Object.create({
    get inputNames() {
      return io.inputNames;
    },
    get outputNames() {
      return io.outputNames;
    },
    get inputMetadata() {
      return io.inputMetadata;
    },
    get outputMetadata() {
      return io.outputMetadata;
    },
  });
  const resolved = applyOnnxFreeDimensionOverrides(source, { batch: 1 });
  assert.equal(resolved.inputMetadata[0].shape[0], 1);
  assert.equal(resolved.outputMetadata[0].shape[0], 1);
  assert.equal(io.inputMetadata[0].shape[0], 'batch');
  assert.deepEqual(resolved.inputNames, ['/input/planes']);
  for (const bad of [0, -1, NaN, Infinity, 1.5]) assert.throws(() => applyOnnxFreeDimensionOverrides(source, { batch: bad }), /Invalid ONNX free dimension/);
  assert.match(lc0GpuIoSpecs(applyOnnxFreeDimensionOverrides(io, { other: 1 })).reason, /dynamic shape/);
});

for (const failCapture of [false, true, 'metadata-mismatch']) {
  test(`LQO batch-1 policy: ${failCapture ? 'capture rejection keeps unpinned CPU fallback' : 'one captured session and serial physical b1 for a logical b8 request'}`, async (t) => {
    const bytes = model({ input: ['batch', 112, 8, 8], policy: ['batch', 1858], wdl: ['batch', 3] });
    const io = readOnnxModelIo(bytes);
    const before = bytes.slice();
    ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
    ort.setOrtDequantFoldForCurrentThread(false);
    const device = Object.getOwnPropertyDescriptor(ort.env.webgpu, 'device');
    Object.defineProperty(ort.env.webgpu, 'device', { configurable: true, value: { createBuffer: () => ({ destroy() {} }) } });
    t.after(() => {
      ort.setRequestedOrtExecutionProviderForCurrentThread(null);
      ort.setOrtDequantFoldForCurrentThread(null);
      if (device) Object.defineProperty(ort.env.webgpu, 'device', device);
      else delete ort.env.webgpu.device;
    });
    t.mock.method(console, 'warn', () => {});
    t.mock.method(ort.Tensor, 'fromGpuBuffer', () => ({ dispose() {} }));
    const runs = [],
      creates = [];
    t.mock.method(OrtGpuIoBinding.prototype, 'run', async (input) => {
      runs.push(input.byteLength);
      return { '/output/policy': new Uint16Array(1858), '/output/wdl': new Uint16Array(3) };
    });
    class Session {
      constructor(metadata = io) {
        this.metadata = metadata;
      }
      get inputNames() {
        return this.metadata.inputNames;
      }
      get outputNames() {
        return io.outputNames;
      }
      get inputMetadata() {
        return this.metadata.inputMetadata;
      }
      get outputMetadata() {
        return this.metadata.outputMetadata;
      }
      async release() {}
    }
    t.mock.method(ort.InferenceSession, 'create', async (input, options) => {
      creates.push({ input, options });
      if (failCapture === true && options.enableGraphCapture) throw Error('injected capture failure');
      return new Session(failCapture === 'metadata-mismatch' && options.enableGraphCapture ? applyOnnxFreeDimensionOverrides(io, { batch: 8 }) : io);
    });
    const evaluator = await Lc0OnnxEvaluator.create(bytes, { ioMode: 'gpu-capture', freeDimensionOverrides: { batch: 1 } });
    assert.deepEqual(creates[0].options.freeDimensionOverrides, { batch: 1 });
    assert.equal(creates[0].input, bytes);
    assert.equal(evaluator.ioMode, failCapture ? 'cpu' : 'gpu-capture');
    assert.equal(creates.length, failCapture ? 2 : 1);
    if (failCapture) {
      assert.equal(creates[1].options.freeDimensionOverrides, undefined);
      assert.equal(creates[1].options.enableGraphCapture, undefined);
      assert.equal(evaluator.fixedBatchSize, undefined, 'no pinned shape survives a CPU fallback');
    } else {
      assert.equal(evaluator.fixedBatchSize, 1);
      const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      assert.equal((await evaluator.evaluateBatch(Array(8).fill(fen))).length, 8);
      assert.deepEqual(runs, Array(9).fill(1 * 112 * 8 * 8 * 2), 'one capture warmup and eight physical batch-1 runs');
    }
    assert.deepEqual(bytes, before, 'the shipped model bytes and cache entry are unchanged');
    await evaluator.dispose();
  });
}

test('metadata-owned GPU parse recovery is not swallowed by the CPU-IO fallback', async (t) => {
  ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
  ort.setOrtDequantFoldForCurrentThread(false);
  t.after(() => {
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
  });
  let calls = 0;
  t.mock.method(ort.InferenceSession, 'create', async () => {
    calls++;
    throw new Error('INVALID_PROTOBUF');
  });
  await assert.rejects(Lc0OnnxEvaluator.create(model(), { ioMode: 'gpu-capture', rejectModelParseErrors: true }), /INVALID_PROTOBUF/);
  assert.equal(calls, 1, 'surface to the cache owner before trying another session');
});

test('URL-only GPU requests use one CPU-IO creation without a shape-probe download', async (t) => {
  ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
  ort.setOrtDequantFoldForCurrentThread(false);
  t.after(() => {
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    ort.setOrtDequantFoldForCurrentThread(null);
  });
  t.mock.method(globalThis, 'fetch', () => {
    throw new Error('unexpected shape-probe download');
  });
  const calls = [];
  t.mock.method(ort.InferenceSession, 'create', async (input, options) => {
    calls.push({ input, options });
    return { async release() {} };
  });
  const evaluator = await Lc0OnnxEvaluator.create('https://unit.test/model.onnx', { ioMode: 'gpu-capture' });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].input, 'https://unit.test/model.onnx');
  assert.equal(calls[0].options.enableGraphCapture, undefined);
  await evaluator.dispose();
});

test('preflight refuses missing/truncated protobuf graphs', () => {
  assert.throws(() => readOnnxModelIo(new Uint8Array()), /missing graph/);
  assert.throws(() => readOnnxModelIo(model().subarray(0, 15)), /protobuf/);
});
