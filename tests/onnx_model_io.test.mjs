import assert from 'node:assert/strict';
import test from 'node:test';
import { lc0GpuIoSpecs, Lc0OnnxEvaluator, defaultLc0OrtIoMode } from '../src/lc0/onnxEvaluator.ts';
import { readOnnxModelIo } from '../src/lc0/onnxModelIo.ts';
import { ProtoWriter, encodeTensorProto, ONNX_FLOAT16 } from '../src/nn/onnxDequantFold.ts';
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
function model({ input = [1, 112, 8, 8], policy = [1, 1858], outputType = ONNX_FLOAT16, omitWdl = false, initializerInput = false } = {}) {
  const graph = new ProtoWriter().bytesField(11, value('/input/planes', input));
  if (initializerInput) {
    graph.bytesField(5, encodeTensorProto('weights', ONNX_FLOAT16, [1], new Uint8Array(2)));
    graph.bytesField(11, value('weights', [1]));
  }
  graph.bytesField(12, value('/output/policy', policy, outputType));
  if (!omitWdl) graph.bytesField(12, value('/output/wdl', [1, 3]));
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
