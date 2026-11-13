import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { OrtGpuIoBinding } from '../src/nn/ortGpuIo.ts';

const input = { name: 'input', type: 'float32', dims: [1] };
const outputs = [{ name: 'output', type: 'float32', dims: [1] }, { name: 'other', type: 'float32', dims: [1] }];
function fakeGpu({ failAllocation = -1, failPin = -1, failWrap = -1, failTensorDispose = false } = {}) {
  const buffers = [], tensors = [];
  let wraps = 0, uploads = 0;
  const device = {
    createBuffer({ size }) {
      if (buffers.length === failAllocation) throw new Error('injected allocation failure');
      const buffer = {
        bytes: new ArrayBuffer(size), destroys: 0,
        async mapAsync() {}, getMappedRange() { return this.bytes; }, unmap() {},
        destroy() { this.destroys++; },
      };
      if (buffers.length === failPin) Object.preventExtensions(buffer);
      buffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() { return { copyBufferToBuffer() {}, finish() { return {}; } }; },
    queue: { writeBuffer() { uploads++; }, submit() {} },
  };
  const Tensor = {
    fromGpuBuffer(buffer) {
      if (wraps++ === failWrap) throw new Error('injected tensor-wrap failure');
      const tensor = {
        buffer, disposes: 0,
        dispose() { this.disposes++; if (failTensorDispose) throw new Error('injected cleanup failure'); },
      };
      tensors.push(tensor);
      return tensor;
    },
  };
  const session = { async run(_feeds, fetches) { return fetches; } };
  return { buffers, tensors, device, Tensor, session, uploads: () => uploads };
}

function evaluatorWithGpu(run = async (_feeds, fetches) => fetches, release = async () => {}) {
  const gpu = fakeGpu();
  const session = {
    inputMetadata: [{ name: '/input/planes', type: 'float32', shape: [1, 112, 8, 8] }],
    run, release,
  };
  const binding = new OrtGpuIoBinding(session, gpu.device, gpu.Tensor,
    { name: '/input/planes', type: 'float32', dims: [1, 112, 8, 8] },
    [{ name: '/output/policy', type: 'float32', dims: [1, 1858] }, { name: '/output/wdl', type: 'float32', dims: [1, 3] }],
    { pin: false });
  const evaluator = new Lc0OnnxEvaluator(session, {}, { binding, mode: 'gpu' });
  return { ...gpu, binding, evaluator };
}

function assertRolledBack(gpu) {
  assert.ok(gpu.buffers.every((buffer) => buffer.destroys === 1), 'every acquired caller buffer destroyed once');
  assert.ok(gpu.tensors.every((tensor) => tensor.disposes === 1), 'every wrapped tensor disposed once');
}

for (const outputMode of ['preallocated', 'session']) {
  const allocations = outputMode === 'session' ? 2 : 4;
  const wraps = outputMode === 'session' ? 1 : 3;
  for (let failAllocation = 0; failAllocation < allocations; failAllocation++) {
    test(`ORT ${outputMode} construction rolls back allocation failure at ${failAllocation}`, () => {
      const gpu = fakeGpu({ failAllocation });
      assert.throws(() => new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, input, outputs, { outputMode, pin: false }), /allocation failure/);
      assert.equal(gpu.buffers.length, failAllocation);
      assertRolledBack(gpu);
    });
  }
  for (let failPin = 0; failPin < wraps; failPin++) {
    test(`ORT ${outputMode} construction rolls back pin failure at ${failPin}`, () => {
      const gpu = fakeGpu({ failPin });
      assert.throws(() => new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, input, outputs, { outputMode }), TypeError);
      assert.equal(gpu.buffers.length, allocations);
      assertRolledBack(gpu);
    });
  }
  for (let failWrap = 0; failWrap < wraps; failWrap++) {
    test(`ORT ${outputMode} construction releases resources on tensor-wrap failure at ${failWrap}`, () => {
      const gpu = fakeGpu({ failWrap });
      assert.throws(() => new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, input, outputs, { outputMode, pin: false }), /tensor-wrap failure/);
      assert.equal(gpu.buffers.length, allocations);
      assert.equal(gpu.tensors.length, failWrap);
      assertRolledBack(gpu);
    });
  }
}

test('ORT rollback continues through tensor cleanup errors and preserves the construction failure', () => {
  const gpu = fakeGpu({ failWrap: 2, failTensorDispose: true });
  assert.throws(() => new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, input, outputs, { pin: false }), /tensor-wrap failure/);
  assertRolledBack(gpu);
});

test('ORT evaluator releases its binding even if the session release rejects', async () => {
  const failure = new Error('injected release failure');
  let releases = 0;
  const gpu = evaluatorWithGpu(undefined, async () => { releases++; throw failure; });
  const close = gpu.evaluator.dispose();
  assert.strictEqual(gpu.evaluator.dispose(), close, 'concurrent callers share close completion');
  await assert.rejects(close, (error) => error === failure);
  assert.ok(gpu.buffers.every((buffer) => buffer.destroys === 1));
  assert.strictEqual(gpu.evaluator.dispose(), close, 'later callers still observe the release failure');
  assert.equal(releases, 1);
});

test('ORT concurrent and later dispose calls return one promise and release once', async () => {
  let finishRelease, releases = 0;
  const gpu = evaluatorWithGpu(undefined, () => {
    releases++;
    return new Promise((resolve) => { finishRelease = resolve; });
  });
  const close = gpu.evaluator.dispose();
  assert.strictEqual(gpu.evaluator.dispose(), close);
  await Promise.resolve();
  assert.ok(gpu.buffers.every((buffer) => buffer.destroys === 0));
  finishRelease();
  await close;
  assert.ok(gpu.buffers.every((buffer) => buffer.destroys === 1));
  assert.strictEqual(gpu.evaluator.dispose(), close);
  assert.equal(releases, 1);
});

test('a normal ORT run rejection leaves the evaluator and serialized IO queue usable', async () => {
  let runs = 0;
  const gpu = evaluatorWithGpu(async (_feeds, fetches) => {
    if (++runs === 1) throw new Error('injected run failure');
    return fetches;
  });
  const failed = gpu.evaluator.evaluate(START_FEN);
  const next = gpu.evaluator.evaluate(START_FEN);
  await assert.rejects(failed, /run failure/);
  assert.equal((await next).fen, START_FEN);
  assert.equal((await gpu.evaluator.evaluate(START_FEN)).fen, START_FEN);
  assert.equal(runs, 3);
  assert.equal(gpu.uploads(), 3);
  await gpu.evaluator.dispose();
});
