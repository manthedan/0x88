import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { downloadOrtOutputs, ortTensorData, releaseOrtGpuOutputs } from '../src/nn/ortRuntime.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

/** Mirrors onnxruntime-common's Tensor location/getData/dispose contract. */
function gpuTensor(values, log = [], name = 'out', { failDownload = false } = {}) {
  let cpu;
  let downloading = false;
  const tensor = {
    location: 'gpu-buffer',
    type: 'float32',
    get data() {
      if (!cpu) throw new Error('The data is not on CPU. Use `getData()` to download GPU data to CPU');
      return cpu;
    },
    async getData(releaseData) {
      if (tensor.location !== 'gpu-buffer') return cpu;
      downloading = true;
      await Promise.resolve();
      downloading = false;
      if (failDownload) throw new Error(`download ${name} failed`);
      cpu = new Float32Array(values);
      tensor.location = 'cpu';
      if (releaseData) log.push(`release ${name}`);
      return cpu;
    },
    dispose() {
      if (downloading) throw new Error('The current tensor is being downloaded.');
      if (tensor.location === 'gpu-buffer') log.push(`release ${name}`);
      cpu = undefined;
      tensor.location = 'none';
    },
  };
  return tensor;
}

function cpuTensor(values) {
  const data = new Float32Array(values);
  return {
    location: 'cpu',
    type: 'float32',
    data,
    getData: async () => data,
    dispose() {
      throw new Error('CPU outputs must not be disposed by the readback helpers');
    },
  };
}

test('ortTensorData downloads a GPU-located output and releases its buffer', async () => {
  const log = [];
  const tensor = gpuTensor([1, 2], log);
  assert.deepEqual(Array.from(await ortTensorData(tensor)), [1, 2]);
  assert.deepEqual(log, ['release out']);
  const cpu = cpuTensor([3]);
  assert.equal(await ortTensorData(cpu), cpu.data);
});

test('downloadOrtOutputs makes .data readable for every output and releases each GPU buffer once', async () => {
  const log = [];
  const outputs = { policy: gpuTensor([0.5], log, 'policy'), wdl: gpuTensor([1, 0, 0], log, 'wdl'), cpu: cpuTensor([7]) };
  await downloadOrtOutputs(outputs);
  assert.deepEqual(Array.from(outputs.policy.data), [0.5]);
  assert.deepEqual(Array.from(outputs.wdl.data), [1, 0, 0]);
  assert.deepEqual(log.sort(), ['release policy', 'release wdl']);
});

test('downloadOrtOutputs releases the remaining GPU outputs when a download fails', async () => {
  const log = [];
  const outputs = { a: gpuTensor([1], log, 'a', { failDownload: true }), b: gpuTensor([2], log, 'b') };
  await assert.rejects(downloadOrtOutputs(outputs), /download a failed/);
  assert.deepEqual(log.sort(), ['release a', 'release b']);
});

test('releaseOrtGpuOutputs frees unread GPU outputs and leaves read and CPU outputs alone', async () => {
  const log = [];
  const read = gpuTensor([1], log, 'read');
  await ortTensorData(read);
  releaseOrtGpuOutputs({ read, unread: gpuTensor([2], log, 'unread'), cpu: cpuTensor([3]) });
  assert.deepEqual(log, ['release read', 'release unread']);
});

test('SquareFormer evaluator reads GPU-located outputs before its session can be released', async () => {
  const log = [];
  let finishRun;
  const session = {
    run() {
      log.push('run');
      return new Promise((resolve) => {
        finishRun = () => resolve({ policy: gpuTensor(new Float32Array(20480), log, 'policy'), wdl: gpuTensor([0, 0, 0], log, 'wdl') });
      });
    },
    release() {
      log.push('session release');
    },
  };
  const evaluator = new SquareFormerEvaluator(session, {
    kind: 'squareformer_v2',
    input_dim: 112,
    token_features: 24,
    input_mode: 'embedding',
    input_format: 'compact_uint8_embeddings',
    policy_size: 20480,
    history_plies: 7,
    board_normalization: 'stm_white_rankflip_v1',
  });
  const evaluation = evaluator.evaluate(parseFen(START_FEN));
  while (!finishRun) await new Promise((resolve) => setTimeout(resolve, 0));
  evaluator.destroy();
  finishRun();
  assert.deepEqual((await evaluation).wdl, [1 / 3, 1 / 3, 1 / 3]);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(log, ['run', 'release policy', 'release wdl', 'session release']);
});
