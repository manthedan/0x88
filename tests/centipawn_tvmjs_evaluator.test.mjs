import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { SquareformerTvmjsWebgpuEvaluator } from '../src/nn/squareformerTvmjsWebgpuEvaluator.ts';

class FakeTensor {
  constructor(shape, dtype, bytes = new Uint8Array(shape.reduce((product, value) => product * value, 1) * 4), sync = async () => {}) {
    this.shape = shape;
    this.dtype = dtype;
    this.bytes = bytes;
    this.device = { sync };
  }
  copyFromRawBytes(bytes) {
    this.bytes = bytes.slice();
  }
  copyFrom(tensor) {
    this.bytes = tensor.bytes.slice();
  }
  toRawBytes() {
    return this.bytes;
  }
}

function bytesOf(values) {
  return new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
}

function fakeRuntime(sync = async () => {}) {
  const state = { invokes: 0, destroyed: false, ended: 0, inputShapes: [] };
  const policy = new FakeTensor([16, 20480], 'float32', bytesOf(new Float32Array(16 * 20480)));
  const wdl = new FakeTensor([16, 3], 'float32', bytesOf(new Float32Array(16 * 3)), sync);
  const functions = {
    set_input(_name, tokens, attack) {
      state.inputShapes.push([tokens.shape, tokens.dtype, attack.shape, attack.dtype]);
    },
    invoke_stateful() {
      state.invokes += 1;
    },
    get_output(_name, index) {
      return index === 0 ? policy : wdl;
    },
  };
  const runtime = {
    webgpu: () => ({}),
    cpu: () => ({}),
    beginScope() {},
    endScope() {
      state.ended += 1;
    },
    empty: (shape, dtype) => new FakeTensor(shape, dtype),
    scalar: (value) => value,
    dispose() {
      state.destroyed = true;
    },
  };
  const vmModule = { getFunction: (name) => functions[name] };
  return { runtime, vmModule, state };
}

const meta = {
  kind: 'squareformer_v2',
  input_dim: 112,
  token_features: 24,
  input_mode: 'embedding',
  input_format: 'compact_uint8_embeddings',
  policy_size: 20480,
  history_plies: 7,
  board_normalization: 'stm_white_rankflip_v1',
  attack_summary_feature_count: 28,
  attack_summary_schema: 'threatgraph_square_summary_v1',
};

function replaceGlobal(name, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  return () => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  };
}

test('Centipawn TVMJS evaluator pads and chunks fixed batch 16 requests', async () => {
  const { runtime, vmModule, state } = fakeRuntime();
  const device = {
    destroy() {
      state.deviceDestroyed = true;
    },
  };
  const evaluator = new SquareformerTvmjsWebgpuEvaluator(runtime, device, vmModule, meta, 16);
  const boards = Array.from({ length: 17 }, () => parseFen(START_FEN));
  const results = await evaluator.evaluateBatch(boards);
  assert.equal(results.length, 17);
  assert.equal(state.invokes, 2);
  assert.deepEqual(state.inputShapes, [
    [[16, 64, 24], 'int32', [16, 64, 28], 'float32'],
    [[16, 64, 24], 'int32', [16, 64, 28], 'float32'],
  ]);
  for (const result of results) {
    assert.ok(result.policy.size > 0);
    assert.ok(Math.abs([...result.policy.values()].reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
    assert.deepEqual(result.wdl, [1 / 3, 1 / 3, 1 / 3]);
  }
  evaluator.destroy();
  await Promise.resolve();
  assert.equal(state.destroyed, true);
  assert.equal(state.deviceDestroyed, true);
});

test('Centipawn TVMJS evaluator defers disposal until an in-flight batch finishes', async () => {
  let releaseSync;
  let syncStarted;
  const started = new Promise((resolve) => {
    syncStarted = resolve;
  });
  const blocked = new Promise((resolve) => {
    releaseSync = resolve;
  });
  const { runtime, vmModule, state } = fakeRuntime(async () => {
    syncStarted();
    await blocked;
  });
  const evaluator = new SquareformerTvmjsWebgpuEvaluator(
    runtime,
    {
      destroy() {
        state.deviceDestroyed = true;
      },
    },
    vmModule,
    meta,
    16,
  );
  const evaluation = evaluator.evaluate(parseFen(START_FEN));
  await started;
  evaluator.destroy();
  assert.equal(state.destroyed, false);
  releaseSync();
  await evaluation;
  await Promise.resolve();
  assert.equal(state.destroyed, true);
  assert.equal(state.deviceDestroyed, true);
});

test('Centipawn TVMJS evaluator rejects untrusted manifests before fetching', async () => {
  const restoreLocation = replaceGlobal('location', new URL('https://app.example/analysis'));
  let fetched = false;
  const restoreFetch = replaceGlobal('fetch', async () => {
    fetched = true;
    throw new Error('unexpected fetch');
  });
  try {
    await assert.rejects(SquareformerTvmjsWebgpuEvaluator.create('https://evil.example/manifest.json', meta), /Untrusted TVMJS manifest URL/);
    assert.equal(fetched, false);
  } finally {
    restoreFetch();
    restoreLocation();
  }
});

test('Centipawn TVMJS evaluator destroys its device when model fetch fails', async () => {
  const restoreLocation = replaceGlobal('location', new URL('https://app.example/analysis'));
  let fetchCount = 0;
  const restoreFetch = replaceGlobal('fetch', async () => {
    fetchCount += 1;
    if (fetchCount === 1) {
      return {
        ok: true,
        async json() {
          return {
            schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
            modelFamily: 'bt4-soap-rem-c19000-final',
            dtype: 'f32',
            target: 'webgpu',
            requiredFeatures: ['webgpu'],
            runtime: { tvmjsBundle: 'tvmjs.bundle.js' },
            models: [{ batch: 16, wasm: 'model.wasm' }],
            files: [{ path: 'tvmjs.bundle.js', bytes: 1, sha256: '0'.repeat(64) }],
          };
        },
      };
    }
    return { ok: false, status: 404 };
  });
  let deviceDestroyed = false;
  const restoreNavigator = replaceGlobal('navigator', {
    gpu: {
      async requestAdapter() {
        return {
          features: { has: () => false },
          limits: {},
          async requestDevice() {
            return {
              destroy() {
                deviceDestroyed = true;
              },
            };
          },
        };
      },
    },
  });
  const restoreTvmjs = replaceGlobal('tvmjs', {
    createPolyfillWASI() {
      return {};
    },
    async instantiate() {
      throw new Error('instantiate should not run');
    },
  });
  const restoreBundleSha = replaceGlobal('__LC0_TVMJS_BUNDLE_SHA256__', '0'.repeat(64));
  try {
    await assert.rejects(SquareformerTvmjsWebgpuEvaluator.create('/runtimes/centipawn/manifest.json', meta), /Fetch failed 404/);
    assert.equal(deviceDestroyed, true);
  } finally {
    restoreBundleSha();
    restoreTvmjs();
    restoreNavigator();
    restoreFetch();
    restoreLocation();
  }
});
