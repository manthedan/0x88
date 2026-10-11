import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { SquareformerTvmjsWebgpuEvaluator } from '../src/nn/squareformerTvmjsWebgpuEvaluator.ts';
import { encodingBatch, encodingFens, legacyPaddedInputs } from './fixtures/centipawn_encoding.mjs';

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
  const state = { invokes: 0, destroyed: false, ended: 0, inputShapes: [], inputBytes: [] };
  const policy = new FakeTensor([16, 20480], 'float32', bytesOf(new Float32Array(16 * 20480)));
  const wdl = new FakeTensor([16, 3], 'float32', bytesOf(new Float32Array(16 * 3)), sync);
  const functions = {
    set_input(_name, tokens, attack) {
      state.inputShapes.push([tokens.shape, tokens.dtype, attack.shape, attack.dtype]);
      state.inputBytes.push({ tokens: tokens.bytes.slice(), attack: attack.bytes.slice() });
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

for (const size of [1, 4, 8, 16]) {
  test(`Centipawn TVMJS b${size} uploaded bytes match the original padded encoding`, async () => {
    for (const normalization of ['stm_white_rankflip_v1', undefined]) {
      for (const masked of [false, true]) {
        // Rotate all positions through the last real row, which supplies padding.
        for (let offset = 0; offset < encodingFens.length; offset++) {
          const { boards, contexts } = encodingBatch(size, offset, masked);
          const currentMeta = { ...meta, board_normalization: normalization };
          const expected = legacyPaddedInputs(boards, contexts, currentMeta);
          const { runtime, vmModule, state } = fakeRuntime();
          const evaluator = new SquareformerTvmjsWebgpuEvaluator(runtime, { destroy() {} }, vmModule, currentMeta, 16);
          assert.equal((await evaluator.evaluateBatch(boards, contexts)).length, size);
          assert.equal(state.inputBytes.length, 1);
          assert.deepEqual(state.inputBytes[0].tokens, expected.tokens, `tokens: ${normalization}/${masked}/${offset}`);
          assert.deepEqual(state.inputBytes[0].attack, expected.attack, `attack: ${normalization}/${masked}/${offset}`);
          evaluator.destroy();
        }
      }
    }
  });
}

test('Centipawn TVMJS padding copies already-masked rows without reading masks again', async () => {
  const { boards, contexts } = encodingBatch(4, 1);
  const reads = Array(boards.length).fill(0);
  contexts.forEach((context, index) => {
    context.attackSummaryChannelMask = new Proxy(Array(28).fill(0.3), {
      get(target, key) {
        if (typeof key === 'string' && /^\d+$/.test(key)) reads[index] += 1;
        return Reflect.get(target, key);
      },
    });
  });
  const { runtime, vmModule } = fakeRuntime();
  const evaluator = new SquareformerTvmjsWebgpuEvaluator(runtime, { destroy() {} }, vmModule, meta, 16);
  await evaluator.evaluateBatch(boards, contexts);
  assert.deepEqual(reads, Array(4).fill(64 * 28));
  evaluator.destroy();
});

test("Centipawn TVMJS padding uses each chunk's final context and rejects invalid masks before invoking", async () => {
  const { boards, contexts } = encodingBatch(17, 1, true);
  const { runtime, vmModule, state } = fakeRuntime();
  const evaluator = new SquareformerTvmjsWebgpuEvaluator(runtime, { destroy() {} }, vmModule, meta, 16);
  await evaluator.evaluateBatch(boards, contexts);
  assert.deepEqual(state.inputBytes, [
    legacyPaddedInputs(boards.slice(0, 16), contexts.slice(0, 16), meta),
    legacyPaddedInputs(boards.slice(16), contexts.slice(16), meta),
  ]);
  await assert.rejects(evaluator.evaluate(boards[0], { attackSummaryChannelMask: [1] }), /attackSummaryChannelMask length 1 does not match 28/);
  assert.equal(state.invokes, 2);
  assert.deepEqual(await evaluator.evaluateBatch([]), []);
  assert.equal(state.invokes, 2);
  evaluator.destroy();
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

test('Centipawn TVMJS model creates share hash-keyed modelCache bytes without IDB', async () => {
  const bytes = new TextEncoder().encode('abc');
  const sha256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
  const restoreLocation = replaceGlobal('location', new URL('https://app.example/analysis'));
  const responses = new Map();
  const cache = {
    match: async (request) => responses.get(request.url)?.clone(),
    put: async (request, response) => responses.set(request.url, response),
    keys: async () => [...responses.keys()].map((url) => new Request(url)),
  };
  const restoreCaches = replaceGlobal('caches', { open: async () => cache });
  // Without IDB the shared loader still uses Cache Storage, but rehashes.
  const restoreIdb = replaceGlobal('indexedDB', undefined);
  let bodyFetches = 0;
  const restoreFetch = replaceGlobal('fetch', async (input) => {
    const url = typeof input === 'string' ? input : (input.url ?? String(input));
    if (url.endsWith('manifest.json')) {
      return new Response(
        JSON.stringify({
          schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
          modelFamily: 'bt4-soap-rem-c19000-final',
          dtype: 'f32',
          target: 'webgpu',
          runtime: { tvmjsBundle: 'tvmjs.bundle.js' },
          models: [{ batch: 16, wasm: 'model.wasm', bytes: 3, sha256 }],
          files: [{ path: 'tvmjs.bundle.js', bytes: 1, sha256: '0'.repeat(64) }],
        }),
      );
    }
    bodyFetches += 1;
    return new Response(bytes);
  });
  const restoreNavigator = replaceGlobal('navigator', {
    gpu: {
      requestAdapter: async () => ({
        features: { has: () => false },
        limits: {},
        requestDevice: async () => ({ destroy() {} }),
      }),
    },
  });
  const restoreTvmjs = replaceGlobal('tvmjs', {
    createPolyfillWASI: () => ({}),
    instantiate: async () => ({
      initWebGPU() {},
      beginScope() {},
      systemLib() {},
      asyncLoadWebGPUPipelines: async () => {},
      webgpu() {},
      createVirtualMachine: () => ({ getInternalModule: () => ({ getFunction() {} }) }),
      dispose() {},
    }),
  });
  const restoreBundleSha = replaceGlobal('__LC0_TVMJS_BUNDLE_SHA256__', '0'.repeat(64));
  try {
    const first = await SquareformerTvmjsWebgpuEvaluator.create('/runtimes/centipawn/manifest.json', meta);
    const second = await SquareformerTvmjsWebgpuEvaluator.create('/runtimes/centipawn/manifest.json', meta);
    first.destroy();
    second.destroy();
    assert.equal(bodyFetches, 1);
    assert.match([...responses.keys()][0], new RegExp(`sha256=${sha256}`));
  } finally {
    restoreBundleSha();
    restoreTvmjs();
    restoreNavigator();
    restoreFetch();
    restoreIdb();
    restoreCaches();
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
