import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// The custom-WebGPU block modules this evaluator imports use TypeScript
// constructor parameter properties, which `--experimental-strip-types` (the
// flag `npm test` runs with) rejects. Re-run this file under
// `--experimental-transform-types` instead of rewriting those GPU modules.
if (!process.execArgv.includes('--experimental-transform-types')) {
  test('custom WebGPU hybrid evaluator lifecycle (re-run under --experimental-transform-types)', () => {
    const env = { ...process.env };
    delete env.NODE_TEST_CONTEXT;
    const child = spawnSync(process.execPath, ['--experimental-transform-types', '--no-warnings', '--test-reporter=tap', fileURLToPath(import.meta.url)], {
      encoding: 'utf8',
      env,
    });
    assert.equal(child.status, 0, `${child.stdout}\n${child.stderr}`);
    assert.match(child.stdout, /^# pass 7$/m);
    assert.match(child.stdout, /^# fail 0$/m);
  });
} else {
  const { parseFen, START_FEN } = await import('../src/chess/board.ts');
  const { disposeEvaluator, RuntimeFallbackEvaluator } = await import('../src/nn/runtimeFallbackEvaluator.ts');
  const { createBrowserSquareformerRuntimeEvaluator } = await import('../src/nn/browserRuntimeEvaluator.ts');
  const { THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES } = await import('../src/nn/squareformerEvaluator.ts');
  const { SquareformerTvmHybridEvaluator } = await import('../src/nn/squareformerTvmHybridEvaluator.ts');
  const { SquareformerTvmFullLayerWebgpuBlock } = await import('../src/nn/squareformerTvmFullLayerWebgpu.ts');
  const { SquareformerTvmInputStemWebgpuBlock } = await import('../src/nn/squareformerTvmInputStemWebgpu.ts');
  const { SquareformerTvmPolicyHeadWebgpuBlock } = await import('../src/nn/squareformerTvmPolicyHeadWebgpu.ts');
  const { SquareformerTvmValueHeadWebgpuBlock } = await import('../src/nn/squareformerTvmValueHeadWebgpu.ts');
  const { SquareformerTvmLegalPolicyGatherWebgpuBlock } = await import('../src/nn/squareformerTvmLegalPolicyGatherWebgpu.ts');
  const { readFileSync } = await import('node:fs');

  const META = {
    kind: 'squareformer_v2',
    input_dim: 112,
    token_features: 24,
    input_mode: 'embedding',
    input_format: 'compact_uint8_tokens',
    policy_size: 20480,
    history_plies: 7,
    board_normalization: 'stm_white_rankflip_v1',
    attack_summary_schema: 'threatgraph_square_summary_v1',
    attack_summary_feature_count: THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES,
  };

  /** Fake GPU device and blocks: readOutput() resolves only when the test lets it. */
  function fakeHybrid() {
    const state = { submits: 0, biasUploads: 0, deviceDestroys: 0, activeReadbacks: 0, maxActiveReadbacks: 0, destroyedWhileActive: 0, pending: [] };
    const destroyed = new Set();
    let loseDevice;
    const device = {
      queue: {
        submit() {
          state.submits += 1;
        },
      },
      createCommandEncoder: () => ({ finish: () => ({}) }),
      lost: new Promise((resolve) => {
        loseDevice = resolve;
      }),
      destroy() {
        state.deviceDestroys += 1;
        if (state.activeReadbacks) state.destroyedWhileActive += 1;
      },
    };
    const block = (name, extra = {}) => ({
      name,
      encode: (_encoder, input) => input ?? { name },
      upload() {},
      uploadIndices() {},
      copyOutputToReadback() {},
      destroy() {
        destroyed.add(name);
        if (state.activeReadbacks) state.destroyedWhileActive += 1;
      },
      ...extra,
    });
    const readback = (value) => () => {
      state.activeReadbacks += 1;
      state.maxActiveReadbacks = Math.max(state.maxActiveReadbacks, state.activeReadbacks);
      return new Promise((resolve) => {
        state.pending.push(() => {
          state.activeReadbacks -= 1;
          resolve(value());
        });
      });
    };
    const layers = [0, 1].map((i) =>
      block(`layer${i}`, {
        uploadAttentionBias() {
          state.biasUploads += 1;
        },
      }),
    );
    const legal = block('legal', { maxLegalCount: 256, readOutput: () => Promise.resolve(new Float32Array(256)) });
    const value = block('value', { readOutput: readback(() => ({ wdl: new Float32Array([0, 0, 0]), q: new Float32Array([0]) })) });
    const evaluator = new SquareformerTvmHybridEvaluator(device, META, block('stem'), layers, block('policy'), value, legal);
    const settleNext = async () => {
      while (!state.pending.length) await new Promise((resolve) => setTimeout(resolve, 0));
      state.pending.shift()();
    };
    return { evaluator, state, destroyed, settleNext, loseDevice };
  }

  const encoded = () => [new Uint32Array(64), new Float32Array(64 * THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES), new Uint32Array([1, 2, 3])];
  const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

  test('custom WebGPU evaluator serialises evaluations and never re-uploads the attention bias', async () => {
    const { evaluator, state, settleNext } = fakeHybrid();
    const board = parseFen(START_FEN);
    const runs = [evaluator.evaluate(board), evaluator.evaluateEncoded(...encoded()), evaluator.evaluateBatch([board, board])];
    for (let i = 0; i < 4; i++) await settleNext();
    const [single, raw, batch] = await Promise.all(runs);
    assert.deepEqual(single.wdl, [1 / 3, 1 / 3, 1 / 3]);
    assert.equal(raw.legalPolicy.length, 256);
    assert.equal(batch.length, 2);
    assert.equal(state.submits, 4);
    assert.equal(state.maxActiveReadbacks, 1);
    assert.equal(state.biasUploads, 0, 'the static attention bias is uploaded once in create(), not per evaluation');
  });

  test('custom WebGPU evaluator destroys its blocks and owned device only after in-flight work settles', async () => {
    const { evaluator, state, destroyed, settleNext } = fakeHybrid();
    const running = evaluator.evaluateEncoded(...encoded());
    const queued = evaluator.evaluateEncoded(...encoded());
    while (!state.pending.length) await tick();
    evaluator.destroy();
    evaluator.destroy();
    await tick();
    assert.equal(destroyed.size, 0);
    assert.equal(state.deviceDestroys, 0);
    await settleNext();
    assert.equal((await running).wdlLogits.length, 3);
    await assert.rejects(queued, /Custom WebGPU evaluator is destroyed/);
    await tick();
    assert.deepEqual([...destroyed].sort(), ['layer0', 'layer1', 'legal', 'policy', 'stem', 'value']);
    assert.equal(state.deviceDestroys, 1);
    assert.equal(state.destroyedWhileActive, 0);
    assert.equal(state.submits, 1, 'an evaluation queued behind destroy() never reaches the GPU');
    await assert.rejects(evaluator.evaluateEncoded(...encoded()), /destroyed/);
  });

  test('custom WebGPU evaluator surfaces device loss as an evaluation error', async () => {
    const { evaluator, state, settleNext, loseDevice } = fakeHybrid();
    const running = evaluator.evaluateEncoded(...encoded());
    while (!state.pending.length) await tick();
    loseDevice({ message: 'GPU hung' });
    await tick();
    await settleNext();
    await assert.rejects(running, /Custom WebGPU runtime failed: GPU hung/);
    await assert.rejects(evaluator.evaluateEncoded(...encoded()), /Custom WebGPU runtime is unavailable: GPU hung/);
    assert.equal(state.submits, 1);
  });

  test('a lost custom WebGPU device switches the wrapping fallback evaluator to ORT', async (t) => {
    const { evaluator, state, loseDevice } = fakeHybrid();
    const audits = [];
    const warnings = [];
    t.mock.method(console, 'info', (_tag, detail) => audits.push(detail));
    t.mock.method(console, 'warn', (message) => warnings.push(message));
    let fallbackCreates = 0;
    const wrapped = new RuntimeFallbackEvaluator(
      evaluator,
      async () => {
        fallbackCreates += 1;
        return { evaluate: async () => ({ policy: new Map(), wdl: [0, 1, 0] }) };
      },
      { source: 'test', family: 'centipawn', requestedRuntime: 'custom-webgpu' },
      'custom-webgpu-fallback-ort',
    );
    loseDevice({ message: 'GPU reset' });
    await tick();
    assert.deepEqual((await wrapped.evaluate(parseFen(START_FEN))).wdl, [0, 1, 0]);
    assert.equal(fallbackCreates, 1);
    assert.equal(state.submits, 0);
    assert.equal(audits.at(-1).resolvedRuntime, 'custom-webgpu-fallback-ort');
    assert.match(audits.at(-1).fallbackReason, /GPU reset/);
    assert.match(warnings[0], /^Custom WebGPU runtime failed/);
    await tick();
    assert.equal(state.deviceDestroys, 1, 'the fallback destroys the failed primary and its device');
  });

  test('disposing the custom WebGPU evaluator resolves only after its queue drains and the device is destroyed', async () => {
    const { evaluator, state, settleNext } = fakeHybrid();
    const running = evaluator.evaluateEncoded(...encoded());
    while (!state.pending.length) await tick();
    let disposed = false;
    const disposal = disposeEvaluator(evaluator).then(() => {
      disposed = true;
    });
    await tick();
    assert.equal(disposed, false, 'a retirement barrier must keep holding while the device is live');
    await settleNext();
    await running;
    await disposal;
    assert.equal(state.deviceDestroys, 1);
  });

  test('cancelling a custom WebGPU load aborts its stalled artifact fetch and releases the acquired device', async (t) => {
    const fetches = [];
    t.mock.method(globalThis, 'fetch', (url, init = {}) => {
      if (String(url).endsWith('.meta.json')) return Promise.resolve({ ok: true, json: async () => META });
      fetches.push(String(url));
      // A stalled request: it only ever settles through its abort signal, like fetch().
      return new Promise((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(init.signal.reason), { once: true }));
    });
    let deviceDestroys = 0;
    const device = {
      destroy() {
        deviceDestroys += 1;
      },
    };
    Object.defineProperty(globalThis.navigator, 'gpu', {
      configurable: true,
      value: { requestAdapter: async () => ({ requestDevice: async () => device }) },
    });
    t.after(() => delete globalThis.navigator.gpu);
    const controller = new AbortController();
    const loading = createBrowserSquareformerRuntimeEvaluator(
      { modelId: 'custom-test', onnx: '/models/custom-test.onnx', meta: '/models/custom-test.meta.json' },
      { runtime: 'custom-webgpu', manifestUrl: '/runtimes/custom-test/manifest.json', signal: controller.signal, fallback: true },
    );
    while (!fetches.length) await tick();
    assert.match(fetches[0], /manifest\.json$/);
    controller.abort();
    const settled = await Promise.race([
      loading.then(
        () => 'loaded',
        (error) => error,
      ),
      new Promise((resolve) => setTimeout(() => resolve('stalled'), 200)),
    ]);
    assert.equal(settled?.name, 'AbortError', 'the owner signal ends the stalled fetch instead of leaving the load pending');
    assert.equal(deviceDestroys, 1, 'the device acquired before cancellation is released');
  });

  test('cancelling a custom WebGPU load during shader compilation returns no evaluator and releases the device', async (t) => {
    const evaluatorSource = readFileSync(new URL('../src/nn/squareformerTvmHybridEvaluator.ts', import.meta.url), 'utf8');
    const arrays = Object.fromEntries([...evaluatorSource.matchAll(/arrays\.(\w+)\.file/g)].map(([, key]) => [key, { file: `${key}.bin` }]));
    const kernelRows = [
      'attn_qkv_:bias',
      'ffn_dense1_gelu_:bias-gelu',
      'ffn_dense2_residual_:bias-residual',
      'attn_out_residual_:bias-residual',
      'policy_pair_proj_:bias',
      'policy_proj_:bias',
    ].map((row) => {
      const [label, epilogue] = row.split(':');
      return { label, epilogue, file: `${label}.wgsl` };
    });
    const files = {
      '/runtimes/custom-test/manifest.json': { schema: 'centipawn.squareformer-tvm-hybrid.v1' },
      '/k/summary.json': kernelRows,
      '/f/summary.json': { layers: [{ path: 'l0', summary: 'summary.json' }] },
      '/f/l0/summary.json': { layer: 0, shape: { rows: 64, dModel: 4, dFf: 8 }, arrays },
    };
    t.mock.method(globalThis, 'fetch', async (url) => {
      if (String(url).endsWith('.meta.json')) return { ok: true, json: async () => META };
      const body = files[url] ? new TextEncoder().encode(JSON.stringify(files[url])).buffer : new ArrayBuffer(4);
      return { ok: true, arrayBuffer: async () => body };
    });
    const blocks = [];
    const fakeBlock = (name) => {
      const block = { name, destroyed: false, uploadAttentionBias() {}, destroy: () => (block.destroyed = true) };
      blocks.push(block);
      return block;
    };
    let compileHead;
    t.mock.method(SquareformerTvmFullLayerWebgpuBlock, 'create', async () => fakeBlock('layer'));
    t.mock.method(SquareformerTvmInputStemWebgpuBlock, 'create', async () => fakeBlock('stem'));
    // Pipeline compilation cannot be aborted: the policy head finishes compiling after cancel.
    t.mock.method(SquareformerTvmPolicyHeadWebgpuBlock, 'create', () => new Promise((resolve) => (compileHead = () => resolve(fakeBlock('policy')))));
    t.mock.method(SquareformerTvmValueHeadWebgpuBlock, 'create', async () => fakeBlock('value'));
    t.mock.method(SquareformerTvmLegalPolicyGatherWebgpuBlock, 'createDynamic', async () => fakeBlock('legal'));
    let deviceDestroys = 0;
    const device = {
      destroy() {
        deviceDestroys += 1;
      },
    };
    Object.defineProperty(globalThis.navigator, 'gpu', {
      configurable: true,
      value: { requestAdapter: async () => ({ requestDevice: async () => device }) },
    });
    t.after(() => delete globalThis.navigator.gpu);
    const controller = new AbortController();
    const loading = createBrowserSquareformerRuntimeEvaluator(
      { modelId: 'custom-test', onnx: '/models/custom-test.onnx', meta: '/models/custom-test.meta.json' },
      {
        runtime: 'custom-webgpu',
        manifestUrl: '/runtimes/custom-test/manifest.json',
        kernelBase: '/k',
        fixtureRoot: '/f',
        signal: controller.signal,
        fallback: true,
      },
    );
    while (!compileHead) await tick();
    controller.abort();
    compileHead();
    const settled = await loading.then(
      (evaluator) => evaluator,
      (error) => error,
    );
    assert.equal(settled?.name, 'AbortError', 'a cancelled owner never receives an evaluator');
    assert.equal(deviceDestroys, 1, 'the device, and every block allocated on it, is released');
  });
}
