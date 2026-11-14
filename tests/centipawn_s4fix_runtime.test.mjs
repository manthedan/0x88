import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// The browser loader imports legacy GPU modules with constructor parameter properties.
if (!process.execArgv.includes('--experimental-transform-types')) {
  test('s4fix runtime routing (transform legacy TypeScript syntax)', () => {
    const env = { ...process.env };
    delete env.NODE_TEST_CONTEXT;
    const child = spawnSync(process.execPath, ['--experimental-transform-types', '--no-warnings', '--test-reporter=tap', fileURLToPath(import.meta.url)], { encoding: 'utf8', env });
    assert.equal(child.status, 0, `${child.stdout}\n${child.stderr}`);
    assert.match(child.stdout, /^# pass 1$/m);
  });
} else {
  const { createBrowserSquareformerRuntimeEvaluator } = await import('../src/nn/browserRuntimeEvaluator.ts');
  const { SquareFormerEvaluator } = await import('../src/nn/squareformerEvaluator.ts');
  const { SquareformerTvmjsWebgpuEvaluator } = await import('../src/nn/squareformerTvmjsWebgpuEvaluator.ts');
  const { SquareformerTvmHybridEvaluator } = await import('../src/nn/squareformerTvmHybridEvaluator.ts');
  const { BT4_SOAP_REM_C19000_FINAL_MODEL_ID, S4FIX_V_EMA_MODEL_ID, CENTIPAWN_TVMJS_WEBGPU_V2 } = await import('../src/nn/runtimeRegistry.ts');

  test('s4fix persisted strict compiled selectors never load an old graph, even with fallback disabled', async (t) => {
    const meta = JSON.parse(readFileSync('public/models/s4fix_v_ema.meta.json'));
    const evaluator = { evaluate: async () => ({ policy: new Map(), wdl: [0, 1, 0] }) };
    t.mock.method(globalThis, 'fetch', async () => ({ ok: true, json: async () => meta }));
    t.mock.method(SquareformerTvmjsWebgpuEvaluator, 'create', () => assert.fail('old TVMJS graph was loaded'));
    t.mock.method(SquareformerTvmHybridEvaluator, 'create', () => assert.fail('old hybrid graph was loaded'));
    t.mock.method(SquareFormerEvaluator, 'create', async (_url, _meta, settings) => {
      assert.deepEqual(settings, { policyTemperature: 2, nativeHistory: true });
      return evaluator;
    });
    for (const runtime of ['auto', 'ort', 'tvmjs', 'custom-webgpu']) {
      const loaded = await createBrowserSquareformerRuntimeEvaluator({
        modelId: BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
        onnx: '/models/s4fix_v_ema.onnx',
        meta: '/models/s4fix_v_ema.meta.json',
        manifestUrl: CENTIPAWN_TVMJS_WEBGPU_V2.artifact.manifestUrl,
      }, { runtime, fallback: false });
      assert.equal(loaded.modelId, S4FIX_V_EMA_MODEL_ID, 'model URL takes precedence over stale model id');
      assert.equal(loaded.evaluator, evaluator);
      assert.match(loaded.resolvedRuntime, /ort$/);
      if (runtime === 'tvmjs' || runtime === 'custom-webgpu') assert.match(loaded.fallbackReason, /ORT-only/);
      else assert.equal(loaded.fallbackReason, undefined);
    }
  });
}
