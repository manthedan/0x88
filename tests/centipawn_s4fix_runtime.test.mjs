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
    const child = spawnSync(process.execPath, ['--experimental-transform-types', '--no-warnings', '--test-reporter=tap', fileURLToPath(import.meta.url)], {
      encoding: 'utf8',
      env,
    });
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
    let compiledFactory;
    t.mock.method(SquareformerTvmjsWebgpuEvaluator, 'create', (...args) =>
      compiledFactory ? compiledFactory('tvmjs', args) : assert.fail('old TVMJS graph was loaded'),
    );
    t.mock.method(SquareformerTvmHybridEvaluator, 'create', (...args) =>
      compiledFactory ? compiledFactory('hybrid', args) : assert.fail('old hybrid graph was loaded'),
    );
    let verified = true;
    t.mock.method(SquareFormerEvaluator, 'create', async (_url, _meta, settings, loadOptions) => {
      assert.deepEqual(settings, { policyTemperature: 2, nativeHistory: true });
      assert.equal(loadOptions.verifiedCentipawn, verified);
      return evaluator;
    });
    for (const runtime of ['auto', 'ort', 'tvmjs', 'custom-webgpu']) {
      const loaded = await createBrowserSquareformerRuntimeEvaluator(
        {
          modelId: BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
          onnx: '/models/s4fix_v_ema.onnx',
          meta: '/models/s4fix_v_ema.meta.json',
          manifestUrl: CENTIPAWN_TVMJS_WEBGPU_V2.artifact.manifestUrl,
        },
        { runtime, fallback: false },
      );
      assert.equal(loaded.modelId, S4FIX_V_EMA_MODEL_ID, 'model URL takes precedence over stale model id');
      assert.equal(loaded.evaluator, evaluator);
      assert.match(loaded.resolvedRuntime, /ort$/);
      if (runtime === 'tvmjs' || runtime === 'custom-webgpu') assert.match(loaded.fallbackReason, /ORT-only/);
      else assert.equal(loaded.fallbackReason, undefined);
    }
    const aliased = await createBrowserSquareformerRuntimeEvaluator(
      {
        modelId: BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
        onnx: '/models/%734fix_v_ema.onnx?v=1',
        meta: '/models/custom.meta.json',
      },
      { runtime: 'ort' },
    );
    assert.equal(aliased.modelId, S4FIX_V_EMA_MODEL_ID, 'the normalized artifact identity takes precedence over unrelated metadata or stale model IDs');
    assert.equal(
      (readFileSync(new URL('../src/nn/browserRuntimeEvaluator.ts', import.meta.url), 'utf8').match(/SquareFormerEvaluator\.create\s*\(/g) ?? []).length,
      1,
      'all direct and fallback ORT routes must share the verified factory',
    );
    verified = false;
    const research = await createBrowserSquareformerRuntimeEvaluator(
      {
        modelId: S4FIX_V_EMA_MODEL_ID,
        onnx: '/models/custom.onnx',
        meta: '/models/s4fix_v_ema.meta.json',
      },
      { runtime: 'ort' },
    );
    assert.equal(research.evaluator, evaluator, 'default metadata does not make an explicit research ONNX the pinned artifact');
    const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    let deviceDestroys = 0;
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          requestAdapter: async () => ({
            requestDevice: async () => ({
              destroy() {
                deviceDestroys++;
              },
            }),
          }),
        },
      },
    });
    t.after(() => (previousNavigator ? Object.defineProperty(globalThis, 'navigator', previousNavigator) : delete globalThis.navigator));
    for (const runtime of ['tvmjs', 'custom-webgpu']) {
      const controller = new AbortController();
      let compiledCalls = 0;
      compiledFactory = (kind, args) => {
        compiledCalls++;
        assert.equal(kind === 'tvmjs' ? args[2] : args[2].signal, controller.signal);
        controller.abort();
        throw controller.signal.reason;
      };
      await assert.rejects(
        createBrowserSquareformerRuntimeEvaluator(
          { modelId: BT4_SOAP_REM_C19000_FINAL_MODEL_ID, onnx: '/models/bt4_soap_rem_c19000_final.onnx', meta: '/models/custom.meta.json' },
          { runtime, signal: controller.signal, manifestUrl: '/runtimes/centipawn/manifest.json', fallback: true },
        ),
        { name: 'AbortError' },
      );
      assert.equal(compiledCalls, 1, 'the compiled factory receives cancellation instead of an ORT fallback');
    }
    assert.equal(deviceDestroys, 1, 'custom-device cleanup is retained when initialization is canceled');
  });
}
