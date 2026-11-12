import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { createCriticalReport, criticalMomentCacheKey, runCriticalReport } from '../src/lc0/criticalMoments.ts';
import {
  createLc0OnnxEvaluatorFromUrl,
  LC0_SMALL_BATCH1_QDQ8_PATH,
  LC0_SMALL_BATCH1_SPLITK_FILE,
  lc0SmallSplitKUrl,
  selectLc0OrtModelUrl,
} from '../src/lc0/lc0SmallModelSelection.ts';
import { planOrtSessionAttempts, resolveOrtSessionPlacement, setRequestedOrtExecutionProviderForCurrentThread } from '../src/nn/ortRuntime.ts';

const ORIGINAL = LC0_SMALL_BATCH1_QDQ8_PATH;
const SPLITK = `/models/lc0/${LC0_SMALL_BATCH1_SPLITK_FILE}`;
const APPLE_WEBGPU = { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' };

test('split-K URL maps only the shipped QDQ8 file and keeps the prefix', () => {
  assert.equal(lc0SmallSplitKUrl(ORIGINAL), SPLITK);
  assert.equal(lc0SmallSplitKUrl(`https://assets.0x88.app${ORIGINAL}?v=1`), `https://assets.0x88.app/models/lc0/${LC0_SMALL_BATCH1_SPLITK_FILE}?v=1`);
  assert.equal(lc0SmallSplitKUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx'), undefined);
  assert.equal(lc0SmallSplitKUrl('/models/lc0/lqo_v2.f16.qdq8.onnx'), undefined);
  assert.equal(lc0SmallSplitKUrl(SPLITK), undefined);
});

test('selection: Apple + WebGPU gets split-K; non-Apple WebGPU and WASM keep the original', () => {
  assert.equal(selectLc0OrtModelUrl(ORIGINAL, APPLE_WEBGPU), SPLITK);
  assert.equal(selectLc0OrtModelUrl(ORIGINAL, { providers: ['webgpu'], adapterVendor: 'Apple' }), SPLITK);
  for (const adapterVendor of ['nvidia', 'amd', 'intel', 'qualcomm', '', undefined]) {
    assert.equal(selectLc0OrtModelUrl(ORIGINAL, { providers: ['webgpu', 'wasm'], adapterVendor }), ORIGINAL, `vendor ${adapterVendor}`);
  }
  assert.equal(selectLc0OrtModelUrl(ORIGINAL, { providers: ['wasm'], adapterVendor: 'apple' }), ORIGINAL);
  assert.equal(selectLc0OrtModelUrl(ORIGINAL, { providers: ['wasm'] }), ORIGINAL);
  // Other models are never swapped, whatever the device.
  assert.equal(selectLc0OrtModelUrl('/models/lc0/lqo_v2.f16.qdq8.onnx', APPLE_WEBGPU), '/models/lc0/lqo_v2.f16.qdq8.onnx');
});

/** Fake loader/session factory; `sessionFails` makes the WebGPU attempt fail like createOrtSession would. */
function fakeDeps(placement, { sessionFails = false, missing = [] } = {}) {
  const loads = [];
  const sessions = [];
  return {
    loads,
    sessions,
    deps: {
      placement: async () => placement,
      load: async (url) => {
        loads.push(url);
        if (missing.includes(url)) throw new Error(`404 ${url}`);
        return { model: `bytes:${url}`, url, mode: 'memory', cacheStatus: 'miss', elapsedMs: 0, telemetry: { source: 'network', totalMs: 0 } };
      },
      createEvaluator: async (model, options) => {
        // Mirrors createOrtSession: the WASM attempt uses options.wasmFallbackModel when given, else the primary bytes.
        const input = sessionFails ? (options.wasmFallbackModel ? await options.wasmFallbackModel() : model) : model;
        sessions.push({ input, providers: sessionFails ? ['wasm'] : placement.providers });
        return { model: input, dispose: async () => {} };
      },
    },
  };
}

test('Apple + WebGPU loads and runs the split-K model under its own URL', async () => {
  const fake = fakeDeps(APPLE_WEBGPU);
  const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, fake.deps);
  assert.deepEqual(fake.loads, [SPLITK]);
  assert.equal(result.servedUrl, SPLITK);
  assert.equal(result.variant, 'splitk');
  assert.equal(result.modelLoad.url, SPLITK);
  assert.equal(fake.sessions[0].input, `bytes:${SPLITK}`);
});

test('non-Apple WebGPU and WASM sessions load the original model', async () => {
  for (const placement of [{ providers: ['webgpu', 'wasm'], adapterVendor: 'nvidia' }, { providers: ['webgpu', 'wasm'] }, { providers: ['wasm'] }]) {
    const fake = fakeDeps(placement);
    const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, fake.deps);
    assert.deepEqual(fake.loads, [ORIGINAL], JSON.stringify(placement));
    assert.equal(result.servedUrl, ORIGINAL);
    assert.equal(result.variant, 'original');
    assert.equal(fake.sessions[0].input, `bytes:${ORIGINAL}`);
  }
});

test('explicit-URL callers (no deviceSelection) get exactly the URL they named', async () => {
  const fake = fakeDeps(APPLE_WEBGPU);
  const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, {}, fake.deps);
  assert.deepEqual(fake.loads, [ORIGINAL]);
  assert.equal(result.servedUrl, ORIGINAL);
});

test('WebGPU session failure: the WASM fallback loads and runs the ORIGINAL URL', async () => {
  const fake = fakeDeps(APPLE_WEBGPU, { sessionFails: true });
  const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, fake.deps);
  assert.deepEqual(fake.loads, [SPLITK, ORIGINAL]);
  assert.deepEqual(fake.sessions, [{ input: `bytes:${ORIGINAL}`, providers: ['wasm'] }]);
  assert.equal(result.servedUrl, ORIGINAL);
  assert.equal(result.variant, 'original');
  assert.equal(result.modelLoad.url, ORIGINAL);
  assert.match(result.fallbackReason, /WebGPU session failed/);
});

test('a missing split-K artifact (CDN not published yet) falls back to the original', async (t) => {
  t.mock.method(console, 'warn', () => {});
  const fake = fakeDeps(APPLE_WEBGPU, { missing: [SPLITK] });
  const result = await createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, fake.deps);
  assert.deepEqual(fake.loads, [SPLITK, ORIGINAL]);
  assert.equal(result.servedUrl, ORIGINAL);
  assert.match(result.fallbackReason, /split-K model unavailable/);
});

test('an aborted split-K load is not retried with the original', async () => {
  const fake = fakeDeps(APPLE_WEBGPU);
  fake.deps.load = async () => {
    throw new DOMException('cancelled', 'AbortError');
  };
  await assert.rejects(createLc0OnnxEvaluatorFromUrl(ORIGINAL, { deviceSelection: true }, fake.deps), { name: 'AbortError' });
});

test('createOrtSession attempt plan: the WASM attempt uses wasmFallbackModel, not the WebGPU bytes', async () => {
  const prepared = { input: 'folded-splitk', original: 'raw-splitk' };
  const withModel = planOrtSessionAttempts(prepared, ['webgpu', 'wasm'], true, { wasmFallbackModel: async () => 'raw-original' });
  assert.deepEqual(
    withModel.map((a) => a.providers),
    [['webgpu', 'wasm'], ['webgpu', 'wasm'], ['wasm']],
  );
  assert.equal(withModel[0].input, 'folded-splitk');
  assert.equal(withModel[1].input, 'raw-splitk');
  assert.equal(typeof withModel[2].input, 'function');
  assert.equal(await withModel[2].input(), 'raw-original');
  // Without the option the existing behaviour is unchanged: WASM reuses the unfolded primary bytes.
  const plain = planOrtSessionAttempts(prepared, ['webgpu', 'wasm'], true);
  assert.equal(plain[2].input, 'raw-splitk');
  // Strict ep=webgpu has no WASM attempt at all.
  assert.equal(planOrtSessionAttempts(prepared, ['webgpu'], false, { wasmFallbackModel: async () => 'x' }).length, 2);
});

test('resolveOrtSessionPlacement reports the adapter vendor for a WebGPU session', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: { hardwareConcurrency: 8, gpu: { requestAdapter: async () => ({ info: { vendor: 'Apple', architecture: 'metal-3' } }) } },
    configurable: true,
    writable: true,
  });
  setRequestedOrtExecutionProviderForCurrentThread('auto');
  try {
    const placement = await resolveOrtSessionPlacement();
    assert.deepEqual(placement, { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' });
    assert.equal(selectLc0OrtModelUrl(ORIGINAL, placement), SPLITK);
  } finally {
    setRequestedOrtExecutionProviderForCurrentThread(null);
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  }
});

test('critical-moment results are keyed by the model the Lc0 lane actually ran', async () => {
  const provenance = { engine: 'Lc0 Small', model: ORIGINAL, build: 'b', runtime: 'r', budget: '400 visits', options: 'o' };
  const lane = {
    id: 'lc0',
    row: { family: 'lc0', variant: 'small', strength: 400 },
    provenance,
    evaluate: async () => {
      provenance.model = SPLITK;
      return {
        cacheKey: '',
        provenance: { ...provenance },
        line: { pvUci: ['e2e4'], scoreCp: 20, multipv: 1, pvSan: 'e4', scoreText: '+0.20', engine: 'Lc0' },
      };
    },
  };
  const moment = { fen: START_FEN, history: [START_FEN], move: { ply: 1 } };
  const report = createCriticalReport('sig', 'base', [moment], [lane]);
  const cache = new Map();
  await runCriticalReport(report, [lane], cache, new AbortController().signal, () => {});
  const splitKKey = criticalMomentCacheKey(moment, lane);
  const originalKey = criticalMomentCacheKey(moment, { row: lane.row, provenance: { ...provenance, model: ORIGINAL } });
  assert.notEqual(splitKKey, originalKey);
  assert.deepEqual([...cache.keys()], [splitKKey]);
  assert.equal(report.cells.lc0[0].result.cacheKey, splitKKey);
});
