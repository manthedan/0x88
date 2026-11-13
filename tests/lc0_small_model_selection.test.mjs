import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { createCriticalReport, criticalMomentCacheKey, runCriticalReport } from '../src/lc0/criticalMoments.ts';
import {
  createLc0OnnxEvaluatorFromUrl,
  LC0_BIG_NET_SPLITK_FILES,
  LC0_SMALL_BATCH1_QDQ8_PATH,
  LC0_SMALL_BATCH1_SPLITK_FILE,
  lc0SmallModelRequest,
  lc0SmallSplitKUrl,
  lc0SplitKUrl,
  selectLc0OrtModelUrl,
} from '../src/lc0/lc0SmallModelSelection.ts';
import {
  collectOrtRuntimeDiagnostics,
  gpuAdapterShaderF16,
  gpuAdapterVendor,
  planOrtSessionAttempts,
  resetWebGpuAdapterProbeForTests,
  resolvedOrtExecutionProviders,
  resolveOrtSessionPlacement,
  setRequestedOrtExecutionProviderForCurrentThread,
} from '../src/nn/ortRuntime.ts';

const ORIGINAL = LC0_SMALL_BATCH1_QDQ8_PATH;
const SPLITK = `/models/lc0/${LC0_SMALL_BATCH1_SPLITK_FILE}`;
const APPLE_WEBGPU = { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' };
// Big nets (docs/bignet_smolgen_splitk_2026-11.md): only t3 serves its smolgen split-K
// rewrite; BT4 (within noise) and LQO (small gain, gate WDL at its limit) keep their shipped model.
const BIG_NETS = [['/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx', '/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx']];
const BT4 = '/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx';
const LQO = '/models/lc0/lqo_v2.f16.qdq8.onnx';

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
  // Models without a split-K rewrite are never swapped, whatever the device.
  for (const url of ['/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx', BT4, LQO, '/models/maia3/maia3_simplified.qdq8.graph.onnx']) {
    assert.equal(selectLc0OrtModelUrl(url, APPLE_WEBGPU), url);
  }
});

test('big-net split-K: the table names exactly t3, and maps by file name', () => {
  assert.deepEqual(Object.keys(LC0_BIG_NET_SPLITK_FILES).sort(), BIG_NETS.map(([original]) => original.split('/').pop()).sort());
  for (const [original, splitk] of BIG_NETS) {
    assert.equal(lc0SplitKUrl(original), splitk);
    assert.equal(lc0SplitKUrl(`https://assets.0x88.app${original}?v=2`), `https://assets.0x88.app${splitk}?v=2`);
    assert.equal(lc0SplitKUrl(splitk), undefined);
    // The Lc0 Small helper keeps its old meaning.
    assert.equal(lc0SmallSplitKUrl(original), undefined);
  }
  assert.equal(lc0SplitKUrl(ORIGINAL), SPLITK);
  assert.equal(lc0SplitKUrl(BT4), undefined);
  assert.equal(lc0SplitKUrl(LQO), undefined);
  assert.equal(lc0SplitKUrl('/models/lc0/toString'), undefined);
});

test('big-net split-K selection follows the same Apple + WebGPU rule', () => {
  for (const [original, splitk] of BIG_NETS) {
    assert.equal(selectLc0OrtModelUrl(original, APPLE_WEBGPU), splitk);
    for (const placement of [
      { providers: ['webgpu', 'wasm'], adapterVendor: 'nvidia' },
      { providers: ['webgpu'] },
      { providers: ['wasm'], adapterVendor: 'apple' },
    ]) {
      assert.equal(selectLc0OrtModelUrl(original, placement), original, JSON.stringify(placement));
    }
  }
});

test('big-net WebGPU failure: the WASM fallback runs the ORIGINAL big net', async () => {
  for (const [original, splitk] of BIG_NETS) {
    const ok = fakeDeps(APPLE_WEBGPU);
    const served = await createLc0OnnxEvaluatorFromUrl(original, { deviceSelection: true }, ok.deps);
    assert.deepEqual(ok.loads, [splitk]);
    assert.equal(served.servedUrl, splitk);
    const failing = fakeDeps(APPLE_WEBGPU, { sessionFails: true });
    const fellBack = await createLc0OnnxEvaluatorFromUrl(original, { deviceSelection: true }, failing.deps);
    assert.deepEqual(failing.loads, [splitk, original]);
    assert.deepEqual(failing.sessions, [{ input: `bytes:${original}`, providers: ['wasm'] }]);
    assert.equal(fellBack.servedUrl, original);
  }
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

test('createOrtSession attempt plan: split-K attempts are WebGPU-only and the WASM attempt uses wasmFallbackModel', async () => {
  const prepared = { input: 'folded-splitk', original: 'raw-splitk' };
  const withModel = planOrtSessionAttempts(prepared, ['webgpu', 'wasm'], true, { wasmFallbackModel: async () => 'raw-original' });
  // Not ['webgpu', 'wasm']: with wasm listed, ORT drops a WebGPU backend that fails
  // to init and resolves a WASM session on the split-K bytes, so the fallback never runs.
  assert.deepEqual(
    withModel.map((a) => a.providers),
    [['webgpu'], ['webgpu'], ['wasm']],
  );
  assert.deepEqual(
    withModel.map((a) => a.kind),
    ['primary', 'unfolded', 'wasm-fallback'],
  );
  assert.equal(withModel[0].input, 'folded-splitk');
  assert.equal(withModel[1].input, 'raw-splitk');
  assert.equal(typeof withModel[2].input, 'function');
  assert.equal(await withModel[2].input(), 'raw-original');
  // Without the option the existing behaviour is unchanged: ['webgpu', 'wasm'], then WASM on the unfolded primary bytes.
  const plain = planOrtSessionAttempts(prepared, ['webgpu', 'wasm'], true);
  assert.deepEqual(plain[0].providers, ['webgpu', 'wasm']);
  assert.equal(plain[2].input, 'raw-splitk');
  // Strict ep=webgpu has no WASM attempt at all.
  assert.equal(planOrtSessionAttempts(prepared, ['webgpu'], false, { wasmFallbackModel: async () => 'x' }).length, 2);
});

async function withFakeAdapter(adapter, run, ep = 'auto') {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: { hardwareConcurrency: 8, gpu: { requestAdapter: async () => adapter } },
    configurable: true,
    writable: true,
  });
  resetWebGpuAdapterProbeForTests();
  setRequestedOrtExecutionProviderForCurrentThread(ep);
  try {
    return await run();
  } finally {
    setRequestedOrtExecutionProviderForCurrentThread(null);
    resetWebGpuAdapterProbeForTests();
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  }
}

test('resolveOrtSessionPlacement reports the adapter vendor for a WebGPU session', async () => {
  await withFakeAdapter({ info: { vendor: 'Apple', architecture: 'metal-3' } }, async () => {
    const placement = await resolveOrtSessionPlacement();
    assert.deepEqual(placement, { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' });
    assert.equal(selectLc0OrtModelUrl(ORIGINAL, placement), SPLITK);
  });
});

test('unreadable adapter metadata leaves the vendor unknown but keeps WebGPU and the original model', async () => {
  const throwingInfo = {
    get info() {
      throw new Error('info getter failed');
    },
  };
  await withFakeAdapter(throwingInfo, async () => {
    const placement = await resolveOrtSessionPlacement();
    assert.deepEqual(placement, { providers: ['webgpu', 'wasm'] });
    assert.equal(selectLc0OrtModelUrl(ORIGINAL, placement), ORIGINAL);
  });
  const throwingVendor = {
    info: {
      get vendor() {
        throw new Error('vendor getter failed');
      },
    },
  };
  await withFakeAdapter(throwingVendor, async () => {
    assert.deepEqual(await resolveOrtSessionPlacement(), { providers: ['webgpu', 'wasm'] });
  });
  assert.equal(await gpuAdapterVendor(throwingInfo), undefined);
  // A throwing `info` still falls through to the older requestAdapterInfo().
  const legacy = {
    get info() {
      throw new Error('no info');
    },
    requestAdapterInfo: async () => ({ vendor: 'Apple' }),
  };
  assert.equal(await gpuAdapterVendor(legacy), 'apple');
});

test('an adapter without shader-f16 resolves ORT to WASM: the f16 nets fail at first run there', async () => {
  // Yukon 3090 with f16 hidden: session creation succeeded, then OrtRun threw
  // "Program Transpose requires f16 but the device does not support it".
  const noF16 = { features: new Set(['timestamp-query']), info: { vendor: 'Apple' } };
  await withFakeAdapter(noF16, async () => {
    const placement = await resolveOrtSessionPlacement();
    assert.deepEqual(placement, { providers: ['wasm'] });
    assert.equal(selectLc0OrtModelUrl(ORIGINAL, placement), ORIGINAL);
    assert.deepEqual(resolvedOrtExecutionProviders(), ['wasm']);
  });
  await withFakeAdapter(noF16, async () => {
    // The big-net support probe reads this, so its UI matches the worker's backend.
    const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
    assert.equal(diag.adapter?.ok, true);
    assert.equal(diag.adapter?.shaderF16, false);
    assert.deepEqual(diag.resolvedExecutionProviders, ['wasm']);
  });
  await withFakeAdapter(
    noF16,
    async () => {
      // A strict page keeps WebGPU for itself, but the adapter verdict is still
      // reported, so the big-net probe matches its ep:'auto' worker (WASM).
      const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
      assert.deepEqual(diag.resolvedExecutionProviders, ['webgpu']);
      assert.equal(diag.adapter?.shaderF16, false);
    },
    'webgpu',
  );
  await withFakeAdapter(
    noF16,
    async () => {
      await resolveOrtSessionPlacement();
      assert.deepEqual(resolvedOrtExecutionProviders(), ['wasm']);
    },
    'webgpu,wasm',
  );
  // Strict ep=webgpu never degrades silently; it keeps WebGPU and surfaces the error.
  await withFakeAdapter(
    noF16,
    async () => {
      await resolveOrtSessionPlacement();
      assert.deepEqual(resolvedOrtExecutionProviders(), ['webgpu']);
    },
    'webgpu',
  );
});

test('an adapter with shader-f16, or one whose features are unreadable, keeps WebGPU', async () => {
  await withFakeAdapter({ features: new Set(['shader-f16']), info: { vendor: 'Apple' } }, async () => {
    assert.deepEqual(await resolveOrtSessionPlacement(), APPLE_WEBGPU);
  });
  const throwingFeatures = {
    get features() {
      throw new Error('features getter failed');
    },
    info: { vendor: 'NVIDIA' },
  };
  await withFakeAdapter(throwingFeatures, async () => {
    assert.deepEqual(await resolveOrtSessionPlacement(), { providers: ['webgpu', 'wasm'], adapterVendor: 'nvidia' });
  });
  assert.equal(gpuAdapterShaderF16({ features: new Set(['shader-f16']) }), true);
  assert.equal(gpuAdapterShaderF16({ features: new Set() }), false);
  assert.equal(gpuAdapterShaderF16({}), null);
  assert.equal(gpuAdapterShaderF16(null), null);
  assert.equal(gpuAdapterShaderF16(throwingFeatures), null);
});

test('an explicit ?model= override, even naming the default file, loads exactly that URL', () => {
  assert.deepEqual(lc0SmallModelRequest(ORIGINAL, null), { modelUrl: ORIGINAL, deviceSelection: true });
  assert.deepEqual(lc0SmallModelRequest(ORIGINAL, undefined), { modelUrl: ORIGINAL, deviceSelection: true });
  assert.deepEqual(lc0SmallModelRequest(ORIGINAL, ORIGINAL), { modelUrl: ORIGINAL, deviceSelection: false });
  assert.deepEqual(lc0SmallModelRequest(ORIGINAL, SPLITK), { modelUrl: SPLITK, deviceSelection: false });
  assert.deepEqual(
    lc0SmallModelRequest('https://cdn.example/x.onnx', '/m.onnx', (url) => `https://cdn.example${url}`),
    {
      modelUrl: 'https://cdn.example/m.onnx',
      deviceSelection: false,
    },
  );
  // The product pages take the opt-in from the request, not from URL equality.
  for (const file of ['analysisBrowser', 'playBrowser', 'arenaBrowser']) {
    const source = readFileSync(new URL(`../src/lc0/${file}.ts`, import.meta.url), 'utf8');
    assert.match(source, /lc0SmallModelRequest\(\s*DEFAULT_MODEL_URL,\s*isV0DeployProfile\(\) \? null : params\.get\('model'\)/, file);
    assert.doesNotMatch(source, /(deviceSelection|deviceModelSelection): MODEL_URL === DEFAULT_MODEL_URL/, file);
    assert.match(source, /(deviceSelection|deviceModelSelection): LC0_DEVICE_MODEL_SELECTION/, file);
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

test('critical moments settle the served model before accepting a cached result', async () => {
  const moment = { fen: START_FEN, history: [START_FEN], move: { ply: 1 } };
  const row = { family: 'lc0', variant: 'small', strength: 400 };
  const line = { pvUci: ['e2e4'], scoreCp: 20, multipv: 1, pvSan: 'e4', scoreText: '+0.20', engine: 'Lc0' };
  const makeLane = (servedModel) => {
    // Starts out claiming split-K (as a previous job's worker did); prepare() reports this job's worker.
    const provenance = { engine: 'Lc0 Small', model: SPLITK, build: 'b', runtime: 'r', budget: '400 visits', options: 'o' };
    const lane = {
      id: 'lc0',
      row,
      provenance,
      evaluations: 0,
      prepare: async () => {
        provenance.model = servedModel;
      },
      evaluate: async () => {
        lane.evaluations++;
        return { cacheKey: '', provenance: { ...provenance }, line: { ...line, scoreCp: 99 } };
      },
    };
    return lane;
  };
  const splitKKey = criticalMomentCacheKey(moment, { row, provenance: { ...makeLane(SPLITK).provenance } });
  const seeded = () => new Map([[splitKKey, { cacheKey: splitKKey, provenance: { model: SPLITK }, line }]]);

  // This job's worker fell back to the original model: the split-K result must not be reused.
  const fellBack = makeLane(ORIGINAL);
  const cache = seeded();
  const report = createCriticalReport('sig', 'base', [moment], [fellBack]);
  await runCriticalReport(report, [fellBack], cache, new AbortController().signal, () => {});
  assert.equal(fellBack.evaluations, 1);
  assert.equal(report.cells.lc0[0].cached, false);
  assert.equal(report.cells.lc0[0].result.provenance.model, ORIGINAL);
  assert.notEqual(report.cells.lc0[0].result.cacheKey, splitKKey);

  // Same model as the cached result: the hit is accepted without a search.
  const same = makeLane(SPLITK);
  const sameReport = createCriticalReport('sig', 'base', [moment], [same]);
  await runCriticalReport(sameReport, [same], seeded(), new AbortController().signal, () => {});
  assert.equal(same.evaluations, 0);
  assert.equal(sameReport.cells.lc0[0].cached, true);
});
