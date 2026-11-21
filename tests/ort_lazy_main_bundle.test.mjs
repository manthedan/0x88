import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { registerHooks, stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import ts from 'typescript';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { hasEngineRetirements } from '../src/lc0/engineRetirement.ts';
import { CachedLc0Evaluator, prepareLc0EvaluatorInput } from '../src/lc0/lc0EvaluatorBase.ts';
import { LC0_SMALL_BATCH1_QDQ8_PATH, lc0SmallModelRequest, lc0SplitKUrl, selectLc0OrtModelUrl } from '../src/lc0/lc0ModelUrls.ts';
import { createPlayEngines } from '../src/lc0/play/playEngines.ts';
import {
  ortDequantFoldEnabled,
  probeOrtWebGpuShaderF16Support,
  requestedOrtExecutionProvider,
  requestedOrtGraphOptimizationLevel,
  resolveFreshWorkerOrtSessionPlacement,
  setOrtDequantFoldForCurrentThread,
  setOrtRuntimeDiagnosticOptionsForCurrentThread,
  setRequestedOrtExecutionProviderForCurrentThread,
} from '../src/nn/ortConfig.ts';

function checkStaticImports(url, chain, visited) {
  if (visited.has(url.href)) return;
  visited.add(url.href);
  const source = ts.createSourceFile(url.pathname, readFileSync(url, 'utf8'), ts.ScriptTarget.Latest);
  for (const node of source.statements) {
    if (!ts.isImportDeclaration(node) && !ts.isExportDeclaration(node)) continue;
    if (node.isTypeOnly || node.importClause?.isTypeOnly || !node.moduleSpecifier) continue;
    const specifier = node.moduleSpecifier.text;
    const next = [...chain, specifier];
    assert.doesNotMatch(specifier, /^onnxruntime-web(?:\/|$)/, next.join(' -> '));
    if (specifier.startsWith('.') && /\.[cm]?[jt]s$/.test(specifier)) {
      checkStaticImports(new URL(specifier, url), next, visited);
    } else if (specifier.startsWith('#lib/')) {
      checkStaticImports(new URL(`../src/lib/${specifier.slice(5)}`, import.meta.url), next, visited);
    }
  }
}

for (const entry of ['analysisBrowser', 'playBrowser', 'arenaBrowser']) {
  test(`${entry} does not statically import ONNX Runtime`, () => {
    checkStaticImports(new URL(`../src/lc0/${entry}.ts`, import.meta.url), [entry], new Set());
  });
}

test('retiring Play Small does not wait for its ORT import or start the factory', async (t) => {
  const started = Promise.withResolvers();
  const download = Promise.withResolvers();
  const tick = () => new Promise(setImmediate);
  const factory = t.mock.fn(() => assert.fail('a retired import must not start an evaluator'));
  globalThis.playSmallImportFixture = { download, started, factory };
  const stub = `data:text/javascript,${encodeURIComponent(`
    const { download, started, factory } = globalThis.playSmallImportFixture;
    started.resolve();
    await download.promise;
    export const createLc0OnnxEvaluatorFromUrl = factory;
  `)}`;
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      return nextResolve(specifier.endsWith('/lc0SmallModelSelection.ts') ? stub : specifier, context);
    },
  });
  const ui = { current: () => () => false, note() {}, hideProgress() {} };
  const engines = createPlayEngines({ loadCentipawn: () => assert.fail('unexpected Centipawn load') });
  const loading = engines.ensureLc0Small({ modelUrl: '/small.onnx', deviceSelection: false }, ui);
  try {
    await Promise.race([started.promise, loading.then(() => assert.fail('load finished before its import'))]);
    let retired = false;
    const aborted = assert.rejects(loading, { name: 'AbortError' });
    const retiring = engines.retireUnusedEngines().then(() => {
      retired = true;
    });
    const settled = Promise.all([aborted, retiring]);
    void settled.catch(() => {});
    await tick();
    assert.equal(retired, true, 'retirement must finish while the import is still pending');
    assert.equal(hasEngineRetirements(), false);
    await settled;
    download.resolve();
    await import(stub);
    await tick();
    assert.equal(factory.mock.callCount(), 0);
  } finally {
    download.resolve();
    hook.deregister();
    delete globalThis.playSmallImportFixture;
  }
});

test('Arena mount renders basic runtime status without importing ORT', async () => {
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const badgeSource = source.slice(source.indexOf('async function renderRuntimeBadge('), source.indexOf('function selectedSeatsNeedLc0Evaluator('));
  const details = { open: false };
  const badge = { isConnected: true, closest: () => details, classList: { toggle() {} }, textContent: '' };
  let ortImports = 0;
  const render = new Function('badge', 'collectOrtRuntimeDiagnostics', `
    let runtimeIsolation = false, runtimeSharedArrayBuffer = false;
    const maybeEl = () => badge;
    const isV0DeployProfile = () => false;
    const refreshStockfishControls = () => {};
    const threadedStockfishAvailable = () => false;
    const centipawnHybridManifestStatusText = () => '';
    ${stripTypeScriptTypes(badgeSource)}
    return renderRuntimeBadge;
  `)(badge, async () => {
    ortImports++;
    return { describe: 'auto->wasm', wasm: { numThreads: 1 }, sessions: { active: 0 }, webgpuAvailable: false };
  });
  await render();
  assert.equal(ortImports, 0);
  assert.match(badge.textContent, /SAB .*WebGPU unavailable.*SF threaded no/);
  assert.doesNotMatch(badge.textContent, /ORT wasm threads/);

  details.open = true;
  await render();
  assert.equal(ortImports, 1);
  assert.match(badge.textContent, /auto->wasm · ORT wasm threads 1 · ORT sessions 0 active/);
});

test('zero-ORT config helpers update thread state and probe WebGPU safely without onnxruntime-web', async (t) => {
  let features = new Set();
  t.mock.getter(globalThis, 'navigator', () => ({ gpu: { requestAdapter: async () => ({ features }) } }));
  assert.equal(await probeOrtWebGpuShaderF16Support(), false);
  features = new Set(['shader-f16']);
  assert.equal(await probeOrtWebGpuShaderF16Support(), true);
  try {
    setRequestedOrtExecutionProviderForCurrentThread('wasm');
    assert.equal(requestedOrtExecutionProvider(), 'wasm');
    assert.deepEqual(await resolveFreshWorkerOrtSessionPlacement('wasm'), { providers: ['wasm'] });

    setOrtRuntimeDiagnosticOptionsForCurrentThread({ graphOptimizationLevel: 'basic' });
    assert.equal(requestedOrtGraphOptimizationLevel(), 'basic');

    setOrtDequantFoldForCurrentThread(false);
    assert.equal(ortDequantFoldEnabled(true), false);
    setOrtDequantFoldForCurrentThread(true);
    assert.equal(ortDequantFoldEnabled(false), true);
  } finally {
    setRequestedOrtExecutionProviderForCurrentThread(null);
    setOrtRuntimeDiagnosticOptionsForCurrentThread(null);
    setOrtDequantFoldForCurrentThread(null);
  }
});

test('zero-ORT LC0 model URL helpers select split-K and device model requests', () => {
  const req = lc0SmallModelRequest(LC0_SMALL_BATCH1_QDQ8_PATH, null, (u) => u);
  assert.equal(req.deviceSelection, true);
  assert.equal(req.modelUrl, LC0_SMALL_BATCH1_QDQ8_PATH);

  const splitK = lc0SplitKUrl(LC0_SMALL_BATCH1_QDQ8_PATH);
  assert.ok(splitK && splitK.includes('.splitk.onnx'));
  assert.equal(selectLc0OrtModelUrl(LC0_SMALL_BATCH1_QDQ8_PATH, { providers: ['webgpu', 'wasm'], adapterVendor: 'apple' }), splitK);
  assert.equal(selectLc0OrtModelUrl(LC0_SMALL_BATCH1_QDQ8_PATH, { providers: ['wasm'] }), LC0_SMALL_BATCH1_QDQ8_PATH);
});

test('zero-ORT CachedLc0Evaluator caches prepared inputs without importing onnxruntime-web', async () => {
  let evalCount = 0;
  const inner = {
    async evaluate(input) {
      evalCount += 1;
      return {
        fen: typeof input === 'string' ? input : input.fen,
        value: 0.1,
        wdl: [0.4, 0.3, 0.3],
        legalPriors: [{ uci: 'e2e4', prior: 1 }],
      };
    },
  };
  const cache = new CachedLc0Evaluator(inner, { maxEntries: 16 });
  const prepared = prepareLc0EvaluatorInput(parseFen(START_FEN));
  const first = await cache.evaluate(prepared);
  const second = await cache.evaluate(prepared);
  assert.equal(evalCount, 1);
  assert.equal(first.value, second.value);
});
