import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { analysisEngineName } from '../src/lc0/analysisPresentation.ts';
import { analysisIsAnalyzing, initialAnalysisRunState, transitionAnalysis } from '../src/lc0/analysisRunState.ts';

const analysisSourceUrl = new URL('../src/lc0/analysisBrowser.ts', import.meta.url);
const engineHostUrl = new URL('../src/lc0/analysis/analysisEngines.ts', import.meta.url);

async function preparationSource() {
  const source = await readFile(analysisSourceUrl, 'utf8');
  return source.slice(source.indexOf('function createAnalysisAssetPreload'), source.indexOf('async function analyzeCurrent'));
}

test('analysis model assets are only preloaded once per mounted page', async () => {
  const source = await preparationSource();
  assert.match(source, /if \(!rawUrl \|\| preloadedAnalysisAssetUrls\.has\(rawUrl\)\) return;/);
  assert.match(source, /preloadedAnalysisAssetUrls\.add\(rawUrl\)/);
});

test('asset download renders the engine label, not the content-addressed filename', async () => {
  const source = await preparationSource();
  const preload = source.slice(0, source.indexOf('async function waitForAnalysisAssetPreload'));
  const items = [];
  const createPreload = new Function(
    'showLoadingProgressItem',
    `let analysisAssetPreloadSequence = 0;
    const preloadingAnalysisAssetPromises = new Map(), preloadedAnalysisAssetUrls = new Set();
    const assetFileLabel = (url) => url.split('/').pop();
    const fetch = async () => new Response(new Uint8Array([1]));
    const hideLoadingProgressItem = () => {};
    ${stripTypeScriptTypes(preload)}
    return createAnalysisAssetPreload;`,
  )((_id, item) => items.push(item));
  await createPreload('Reckless Relaxed SIMD', 'https://assets.example/000955cb88fbe762d123.wasm').promise;
  assert.ok(items.length >= 2);
  assert.deepEqual([...new Set(items.map(({ label }) => label))], ['Reckless Relaxed SIMD']);
});

test('warm CPU engine preparation uses search progress without reopening startup progress', async () => {
  const host = await readFile(engineHostUrl, 'utf8');
  const source = host.slice(host.indexOf('async function prepareCpuEngine('), host.indexOf('async function analyzeCpuPosition('));
  assert.match(source, /if \(prepared\) beginSearch\(\);/);
  assert.match(source, /if \(prepared\) \{\s*if \(cpuSessions\.needsReset\(engine, game\)\) await start\(signal\);/);
  assert.match(source, /preparedCpuEngineKeys\.add\(preparationKey\)/);
});

test('warm Centipawn analysis skips its evaluator preparation indicator', async () => {
  const source = await readFile(analysisSourceUrl, 'utf8');
  assert.match(source, /const evaluatorPrepared = engines\.hasCentipawnEvaluator\(evaluatorKey\);/);
  assert.match(source, /if \(evaluatorPrepared\) beginSearch\(\);\s*else showLoadingProgressItem/);
});

test('a Small-to-BT4 settings change retires the old run’s displayed progress before the new run starts', async () => {
  const source = await readFile(analysisSourceUrl, 'utf8');
  const progress = source.slice(source.indexOf('function clearAnalysisSearchProgress('), source.indexOf('async function workerLc0Lines('));
  const invalidate = source.slice(source.indexOf('const ANALYSIS_REFRESH_PENDING_MESSAGE'), source.indexOf('function deferAnalysisSettingsRefresh('));
  const message = { textContent: '' };
  const searchProgressByEngine = new Map();
  const wireProgress = new Function(
    'deps',
    `const { message, searchProgressByEngine, START_FEN, analysisIsAnalyzing } = deps;
    const hiddenTab = { hidden: false }, mountAbort = new AbortController(), tree = { current: { fen: START_FEN } };
    const lineCache = new Map(), el = () => message;
    const renderAnalysisSearchProgress = () => {}, renderEngineComparison = () => {};
    const analysisProgressText = (label) => label;
    const { transitionAnalysis, initialAnalysisRunState } = deps;
    let analysisRunState = initialAnalysisRunState(), activeAnalysisRunId, analysisUiProgressSamples = 0, nextId = 0;
    const dispatchAnalysis = (event) => {
      analysisRunState = transitionAnalysis(analysisRunState, { ...event, nextId: ++nextId }).state;
      activeAnalysisRunId = analysisRunState.execution.owner?.runId;
    };
    const automaticAnalysisAllowed = () => true, isMemoryConstrainedDevice = () => false;
    const analysisIntent = () => ({ requestId: nextId + 1, force: false });
    ${stripTypeScriptTypes(progress + invalidate)}
    return { showAnalysisProgress, showIndeterminateSearchProgress, invalidateAnalysisSettings, dispatchAnalysis, analysisIntent, state: () => analysisRunState };`,
  );
  const ui = wireProgress({
    message,
    searchProgressByEngine,
    START_FEN,
    analysisIsAnalyzing,
    initialAnalysisRunState,
    transitionAnalysis,
  });
  const admit = (variant) => {
    const { revision, execution } = ui.state();
    const rows = [
      { family: 'lc0', variant },
      { family: 'sf', variant: 'lite' },
    ];
    ui.dispatchAnalysis({ type: 'PREPARED', owner: execution.owner, revision, snapshot: { rows } });
    return execution.owner.runId;
  };
  ui.dispatchAnalysis({ type: 'REQUEST', intent: ui.analysisIntent(), delayMs: 0 });
  const smallRun = admit('small');
  const sample = { visits: 392, completedVisits: 352, requestedVisits: 400, elapsedMs: 1000, move: 'd2d4', value: 0.2 };
  ui.showAnalysisProgress(smallRun, START_FEN, 'Lc0', sample);
  assert.equal(searchProgressByEngine.size, 1);

  ui.invalidateAnalysisSettings();
  assert.equal(ui.state().execution.tag, 'running', 'the scheduler lets the superseded search finish');
  assert.equal(searchProgressByEngine.size, 0, 'the new selection clears the old meter');
  const settingsMessage = message.textContent;
  ui.showAnalysisProgress(smallRun, START_FEN, 'Lc0', sample);
  ui.showIndeterminateSearchProgress(smallRun, START_FEN, 'SF Lite');
  assert.equal(message.textContent, settingsMessage, 'the old Small status must not be shown beside the BT4 card');
  assert.equal(searchProgressByEngine.size, 0, 'neither engine may repaint progress from the old run');

  ui.dispatchAnalysis({ type: 'SETTLED', operationId: smallRun });
  const bt4Run = admit('bt4');
  ui.showAnalysisProgress(bt4Run, START_FEN, 'Lc0 BT4-it332', sample);
  ui.showIndeterminateSearchProgress(bt4Run, START_FEN, 'SF Lite');
  const [net, cpu] = searchProgressByEngine.values();
  assert.equal(analysisEngineName(net.label), 'Lc0 BT4-it332');
  assert.equal(net.completed, 392, 'the visit-budget meter includes inherited root visits');
  assert.equal(net.requested, 400);
  assert.equal(net.nps, 352, 'the speed counts only new visits');
  assert.equal(analysisEngineName(cpu.label), 'Stockfish Lite');
  assert.equal(cpu.indeterminate, true);
});
