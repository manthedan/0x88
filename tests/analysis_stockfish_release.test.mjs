import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { createAnalysisEngines } from '../src/lc0/analysis/analysisEngines.ts';
import { cpuEngineHashMbForSurface } from '../src/lc0/cpuEngineMemory.ts';
import { stockfishFlavorUrl } from '../src/lc0/stockfishEngine.ts';
import { testAnalysisEngines } from './lib/analysis_engines.mjs';

function extracted(path, start, end) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `${start} not found in ${path}`);
  return stripTypeScriptTypes(source.slice(from, to));
}

test('isolated Analysis uses single-threaded Stockfish and smaller hash budgets on phones only', (t) => {
  globalThis.crossOriginIsolated = true;
  t.after(() => {
    delete globalThis.crossOriginIsolated;
    delete globalThis.location;
  });
  for (const memoryClass of ['phone', 'desktop']) {
    globalThis.location = { search: `?memoryClass=${memoryClass}` };
    const builds = [];
    const engines = testAnalysisEngines({
      createStockfish: ({ hashMb }, url) => {
        builds.push({ hashMb, url });
        return {};
      },
    });
    engines.getStockfish('lite');
    engines.getStockfish('full');
    const phone = memoryClass === 'phone';
    const hashMb = phone ? 32 : 128;
    assert.deepEqual(builds, [
      { hashMb, url: stockfishFlavorUrl(phone ? 'lite-single' : 'lite-threaded') },
      { hashMb, url: stockfishFlavorUrl(phone ? 'single' : 'threaded') },
    ]);
    assert.equal(cpuEngineHashMbForSurface('play'), phone ? 16 : 64);
    assert.equal(cpuEngineHashMbForSurface('arena'), phone ? 16 : 64);
  }
});

const PREPARATION_KEYS = ['sf:lite', 'sf:full', 'berserk:x'];

// The production disposeUnusedEngines on the real host; only the engines are fakes.
async function run(rows, { reviewing = false } = {}) {
  const disposed = [];
  const startups = [];
  const engines = createAnalysisEngines({
    port: { live: () => true, loading: (id, item) => item && startups.push(id), runtimeInfoChanged: () => {} },
    dial: 'balanced',
    centipawn: { legacyOverride: false, modelUrl: '', metaUrl: '', legacyModelUrl: '', legacyMetaUrl: '', manifestUrl: '' },
    loadCentipawn: () => assert.fail('unused'),
    centipawnCache: { retain() {}, clear() {}, has: () => false, get: () => assert.fail('unused') },
    variants: { reckless: () => ({}), viridithas: () => ({}), berserk: () => ({}), plentyChess: () => ({}), stormphrax: () => ({}) },
    activeRows: () => rows,
    preload: async () => {},
    createStockfish: (_options, url) => {
      const kind = url.includes('lite') ? 'lite' : 'full';
      return { dispose: () => disposed.push(kind) };
    },
    bigNets: { peek: () => null, release() {}, acquire: () => assert.fail('unused'), disposeNow() {} },
  });
  const game = {};
  const prepare = (key) =>
    engines.prepareCpuEngine(
      { family: 'sf', strength: 1 },
      {},
      key,
      key,
      game,
      new AbortController().signal,
      [],
      async () => {},
      () => {},
    );
  const lite = engines.getStockfish('lite');
  const full = engines.getStockfish('full');
  for (const key of PREPARATION_KEYS) await prepare(key);
  startups.length = 0;

  const dependencies = {
    engines,
    engineRows: rows,
    resetSearchWorker: () => {},
    mainEvaluator: null,
    searcher: null,
    isLc0BigNetVariant: (variant) => ['bt4', 't3'].includes(variant),
    activeEngineRows: () => rows,
    reviewAbort: reviewing ? new AbortController() : null,
    reviewEngineRows: () => [{ family: 'sf', variant: 'lite' }],
    renderRecklessRuntimeInfo: () => {},
  };
  const mark = disposed.length;
  new Function(
    ...Object.keys(dependencies),
    `${extracted('../src/lc0/analysisBrowser.ts', 'function disposeUnusedEngines(', 'function bestShapes(')}\ndisposeUnusedEngines();`,
  )(...Object.values(dependencies));
  const released = disposed.slice(mark);

  const kept = [];
  if (engines.getStockfish('lite') === lite) kept.push('lite');
  if (engines.getStockfish('full') === full) kept.push('full');
  // A key is still prepared when its next preparation shows no engine startup.
  const prepared = [];
  for (const key of PREPARATION_KEYS) {
    const before = startups.length;
    await prepare(key);
    if (startups.length === before) prepared.push(key);
  }
  engines.disposeAnalysisCpuEngines();
  return { disposed: released, kept, prepared };
}

test('Analysis disposes Stockfish Lite and Full when no row (or running review) uses them', async () => {
  assert.deepEqual(await run([{ family: 'lc0', variant: 'small' }]), { disposed: ['lite', 'full'], kept: [], prepared: ['berserk:x'] });
  assert.deepEqual((await run([{ family: 'sf', variant: 'full' }])).disposed, ['lite']);
  assert.deepEqual((await run([{ family: 'sf', variant: 'lite' }])).kept, ['lite']);
  assert.deepEqual((await run([{ family: 'berserk', variant: 'x' }], { reviewing: true })).kept, ['lite'], 'review keeps SF Lite');
});
