import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';

function extracted(path, start, end) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `${start} not found in ${path}`);
  return stripTypeScriptTypes(source.slice(from, to));
}

test('Analysis disposes Stockfish Lite and Full when no row (or running review) uses them', () => {
  const source = `let stockfishLite = state.lite; let stockfishFull = state.full;
${extracted('../src/lc0/analysisBrowser.ts', 'function disposeStockfish(', 'const ANALYSIS_CPU_ENGINE_OPTIONS')}
${extracted('../src/lc0/analysisBrowser.ts', 'function disposeUnusedEngines(', '// Default: one best-move arrow')}
disposeUnusedEngines();
return { lite: stockfishLite, full: stockfishFull };`;
  const pool = { retain() {} };
  const run = (rows, { reviewing = false } = {}) => {
    const disposed = [];
    const engine = (name) => ({ dispose: () => disposed.push(name) });
    const prepared = new Set(['sf:lite', 'sf:full', 'berserk:x']);
    const dependencies = {
      state: { lite: engine('lite'), full: engine('full') },
      ANALYSIS_BIG_NET_KEYS: [],
      usesBigNetRow: () => false,
      peekBigNetSearcher: () => null,
      releaseBigNetSearcher: () => {},
      disposeUnusedCentipawnEvaluators: () => {},
      activeEngineRows: () => rows,
      reviewAbort: reviewing ? new AbortController() : null,
      reviewEngineRows: () => [{ family: 'sf', variant: 'lite' }],
      preparedCpuEngineKeys: prepared,
      recklessEngines: pool,
      viridithasEngines: pool,
      berserkEngines: pool,
      plentyChessEngines: pool,
      stormphraxEngines: pool,
      recklessVariantForKey: () => ({}),
      viridithasVariantForKey: () => ({}),
      berserkVariantForKey: () => ({}),
      plentyChessVariantForKey: () => ({}),
      stormphraxVariantForKey: () => ({}),
      renderRecklessRuntimeInfo: () => {},
    };
    const left = new Function(...Object.keys(dependencies), source)(...Object.values(dependencies));
    return { disposed, kept: Object.keys(left).filter((kind) => left[kind]), prepared: [...prepared] };
  };
  assert.deepEqual(run([{ family: 'lc0', variant: 'small' }]), { disposed: ['lite', 'full'], kept: [], prepared: ['berserk:x'] });
  assert.deepEqual(run([{ family: 'sf', variant: 'full' }]).disposed, ['lite']);
  assert.deepEqual(run([{ family: 'sf', variant: 'lite' }]).kept, ['lite']);
  assert.deepEqual(run([{ family: 'berserk', variant: 'x' }], { reviewing: true }).kept, ['lite'], 'review keeps SF Lite');
});
