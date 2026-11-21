import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { createAnalysisEngines } from '../../src/lc0/analysis/analysisEngines.ts';

const controller = readFileSync(new URL('../../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');

/**
 * The controller's own AnalysisEnginePort object literal, evaluated against the
 * given page bindings (mountAbort, show/hideLoadingProgressItem,
 * renderRecklessRuntimeInfo), so host tests run through the production wiring.
 */
export function productionEnginePort({ showAnalysisDeviceWarning = () => {}, ...bindings }) {
  const start = controller.indexOf('const engines = createAnalysisEngines({\n  port: {');
  const from = controller.indexOf('{', controller.indexOf('port:', start));
  const to = controller.indexOf('\n  },\n', from);
  assert.ok(start >= 0 && to > from, 'controller engine port');
  const literal = stripTypeScriptTypes(`(${controller.slice(from, to + 4)})`);
  return new Function('showAnalysisDeviceWarning', ...Object.keys(bindings), `return ${literal};`)(showAnalysisDeviceWarning, ...Object.values(bindings));
}

/** The real host with inert defaults; tests override what they exercise. */
export function testAnalysisEngines(deps = {}) {
  return createAnalysisEngines({
    port: { live: () => true, loading() {}, runtimeInfoChanged() {} },
    dial: 'balanced',
    centipawn: { legacyOverride: false, modelUrl: '', metaUrl: '', legacyModelUrl: '', legacyMetaUrl: '', manifestUrl: '' },
    loadCentipawn: () => assert.fail('unexpected Centipawn load'),
    variants: {},
    activeRows: () => [],
    preload: async () => {},
    bigNets: { acquire: () => assert.fail('unexpected big-net load'), peek: () => null, release() {}, disposeNow() {} },
    ...deps,
  });
}
