import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { ENGINE_PROFILE_BACKUP_KIND, engineSetupRuntime, normalizeEngineProfiles } from '../src/lc0/engineSetups.ts';

const row = { family: 'sf', variant: 'lite', strength: 8 };
const sanitize = (value) => (value?.family === 'sf' && value.variant !== 'custom' ? { ...value } : null);
const profile = (name, extra = {}) => ({ name, rows: [row], multiPv: 3, lc0Runtime: 'onnx', ...extra });

test('saved setups and version-one backups use the same normalization', () => {
  const profiles = [profile(' Work '), profile('Fast', { multiPv: 25 })];
  const expected = [profile('Fast', { multiPv: 10 }), profile('Work')];
  assert.deepEqual(normalizeEngineProfiles(profiles, sanitize), expected);
  assert.deepEqual(normalizeEngineProfiles({ kind: ENGINE_PROFILE_BACKUP_KIND, version: 1, profiles }, sanitize), expected);
  assert.equal(profiles[0].name, ' Work ');
});

test('setup imports reject invalid envelopes, rows and reserved names', () => {
  for (const input of [
    null,
    3,
    {},
    { kind: 'other', version: 1, profiles: [profile('A')] },
    { kind: ENGINE_PROFILE_BACKUP_KIND, version: 2, profiles: [profile('A')] },
  ]) {
    assert.deepEqual(normalizeEngineProfiles(input, sanitize), []);
  }
  assert.deepEqual(
    normalizeEngineProfiles(
      [
        null,
        {},
        profile(''),
        profile('builtin:probe'),
        profile('Custom', { rows: [{ ...row, variant: 'custom' }] }),
        profile('Invalid', { rows: [{}] }),
        profile('Good', { lc0Runtime: {}, multiPv: -3 }),
      ],
      sanitize,
    ),
    [profile('Good', { multiPv: 1 })],
  );
});

test('duplicate names are unambiguous and valid row policy is delegated', () => {
  const seen = [];
  const results = normalizeEngineProfiles([profile('Same'), profile('Same', { multiPv: 1, rows: [row, row] })], (value, index) => {
    seen.push(index);
    return sanitize(value);
  });
  assert.deepEqual(seen, [0, 0, 1]);
  assert.equal(results.length, 1);
  assert.equal(results[0].multiPv, 1);
  assert.equal(results[0].rows.length, 2);
});

test('large-net setup confirmation reads metadata without allocating an engine session', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const guard = source.slice(source.indexOf('function profileRowsForUse('), source.indexOf('function loadEngineProfiles('));
  assert.match(guard, /const config = BIG_NETS\[/);
  // Whitespace-tolerant: the formatter may wrap this condition across lines.
  assert.match(
    guard,
    /!phoneBigNetBlocked\(config\.key\)\s*&&\s*bigNetSelectable\(config\)\s*&&\s*allowBt4Prompt\s*&&\s*\(?\s*await requestAnalysisBigNetConsent/,
  );
  assert.match(guard, /if \(!allowed\) next.variant = 'small'/);
  assert.doesNotMatch(guard, /bigNetFor\(|acquireBigNetSearcher\(/);
});

test('a large-net setup waits for its nets to become available before asking for consent', async () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const loader = source.slice(source.indexOf('async function profileRowsForUse('), source.indexOf('function loadEngineProfiles('));
  const pending = Promise.withResolvers();
  const checks = [];
  const prompts = [];
  const present = new Set();
  const deps = {
    BIG_NETS: { bt4: { key: 'bt4' }, t3: { key: 't3' } },
    normalizeDeployEngineRow: (row) => row,
    clampStrengthForRow: (row) => row.strength,
    isLc0BigNetVariant: (key) => key === 'bt4' || key === 't3',
    phoneBigNetBlocked: () => false,
    bigNetSelectable: ({ key }) => present.has(key),
    waitForBigNetAsset: async ({ key }) => {
      checks.push(key);
      await pending.promise;
      present.add(key);
      return 'present';
    },
    requestAnalysisBigNetConsent: async ({ key }) => {
      prompts.push(key);
      return true;
    },
  };
  const load = new Function(...Object.keys(deps), `${stripTypeScriptTypes(loader)}; return profileRowsForUse;`)(...Object.values(deps));
  const rows = ['bt4', 't3'].map((variant) => ({ family: 'lc0', variant, strength: 100 }));
  const loading = load(rows, true, new AbortController().signal);
  assert.deepEqual(checks, ['bt4']);
  assert.deepEqual(prompts, []);
  pending.resolve();
  assert.deepEqual(await loading, rows);
  assert.deepEqual(checks, ['bt4', 't3']);
  assert.deepEqual(prompts, ['bt4', 't3']);
});

test('refreshing Analysis availability preserves the focused version control', async () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('async function refreshBigNetAvailability('), source.indexOf('function renderBigNetWarning('));
  const label = 'Engine 1 model or variant';
  const original = { getAttribute: () => label };
  const document = { activeElement: original };
  const replacement = {
    focus: () => {
      document.activeElement = replacement;
    },
  };
  const deps = {
    document,
    CSS: { escape: (value) => value },
    el: () => ({ contains: (node) => node === original, querySelector: (selector) => (selector === `[aria-label="${label}"]` ? replacement : null) }),
    BIG_NETS: { bt4: {}, t3: {} },
    engineRows: [],
    probeBt4Support: async () => true,
    checkBigNetAsset: async () => 'present',
    isStaleMount: () => false,
    whileAnalysisMounted: (callback) => callback,
    renderRecklessRuntimeInfo() {},
    renderEngineList: () => {
      document.activeElement = null;
    },
  };
  const refresh = new Function(...Object.keys(deps), `${stripTypeScriptTypes(body)}; return refreshBigNetAvailability;`)(...Object.values(deps));
  await refresh(new AbortController().signal);
  assert.equal(document.activeElement, replacement);
});

for (const action of ['add', 'change']) {
  test(`Analysis ${action} waits for the hosted CPU default before committing the row`, async () => {
    const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
    const extract = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
    const handlers = new Map();
    const rows = [{ family: 'sf', variant: 'lite', strength: 8 }];
    const probe = Promise.withResolvers();
    const deps = {
      mountAbort: new AbortController(),
      el: (id) => ({ addEventListener: (event, handler) => handlers.set(`${id}:${event}`, handler) }),
      nextEngineFamily: () => 'reckless',
      resolveDefaultRecklessVariantAssetFallback: async () => probe.promise,
      whileAnalysisMounted: (callback) => callback,
      renderRecklessRuntimeInfo() {},
      defaultStrength: () => 12,
      normalizeDeployEngineRow: (row) => row,
      renderEngineList() {},
      renderEngineSetupSummary() {},
      persistManualEngineRows() {},
      deferAnalysisSettingsRefresh() {},
      prefetchAnalysisAssetsForAddedRow() {},
      cancelAnalysisAssetPrefetch() {},
      reconcileActiveRuntimeFallbackWarning() {},
      lineCache: new Map(),
      tree: { current: { fen: 'start' } },
    };
    const controller = `
      let REQUESTED_RECKLESS_VARIANT = { key: 'relaxed-simd' };
      const REQUESTED_RECKLESS_EXPLICIT = false;
      ${extract('let profileLoadAbort:', 'async function applyEngineProfile(')}
      ${extract('function defaultVariant(', 'function rowLabel(')}
      ${extract('async function resolveAnalysisCpuDefault(', 'async function init(mountSignal:')}
      ${extract("  el('engineList').addEventListener('change'", "  el('engineProfileSelect').addEventListener(")}
      ${extract('  const addEngineRow =', "  el('lc0RuntimeSelect').addEventListener(")}
    `;
    new Function('engineRows', ...Object.keys(deps), stripTypeScriptTypes(controller))(rows, ...Object.values(deps));
    const updating =
      action === 'add'
        ? handlers.get('addEngine:click')()
        : handlers.get('engineList:change')({ target: { value: 'reckless', dataset: { i: '0' }, classList: { contains: (name) => name === 'row-fam' } } });
    assert.deepEqual(
      rows.map(({ variant }) => variant),
      ['lite'],
      'do not start the unavailable relaxed build',
    );
    probe.resolve({ key: 'simd' });
    await updating;
    assert.deepEqual(rows.at(-1), { family: 'reckless', variant: 'simd', strength: 12 });
  });
}

test('normal saved setups cannot activate developer-only runtimes, including on restore/import', () => {
  for (const runtime of ['onnx', 'hybrid-wgsl-heads', 'hybrid-ort-heads', 'whole-onnx-webgpu']) {
    assert.equal(engineSetupRuntime(runtime, false), 'onnx');
  }
  assert.equal(engineSetupRuntime('hybrid-wgsl-heads', true), 'hybrid-wgsl-heads');
  const [research] = normalizeEngineProfiles([profile('Research', { lc0Runtime: 'hybrid-wgsl-heads' })], sanitize);
  assert.equal(engineSetupRuntime(research.lc0Runtime, false), 'onnx');
});
