import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { LatestOnlyTask } from '../src/lc0/latestOnlyTask.ts';
import { DEFAULT_PLAY_PREFERENCES } from '../src/lc0/playPreferences.ts';

function extracted(path, start, end) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  return source.slice(source.indexOf(start), source.indexOf(end));
}
function compile(code, names, values) {
  return new Function(...names, ts.transpile(code, { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext }))(...values);
}
function factory() {
  const created = [];
  return {
    created,
    async create(options) {
      const evaluator = { failure: null, modelLoad: { bytes: 1 }, options };
      created.push(evaluator);
      return evaluator;
    },
  };
}

test('Play Retry evicts a fatally failed Maia cache once, even with concurrent callers', async () => {
  const Maia = factory();
  const ensure = compile(
    `
    let maia3Promise = null, maia3Status = null;
    const ctxAsyncUiIsCurrent = () => false;
    ${extracted('../src/lc0/playBrowser.ts', 'async function ctxEnsureMaia3(', 'function ctxCpuEngineFor(')}
    return ctxEnsureMaia3;
  `,
    ['Maia3BrowserEvaluator'],
    [Maia],
  );
  const ctx = { opponent: { active: DEFAULT_PLAY_PREFERENCES }, enginePreparationGeneration: 0 };
  const first = await ensure(ctx);
  first.failure = new Error('runtime crash');
  const [second, concurrent] = await Promise.all([ensure(ctx), ensure(ctx)]);
  assert.notEqual(second, first);
  assert.equal(second, concurrent);
  assert.equal(Maia.created.length, 2);
  assert.equal(await ensure(ctx), second);
});

test('Analysis Maia failure removes the failed owner, settles task ownership and re-enables a clean retry', async () => {
  const Maia = factory();
  const api = compile(
    `
    let maia3PanelEvaluator = null, maia3PanelLoading = false, maia3PanelLoadGeneration = 0;
    let maia3PanelTask = null, maia3PanelTimer = null, explorerMaia = null;
    const elements = new Map();
    const el = (id) => { if (!elements.has(id)) elements.set(id, {}); return elements.get(id); };
    const mountAbort = new AbortController(), isStaleMount = () => false;
    const showModelProgress = () => {}, hideModelProgress = () => {}, renderExplorerComparison = () => {};
    const renderMaia3Panel = async () => {}, scheduleMaia3Panel = () => {};
    const maia3PanelStatus = (text) => { el('maia3PanelStatus').textContent = text; };
    ${extracted('../src/lc0/analysisBrowser.ts', 'async function enableMaia3Panel(', 'function scheduleMaia3Panel(')}
    return { enableMaia3Panel, el, snapshot: () => ({ evaluator: maia3PanelEvaluator, task: maia3PanelTask }) };
  `,
    ['Maia3BrowserEvaluator', 'LatestOnlyTask'],
    [Maia, LatestOnlyTask],
  );
  await api.enableMaia3Panel();
  const first = api.snapshot().evaluator;
  const firstTask = api.snapshot().task;
  first.options.onFailure(new Error('runtime worker crash'));
  assert.equal(api.snapshot().evaluator, null);
  assert.equal(api.snapshot().task, null);
  assert.equal(firstTask.disposed, true);
  assert.equal(api.el('maia3Enable').hidden, false);
  assert.equal(api.el('maia3Enable').disabled, false);
  assert.equal(api.el('maia3Grid').hidden, true);
  assert.match(api.el('maia3PanelStatus').textContent, /runtime worker crash.*retry/);
  await api.enableMaia3Panel();
  const second = api.snapshot().evaluator;
  assert.notEqual(second, first);
  first.options.onFailure(new Error('stale crash'));
  assert.equal(api.snapshot().evaluator, second);
  assert.equal(api.el('maia3Enable').hidden, true);
});
