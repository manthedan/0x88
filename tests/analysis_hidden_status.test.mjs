import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { hiddenAnalysisStopEnabled, initialHiddenTabPause, requestHiddenTabWork } from '../src/lc0/hiddenTabPause.ts';

const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
function slice(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from);
  return source.slice(from, to);
}

function statusPage({ hidden = false, active = false, aborted = false, reload = false } = {}) {
  const nodes = new Map();
  const el = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      disabled: id === 'stop', dataset: {}, textContent: '', title: '',
      toggleAttribute(name, force) { if (name === 'disabled') this.disabled = force; },
      closest() { return null; },
    });
    return nodes.get(id);
  };
  const code = ts.transpile(`
    let hiddenTab = initialHiddenTabPause(hidden);
    const analyzing = active;
    const analysisAbort = active ? { signal: { aborted } } : null;
    const analysisReloadAbort = reload ? { signal: { aborted: false } } : null;
    const retry = null, lineCache = new Map(), tree = { current: { fen: 'current' } };
    const document = { getElementById: el };
    const renderPgnImportNotice = () => {};
    const analysisEngineName = (name) => name;
    ${slice('function hasActiveAnalysisRun(', 'function hiddenTabActive(')}
    ${slice('function deferHiddenTabWork(', 'function applyHiddenTab(')}
    ${slice('function renderAnalysisStatus(', 'function activeEngineNames(')}
    return { deferHiddenTabWork, renderAnalysisStatus, state: () => hiddenTab, el };
  `, { target: ts.ScriptTarget.ES2022 });
  return new Function('el', 'initialHiddenTabPause', 'requestHiddenTabWork', 'hiddenAnalysisStopEnabled', 'hidden', 'active', 'aborted', 'reload', code)(
    el, initialHiddenTabPause, requestHiddenTabWork, hiddenAnalysisStopEnabled, hidden, active, aborted, reload,
  );
}

test('hidden queued analysis is paused while Stop can cancel its resume', () => {
  const page = statusPage({ hidden: true });
  assert.equal(page.deferHiddenTabWork('analysis'), true, 'real admission helper defers; no search starts');
  assert.deepEqual(page.state(), { hidden: true, suspended: ['analysis'] });
  assert.equal(page.el('stop').disabled, false, 'Stop legitimately cancels queued resume');
  // The real MutationObserver invokes this renderer after disabled changes.
  page.renderAnalysisStatus();
  assert.equal(page.el('analysisStatus').dataset.state, 'paused');
  assert.equal(page.el('analysisStatus').textContent, 'Paused');
  assert.match(page.el('analysisStatus').title, /queued/);
  assert.equal(page.el('stop').disabled, false);
});

test('active search and reload status come from the scheduling owner, not Stop', () => {
  for (const options of [{ active: true }, { reload: true }]) {
    const page = statusPage(options);
    assert.equal(page.el('stop').disabled, true);
    page.renderAnalysisStatus();
    assert.equal(page.el('analysisStatus').dataset.state, 'running');
    assert.equal(page.el('analysisStatus').textContent, 'Analyzing…');
  }
});

test('an aborted run is not advertised as searching even if Stop is enabled', () => {
  const page = statusPage({ active: true, aborted: true });
  page.el('stop').disabled = false;
  page.renderAnalysisStatus();
  assert.equal(page.el('analysisStatus').dataset.state, 'idle');
});

test('hidden queued work takes precedence over an old draining run', () => {
  const page = statusPage({ hidden: true, active: true });
  page.deferHiddenTabWork('analysis');
  page.renderAnalysisStatus();
  assert.equal(page.el('analysisStatus').dataset.state, 'paused');
});
