import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
  cancelHiddenTabWork,
  classifyHiddenReload,
  finishedRunFollowUp,
  hiddenAnalysisStopEnabled,
  initialHiddenTabPause,
  rememberDeferredForce,
  requestHiddenTabWork,
  shouldResumeHiddenAnalysis,
  stopHiddenTabResume,
  transitionHiddenTab,
} from '../src/lc0/hiddenTabPause.ts';

test('hiding pauses only the unbounded work that is running', () => {
  const hidden = transitionHiddenTab(initialHiddenTabPause(false), true, ['prefetch', 'analysis']);
  assert.deepEqual(hidden.pause, ['analysis', 'prefetch']);
  assert.deepEqual(hidden.resume, []);
  assert.equal(hidden.state.hidden, true);
  assert.deepEqual(hidden.state.suspended, ['analysis', 'prefetch']);
});

test('showing resumes suspended work in a stable order and clears it', () => {
  const hidden = transitionHiddenTab(initialHiddenTabPause(false), true, ['maia', 'analysis']);
  const shown = transitionHiddenTab(hidden.state, false, ['analysis']);
  assert.deepEqual(shown.resume, ['analysis', 'maia']);
  assert.deepEqual(shown.pause, []);
  assert.equal(shown.state.hidden, false);
  assert.deepEqual(shown.state.suspended, []);
});

test('a repeated hide does not drop work that is already suspended', () => {
  const hidden = transitionHiddenTab(initialHiddenTabPause(false), true, ['analysis']);
  const again = transitionHiddenTab(hidden.state, true, []);
  assert.equal(again.state, hidden.state);
  assert.deepEqual(again.pause, []);
  assert.deepEqual(again.resume, []);
});

test('work requested while hidden waits until the tab is visible', () => {
  let state = transitionHiddenTab(initialHiddenTabPause(false), true, []).state;
  const requested = requestHiddenTabWork(state, 'analysis');
  assert.equal(requested.start, false);
  state = requested.state;
  const duplicate = requestHiddenTabWork(state, 'analysis');
  assert.equal(duplicate.start, false);
  assert.deepEqual(duplicate.state.suspended, ['analysis']);
  assert.deepEqual(transitionHiddenTab(state, false, []).resume, ['analysis']);
});

test('a visible request starts immediately and is not remembered', () => {
  const state = initialHiddenTabPause(false);
  const requested = requestHiddenTabWork(state, 'prefetch');
  assert.equal(requested.start, true);
  assert.equal(requested.state, state);
});

test('queued maintenance is restored only when the tab is shown', () => {
  const hidden = transitionHiddenTab(initialHiddenTabPause(false), true, ['analysis']);
  assert.equal(shouldResumeHiddenAnalysis(true, hidden.resume, true, false), false);
  assert.equal(shouldResumeHiddenAnalysis(true, hidden.resume, false, true), false);
  const shown = transitionHiddenTab(hidden.state, false, []);
  assert.equal(shouldResumeHiddenAnalysis(false, shown.resume, true, false), true);
  assert.equal(shouldResumeHiddenAnalysis(false, shown.resume, false, false), true);
  assert.equal(shouldResumeHiddenAnalysis(false, [], false, false), false);
});

test('a maintenance-only reload is not an analysis resume', () => {
  assert.deepEqual(classifyHiddenReload(false, true, false, false), { analysis: false, pauseReload: true });
  assert.deepEqual(classifyHiddenReload(false, true, true, false), { analysis: true, pauseReload: false });
  assert.deepEqual(classifyHiddenReload(false, true, false, true), { analysis: true, pauseReload: false });
  assert.deepEqual(classifyHiddenReload(true, false, false, false), { analysis: true, pauseReload: false });
  assert.deepEqual(classifyHiddenReload(false, false, false, false), { analysis: false, pauseReload: false });
});

test('a run that finishes while hidden does not start its follow-up', () => {
  assert.deepEqual(finishedRunFollowUp(true, { reload: true, dispose: true, analyzeAgain: false }), {
    start: false,
    reload: true,
    dispose: true,
    analyzeAgain: false,
  });
  assert.deepEqual(finishedRunFollowUp(false, { reload: true, dispose: false, analyzeAgain: true }), {
    start: true,
    reload: true,
    dispose: false,
    analyzeAgain: true,
  });
});

test('a forced request while hidden stays forced', () => {
  assert.equal(rememberDeferredForce(false, true), true);
  assert.equal(rememberDeferredForce(true, false), true);
  assert.equal(rememberDeferredForce(false, false), false);
});

test('a hidden suspended analysis keeps Stop available', async () => {
  const paused = transitionHiddenTab(initialHiddenTabPause(false), true, ['analysis']).state;
  assert.equal(hiddenAnalysisStopEnabled(paused), true);
  assert.equal(hiddenAnalysisStopEnabled(cancelHiddenTabWork(paused, 'analysis')), false);
  assert.equal(hiddenAnalysisStopEnabled(initialHiddenTabPause(false)), false);
  assert.equal(hiddenAnalysisStopEnabled({ hidden: true, suspended: ['maia'] }), false);
  const source = await readFile(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  assert.match(source, /toggleAttribute\('disabled', !hiddenAnalysisStopEnabled\(hiddenTab\)\)/);
  const pauseAt = source.indexOf("if (transition.pause.includes('analysis'))");
  const pauseBlock = source.slice(pauseAt, pauseAt + 1400);
  const clearAt = pauseBlock.indexOf('clearAnalysisSearchProgress()');
  const idleAt = pauseBlock.indexOf('analyzing = false');
  const dropAt = pauseBlock.indexOf('lineCache.delete(tree.current.fen)');
  const renderAt = pauseBlock.indexOf('renderLines()');
  assert.ok(pauseAt >= 0);
  assert.ok(idleAt >= 0 && clearAt > idleAt && dropAt > clearAt && renderAt > dropAt);
});

test('stop while hidden drops the search and keeps queued maintenance', () => {
  let state = transitionHiddenTab(initialHiddenTabPause(false), true, ['analysis']).state;
  state = cancelHiddenTabWork(state, 'analysis');
  const kept = stopHiddenTabResume(true, true);
  assert.deepEqual(kept, { reload: true, dispose: true });
  const shown = transitionHiddenTab(state, false, []);
  assert.deepEqual(shown.resume, []);
  assert.equal(shouldResumeHiddenAnalysis(false, shown.resume, kept.reload, kept.dispose), true);
});

test('explicit cancel while hidden is not resumed', () => {
  let state = transitionHiddenTab(initialHiddenTabPause(false), true, ['analysis', 'prefetch']).state;
  state = cancelHiddenTabWork(state, 'analysis');
  assert.deepEqual(state.suspended, ['prefetch']);
  assert.deepEqual(transitionHiddenTab(state, false, []).resume, ['prefetch']);
  assert.deepEqual(cancelHiddenTabWork(initialHiddenTabPause(false), 'maia').suspended, []);
});
