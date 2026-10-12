import assert from 'node:assert/strict';
import test from 'node:test';
import {
  cancelHiddenTabWork,
  initialHiddenTabPause,
  requestHiddenTabWork,
  shouldResumeHiddenAnalysis,
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

test('explicit cancel while hidden is not resumed', () => {
  let state = transitionHiddenTab(initialHiddenTabPause(false), true, ['analysis', 'prefetch']).state;
  state = cancelHiddenTabWork(state, 'analysis');
  assert.deepEqual(state.suspended, ['prefetch']);
  assert.deepEqual(transitionHiddenTab(state, false, []).resume, ['prefetch']);
  assert.deepEqual(cancelHiddenTabWork(initialHiddenTabPause(false), 'maia').suspended, []);
});
