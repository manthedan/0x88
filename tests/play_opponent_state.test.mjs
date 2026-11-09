import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { committedPlayOpponent, transitionPlayOpponent } from '../src/lc0/playOpponentState.ts';
import { DEFAULT_PLAY_PREFERENCES } from '../src/lc0/playPreferences.ts';

const preferences = { ...DEFAULT_PLAY_PREFERENCES, level: 4, maiaElo: 1700, maiaTemperature: 0.8 };
const edit = (state, patch, gameInProgress = true) => transitionPlayOpponent(state, {
  type: 'edit', gameInProgress, preferences: { ...(state.pending ?? state.active), ...patch },
});

for (const patch of [{ engineId: 'sf-lite' }, { color: 'black' }, { color: 'random' }, { engineId: 'lqo', color: 'black' }]) {
  test(`engine/colour edit ${JSON.stringify(patch)} commits only outside a game`, () => {
    const initial = committedPlayOpponent(preferences);
    const pending = edit(initial, patch);
    assert.equal(pending.tag, 'pending');
    assert.deepEqual(pending.active, preferences);
    assert.deepEqual(pending.pending, { ...preferences, ...patch });
    const setup = edit(initial, patch, false);
    assert.equal(setup.tag, 'committed');
    assert.deepEqual(setup.active, { ...preferences, ...patch });
  });
}

test('multiple pending edits and tuning never change the live reply, strip or autosave preferences', () => {
  const initial = committedPlayOpponent(preferences);
  let state = edit(initial, { engineId: 'sf-lite' });
  state = transitionPlayOpponent(state, { type: 'tune', preferences: { ...state.pending, level: 0, maiaElo: 600, maiaStyle: 'argmax' } });
  state = edit(state, { color: 'black' });
  assert.deepEqual(state.active, preferences);
  assert.equal(state.pending.level, 0);
  // Dismiss restores the whole form, not just engine and colour.
  assert.deepEqual(transitionPlayOpponent(state, { type: 'dismiss' }), initial);
  const confirmed = transitionPlayOpponent(state, { type: 'confirm' });
  assert.equal(confirmed.pending, null);
  assert.deepEqual(confirmed.active, state.pending);
  assert.deepEqual(initial.active, preferences, 'transitions must not mutate earlier state');
});

test('returning the draft to the active identity discards draft tuning', () => {
  const initial = committedPlayOpponent(preferences);
  let state = edit(initial, { engineId: 'sf-lite', level: 0 });
  state = edit(state, { engineId: 'maia3' });
  assert.deepEqual(state, initial);
});

test('live strength changes keep their existing policy and cannot change opponent identity', () => {
  const initial = committedPlayOpponent(preferences);
  const state = transitionPlayOpponent(initial, { type: 'tune', preferences: { ...preferences, engineId: 'sf-lite', color: 'black', maiaElo: 1900, level: 1 } });
  assert.equal(state.tag, 'committed');
  assert.equal(state.active.engineId, 'maia3');
  assert.equal(state.active.color, 'white');
  assert.equal(state.active.maiaElo, 1900);
  assert.equal(state.active.level, 1);
});

test('Stormphrax residency is retired at commit, never while restart consent is pending', () => {
  const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const setter = source.slice(source.indexOf('function ctxSetOpponent('), source.indexOf('function ctxRestoreOpponentControls('));
  const disposed = [];
  const setOpponent = new Function('disposeCachedCpuEngine', ts.transpile(`${setter}; return ctxSetOpponent;`, { target: ts.ScriptTarget.ESNext }))((id) => disposed.push(id));
  const ctx = { opponent: committedPlayOpponent({ ...preferences, engineId: 'stormphrax' }) };
  setOpponent(ctx, edit(ctx.opponent, { engineId: 'maia3' }));
  assert.deepEqual(disposed, []);
  setOpponent(ctx, transitionPlayOpponent(ctx.opponent, { type: 'dismiss' }));
  assert.deepEqual(disposed, []);
  setOpponent(ctx, edit(ctx.opponent, { engineId: 'sf-lite' }));
  setOpponent(ctx, transitionPlayOpponent(ctx.opponent, { type: 'confirm' }));
  assert.deepEqual(disposed, ['stormphrax']);
  // Fresh setup changes also commit immediately and retire the previous worker.
  ctx.opponent = committedPlayOpponent({ ...preferences, engineId: 'stormphrax' });
  setOpponent(ctx, edit(ctx.opponent, { engineId: 'maia3' }, false));
  assert.deepEqual(disposed, ['stormphrax', 'stormphrax']);
});

test('takeback leaves pending consent intact; reload restores only the active settings', () => {
  const initial = committedPlayOpponent(preferences);
  const state = edit(initial, { engineId: 'sf-lite' });
  // Position changes do not own opponent state. Restoring a saved game uses
  // its active preferences, even if it was saved while consent was pending.
  const restored = committedPlayOpponent(JSON.parse(JSON.stringify(state.active)));
  assert.deepEqual(restored, initial);
  const newGame = transitionPlayOpponent(state, { type: 'reset', preferences: state.pending });
  assert.equal(newGame.tag, 'committed');
  assert.equal(newGame.active.engineId, 'sf-lite');
  // An ended game's next setup has no confirmation requirement.
  assert.equal(edit(state, { color: 'random' }, false).pending, null);
});
