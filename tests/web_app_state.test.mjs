import test from 'node:test';
import assert from 'node:assert/strict';
import { START_FEN, boardToFen } from '../src/chess/board.ts';
import { moveFromUci } from '../src/chess/moveCodec.ts';
import { appReducer, createAppState, selectLastMove, selectLivePly } from '../src/web/appState.ts';

function apply(state, action) {
  return appReducer(state, action);
}

test('web app state keeps move history and historyFens from a single source of truth', () => {
  let state = createAppState({ uiMode: 'analysis' });
  assert.equal(boardToFen(state.board), START_FEN);
  assert.equal(state.currentPly, 0);
  assert.deepEqual(state.moves, []);
  assert.deepEqual(state.positionFens, [START_FEN]);
  assert.deepEqual(state.historyFens, []);
  assert.equal(selectLivePly(state), true);

  state = apply(state, { type: 'record-move', move: moveFromUci('e2e4'), san: 'e4' });
  state = apply(state, { type: 'record-move', move: moveFromUci('e7e5'), san: 'e5' });

  assert.equal(state.currentPly, 2);
  assert.deepEqual(state.moves.map((m) => m.uci), ['e2e4', 'e7e5']);
  assert.equal(state.positionFens.length, 3);
  assert.deepEqual(state.historyFens, [state.positionFens[1], START_FEN]);
  assert.equal(selectLastMove(state), 'e7e5');
  assert.equal(selectLivePly(state), true);
});

test('web app state navigation is bounded and branch moves truncate stale future history', () => {
  let state = createAppState({ uiMode: 'analysis' });
  state = apply(state, { type: 'record-move', move: moveFromUci('e2e4'), san: 'e4' });
  state = apply(state, { type: 'record-move', move: moveFromUci('e7e5'), san: 'e5' });
  const mainlineFenAfterE5 = state.positionFens[2];

  state = apply(state, { type: 'navigate-history', ply: 99 });
  assert.equal(state.currentPly, 2, 'navigation past the end clamps to live ply');

  state = apply(state, { type: 'navigate-history', ply: 1 });
  assert.equal(state.currentPly, 1);
  assert.equal(selectLivePly(state), false);
  assert.deepEqual(state.historyFens, [START_FEN]);

  state = apply(state, { type: 'record-move', move: moveFromUci('c7c5'), san: 'c5' });
  assert.equal(state.currentPly, 2);
  assert.deepEqual(state.moves.map((m) => m.uci), ['e2e4', 'c7c5']);
  assert.equal(state.positionFens.length, 3);
  assert.notEqual(state.positionFens[2], mainlineFenAfterE5, 'branch replaces stale future position');
  assert.equal(selectLivePly(state), true);
});

test('web app state quarantines analysis mode and async engine state transitions', () => {
  let state = createAppState({ uiMode: 'play' });
  state = apply(state, { type: 'set-pending-premove', pending: { from: 'e2', to: 'e4' } });
  state = apply(state, { type: 'set-brain-piece', piece: 'n' });
  state = apply(state, { type: 'set-busy', busy: true });
  state = apply(state, { type: 'bump-engine-request' });

  assert.equal(state.engine.busy, true);
  assert.equal(state.engine.requestId, 1);
  assert.deepEqual(state.pendingPremove, { from: 'e2', to: 'e4' });
  assert.equal(state.brainPiece, 'n');

  state = apply(state, { type: 'set-ui-mode', mode: 'analysis' });
  assert.equal(state.uiMode, 'analysis');
  assert.equal(state.pendingPremove, null);
  assert.equal(state.brainPiece, null);
});
