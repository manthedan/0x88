import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { boardToFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';

const fixtures = JSON.parse(readFileSync(new URL('../fixtures/lc0/history.json', import.meta.url), 'utf8'));

test('LC0 explicit history fixtures replay to their final FEN', () => {
  assert.deepEqual(fixtures.map((fixture) => fixture.id), [
    'italian-four-ply-history',
    'castled-black-to-move-history',
    'history-current-en-passant',
    'queens-gambit-castled-history',
    'queen-capture-retreat-history',
    'history-black-en-passant',
  ]);
  for (const fixture of fixtures) {
    const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
    assert.equal(boardToFen(positions[positions.length - 1]), fixture.finalFen, fixture.id);
    assert.equal(positions.length, fixture.moves.length + 1, fixture.id);
  }
});

test('LC0 explicit history fixtures end with legal moves for policy masking', () => {
  for (const fixture of fixtures) {
    const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
    const board = positions[positions.length - 1];
    const legal = legalMoves(board).map(moveToUci);
    assert.equal(legal.length > 0, true, fixture.id);
    assert.equal(new Set(legal).size, legal.length, fixture.id);
    if (fixture.id === 'history-black-en-passant') assert.ok(legal.includes('h4g3'), 'black en-passant remains legal after replay');
  }
});
