import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { boardToFen } from '../src/chess/board.ts';
import { automaticDrawReason } from '../src/chess/drawRules.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { LC0_PLANES_PER_HISTORY, encodeLc0Classical112 } from '../src/lc0/encoder112.ts';

const fixtures = JSON.parse(readFileSync(new URL('../fixtures/lc0/repetition_history.json', import.meta.url), 'utf8'));
const ALL = (1n << 64n) - 1n;

test('LC0 repetition history fixture replays to a threefold draw', () => {
  assert.deepEqual(fixtures.map((fixture) => fixture.id), ['threefold-knight-shuffle-history']);
  for (const fixture of fixtures) {
    const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
    const board = positions[positions.length - 1];
    const historyFens = positions.slice(0, -1).reverse().map(boardToFen);
    assert.equal(boardToFen(board), fixture.finalFen, fixture.id);
    assert.equal(automaticDrawReason(board, historyFens), fixture.expectedDraw, fixture.id);
  }
});

test('LC0 repetition history fixture sets only expected repetition planes', () => {
  const fixture = fixtures[0];
  const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
  const encoded = encodeLc0Classical112({ positions }, { historyFill: 'fen_only' });
  const expected = new Set(fixture.expectedRepetitionPlanes);
  for (let i = 0; i < 8; i++) {
    assert.equal(encoded.masks[i * LC0_PLANES_PER_HISTORY + 12], expected.has(i) ? ALL : 0n, `repetition plane slot ${i}`);
  }
});
