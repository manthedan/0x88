import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { boardToFen } from '../src/chess/board.ts';
import { automaticDrawReason, repetitionCount, repetitionKey, repetitionKeyFromFen } from '../src/chess/drawRules.ts';
import { encodeLc0Classical112, LC0_PLANES_PER_HISTORY } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';

const fixtures = JSON.parse(readFileSync(new URL('../fixtures/lc0/repetition_history.json', import.meta.url), 'utf8'));
const ALL = (1n << 64n) - 1n;

test('LC0 repetition history fixture replays to a threefold draw', () => {
  assert.deepEqual(
    fixtures.map((fixture) => fixture.id),
    ['threefold-knight-shuffle-history'],
  );
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

function finalAndHistory(moves) {
  const positions = buildBoardHistoryFromMoves(moves);
  return { positions, board: positions[positions.length - 1], historyFens: positions.slice(0, -1).reverse().map(boardToFen) };
}

test('repetition ignores an en-passant square no pawn can capture onto', () => {
  // After 1.e4 no black pawn attacks e3, so FIDE and LC0 (board.cc only records
  // a capturable en-passant square) count the position after 1.e4 as recurring.
  const { positions, board, historyFens } = finalAndHistory(['e2e4', 'g8f6', 'g1f3', 'f6g8', 'f3g1', 'g8f6', 'g1f3', 'f6g8', 'f3g1']);
  assert.equal(repetitionCount(board, historyFens), 3);
  assert.equal(automaticDrawReason(board, historyFens), 'threefold');
  assert.equal(repetitionKey(positions[1]), repetitionKey(board));
  assert.equal(boardToFen(positions[1]).split(' ')[3], 'e3', 'FEN output still reports the en-passant square');
  const encoded = encodeLc0Classical112({ positions });
  for (let i = 0; i < 8; i++) assert.equal(encoded.masks[i * LC0_PLANES_PER_HISTORY + 12], i <= 4 ? ALL : 0n, `repetition plane slot ${i}`);
});

test('repetition keeps an en-passant square a pawn can capture onto', () => {
  // After 3.e5 d5 the e5 pawn can take on d6, so that position differs from the
  // later identical boards where the en-passant right has lapsed.
  const { positions, board, historyFens } = finalAndHistory(['e2e4', 'g8f6', 'e4e5', 'd7d5', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6', 'f3g1', 'c6b8']);
  assert.notEqual(repetitionKey(positions[4]), repetitionKey(board));
  assert.equal(repetitionCount(board, historyFens), 2);
  assert.equal(automaticDrawReason(board, historyFens), null);
  assert.equal(repetitionKeyFromFen('rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3').split(' ')[3], 'd6');
  assert.equal(repetitionKeyFromFen('rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2').split(' ')[3], '-');
  assert.equal(repetitionKeyFromFen('rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3').split(' ')[3], 'd3');
  assert.equal(repetitionKeyFromFen('rnbqkbnr/pppp1ppp/8/8/P3p3/8/1PPP1PPP/RNBQKBNR b KQkq a3 0 3').split(' ')[3], '-');
});

test('repetition keeps the en-passant square written in the history root FEN', () => {
  // LC0's SetFromFen records a FEN en-passant square even when no pawn can use
  // it, so a later identical board is not a repetition of the FEN root itself.
  const root = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
  const positions = buildBoardHistoryFromMoves(['g8f6', 'g1f3', 'f6g8', 'f3g1', 'g8f6', 'g1f3', 'f6g8', 'f3g1'], root);
  const board = positions[positions.length - 1];
  const historyFens = positions.slice(0, -1).reverse().map(boardToFen);
  assert.equal(repetitionCount(board, historyFens), 2);
  assert.equal(automaticDrawReason(board, historyFens), null);
  const encoded = encodeLc0Classical112({ positions });
  assert.equal(encoded.masks[12], ALL, 'repeats the occurrence four plies back');
  assert.equal(encoded.masks[4 * LC0_PLANES_PER_HISTORY + 12], 0n, 'that occurrence does not repeat the FEN root');
});
