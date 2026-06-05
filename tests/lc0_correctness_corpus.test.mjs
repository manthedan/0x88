import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { LC0_MIRROR_TRANSFORM, uciToLc0PolicyIndex } from '../src/lc0/policyMap.ts';

const corpus = JSON.parse(readFileSync(new URL('../fixtures/lc0/correctness_corpus.json', import.meta.url), 'utf8'));

function replayLine(line) {
  let board = parseFen(START_FEN);
  for (const uci of line) {
    const legal = new Map(legalMoves(board).map((move) => [moveToUci(move), move]));
    const move = legal.get(uci);
    assert.ok(move, `line move ${uci} is legal from ${boardToFen(board)}`);
    board = makeMove(board, move);
  }
  return board;
}

test('LC0 broad correctness corpus has unique deterministic fixture IDs', () => {
  assert.ok(corpus.length >= 50, 'expected at least 50 broad-corpus fixtures');
  const ids = corpus.map((fixture) => fixture.id);
  assert.equal(new Set(ids).size, ids.length, 'fixture IDs are unique');
});

test('LC0 broad correctness corpus replays to its final FENs', () => {
  for (const fixture of corpus) {
    assert.equal(boardToFen(replayLine(fixture.line)), fixture.fen, fixture.id);
  }
});

test('LC0 broad correctness corpus parses, has legal moves, and maps legal moves to LC0 policy slots', () => {
  let blackToMove = 0;
  let enPassantTargets = 0;
  let castlingRights = 0;
  for (const fixture of corpus) {
    const board = parseFen(fixture.fen);
    assert.equal(boardToFen(board), fixture.fen, `${fixture.id} FEN roundtrip`);
    if (board.turn === 'b') blackToMove += 1;
    if (board.epSquare !== null) enPassantTargets += 1;
    if (board.castling.length) castlingRights += 1;
    const legal = legalMoves(board);
    assert.ok(legal.length > 0, `${fixture.id} has legal moves`);
    const transform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
    for (const move of legal) {
      const uci = moveToUci(move);
      const isStandardCastle = ['e1g1', 'e1c1', 'e8g8', 'e8c8'].includes(uci);
      assert.equal(typeof uciToLc0PolicyIndex(uci, transform, { standardCastling: isStandardCastle }), 'number', `${fixture.id} maps ${uci}`);
    }
  }
  assert.ok(blackToMove > 0, 'corpus includes black-to-move positions');
  assert.ok(enPassantTargets > 0, 'corpus includes en-passant-target positions');
  assert.ok(castlingRights > 0, 'corpus includes castling-right positions');
});
