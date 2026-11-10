import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { repetitionKey, rootRepetitionKey } from '../src/chess/drawRules.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';

const ALL = (1n << 64n) - 1n;

test('encoder repetition flags match an unbounded reference across irreversible moves', () => {
  for (const [fen, moves] of [
    [START_FEN, ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'e2e4', 'b8c6', 'g1f3', 'c6b8', 'f3g1', 'g8f6', 'g1f3', 'f6g8', 'f3g1', 'd7d5', 'e4d5']],
    ['rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', ['g8f6', 'g1f3', 'f6g8', 'f3g1', 'g8f6', 'g1f3', 'f6g8', 'f3g1']],
  ]) {
    const game = buildBoardHistoryFromMoves(moves, fen);
    for (let n = 1; n <= game.length; n++) {
      const positions = game.slice(0, n);
      const keys = positions.map((board, i) => (i === 0 ? rootRepetitionKey(board) : repetitionKey(board)));
      const encoded = encodeLc0Classical112({ positions });
      for (let slot = 0; slot < Math.min(n, 8); slot++) {
        const index = n - slot - 1;
        assert.equal(encoded.masks[slot * 13 + 12], keys.slice(0, index).includes(keys[index]) ? ALL : 0n, `prefix ${n}, slot ${slot}`);
      }
    }
  }
});

test('encoder metadata describes every float plane, including zero rule50 and black perspective', () => {
  for (const fen of [START_FEN, 'r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 50', 'r3k2r/8/8/8/8/8/8/R3K2R b Kq - 37 50']) {
    const encoded = encodeLc0Classical112(fen);
    const masks = encoded.masks;
    assert.equal(encoded.masks, masks, 'metadata is stable across reads');
    for (let plane = 0; plane < 112; plane++) {
      for (let sq = 0; sq < 64; sq++) {
        assert.equal(encoded.planes[plane * 64 + sq], masks[plane] & (1n << BigInt(sq)) ? Math.fround(encoded.values[plane]) : 0);
      }
    }
    assert.equal(masks[109], ALL, 'zero-valued rule50 still has an all-squares mask');
  }
});

test('cached board repetition keys survive synchronous legality make/undo', () => {
  const board = parseFen('r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 50');
  const before = encodeLc0Classical112({ positions: [board] });
  const keys = [repetitionKey(board), rootRepetitionKey(board)];
  legalMoves(board);
  assert.deepEqual([repetitionKey(board), rootRepetitionKey(board)], keys);
  assert.deepEqual(encodeLc0Classical112({ positions: [board] }), before);
});
