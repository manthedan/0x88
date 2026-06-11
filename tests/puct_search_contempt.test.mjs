import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { moveFromUci, moveToActionId } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

// Trap world: every position is comfortably winning for the side that just
// moved, EXCEPT after 1.e4 e5, where the side to move (White) is lost — e5 is
// the one refutation among Black's 20 uniform-prior replies. An unlimited
// opponent finds it and 1.e4 backs up as losing; a budget-limited opponent
// (search contempt) keeps sampling the harmless replies it found early.
const REFUTED = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w';

const trapEvaluator = {
  async evaluate(board, context) {
    const moves = context?.legalMoves ?? [];
    const uniform = moves.length ? 1 / moves.length : 0;
    const policy = new Map(moves.map((move) => [moveToActionId(move), uniform]));
    // The whole subtree behind 1.e4 e5 is lost for White (the refutation
    // sticks at any depth); everything else is dead neutral.
    const fens = [boardToFen(board), ...(context?.historyFens ?? [])];
    const refuted = fens.some((fen) => fen.startsWith(REFUTED));
    const wdl = !refuted
      ? [0.34, 0.32, 0.34]
      : board.turn === 'w' ? [0.02, 0.04, 0.94] : [0.94, 0.04, 0.02];
    return { policy, wdl };
  },
};

async function e4Quality(searchContemptLimit) {
  const board = parseFen(START_FEN);
  const result = await searchRoot(board, trapEvaluator, {
    visits: 500,
    rootMoves: [moveFromUci('e2e4')],
    searchContemptLimit,
  });
  return { q: result.policy[0].q, stats: result.stats };
}

test('search contempt keeps the trap line alive against a limited opponent', async () => {
  const plain = await e4Quality(0);
  const contempt = await e4Quality(12);
  assert.equal(plain.stats.scFrozenNodes, undefined, 'counters absent when disabled');
  assert.ok((contempt.stats.scFrozenNodes ?? 0) > 0, 'opponent nodes froze');
  assert.ok((contempt.stats.scSampledSelections ?? 0) > 0, 'frozen sampling used');
  // Unlimited opponent: e5 dominates and e4 backs up as clearly bad for us.
  assert.ok(plain.q < -0.5, `plain search should refute e4, q=${plain.q.toFixed(3)}`);
  // Limited opponent: the refutation stays a minority of the frozen mixture.
  assert.ok(contempt.q > plain.q + 0.4, `contempt should keep e4 attractive: plain=${plain.q.toFixed(3)} contempt=${contempt.q.toFixed(3)}`);
});
