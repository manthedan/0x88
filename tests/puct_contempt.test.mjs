import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { makeMove } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

// Knight-shuffle history: from the start position both sides bounce their
// kingside knights so that Black, to move, can repeat the start position for
// the third time with Ng8 (immediate threefold) or play anything else.
function shuffleHistory() {
  const ucis = ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1'];
  let board = parseFen(START_FEN);
  const fens = [boardToFen(board)];
  for (const uci of ucis) {
    board = makeMove(board, moveFromUci(uci));
    fens.push(boardToFen(board));
  }
  // Search input: current board + prior FENs (most recent first).
  return { board, historyFens: fens.slice(0, -1).reverse() };
}

// A constant near-neutral evaluator: every position is a hair better than a
// draw for the side to move, so only the drawScore moves the draw decision.
const flatEvaluator = {
  async evaluate(board, context) {
    const moves = context?.legalMoves ?? [];
    const uniform = moves.length ? 1 / moves.length : 0;
    return { policy: new Map(moves.map((move) => [moveToActionId(move), uniform])), wdl: [0.36, 0.30, 0.34] };
  },
};

async function repeatPreference(drawScore) {
  const { board, historyFens } = shuffleHistory();
  const result = await searchRoot(board, flatEvaluator, { visits: 400, historyFens, drawScore });
  const repeat = result.policy.find((entry) => moveToUci(entry.move) === 'f6g8');
  assert.ok(repeat, 'repetition move present');
  const totalVisits = result.policy.reduce((sum, entry) => sum + entry.visits, 0);
  return { share: repeat.visits / Math.max(1, totalVisits), q: repeat.q, move: result.move ? moveToUci(result.move) : undefined };
}

test('positive drawScore embraces the immediate repetition', async () => {
  const pro = await repeatPreference(0.5);
  assert.equal(pro.move, 'f6g8', `draw-seeking search should repeat (q=${pro.q.toFixed(3)}, share=${pro.share.toFixed(2)})`);
  assert.ok(pro.q > 0.4, `repetition Q should be ~ +drawScore for the root side, got ${pro.q}`);
});

test('negative drawScore avoids the immediate repetition', async () => {
  const anti = await repeatPreference(-0.5);
  assert.notEqual(anti.move, 'f6g8', 'draw-avoiding search must not repeat');
  assert.ok(anti.q < -0.4, `repetition Q should be ~ drawScore for the root side, got ${anti.q}`);
});

test('zero drawScore scores the repetition as a plain draw', async () => {
  const neutral = await repeatPreference(0);
  assert.ok(Math.abs(neutral.q) < 0.05, `repetition Q should be ~0, got ${neutral.q}`);
});
