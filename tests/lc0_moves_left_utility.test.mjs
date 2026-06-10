import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, parseFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

// Moves-left utility: in a decisively won position with two equal-Q,
// equal-prior candidate moves, the searcher should prefer the one whose
// subtree the moves-left head says ends sooner — but only when the effect is
// enabled. The mock evaluator makes everything symmetric except movesLeft.

const ROOT_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const SHORT_UCI = 'a2a3';
const LONG_UCI = 'h2h3';
const VISITS = 64;

function makeMockEvaluator() {
  const board = parseFen(ROOT_FEN);
  const byUci = new Map(legalMoves(board).map((move) => [moveToUci(move), move]));
  const shortFen = boardToFen(makeMove(board, byUci.get(SHORT_UCI)));
  const longFen = boardToFen(makeMove(board, byUci.get(LONG_UCI)));
  const movesLeftFor = (fen) => fen === shortFen ? 8 : fen === longFen ? 28 : 20;
  const evaluate = async (evalBoard, context) => {
    const moves = context?.legalMoves ?? legalMoves(evalBoard);
    const policy = new Map();
    const fen = boardToFen(evalBoard);
    if (fen === ROOT_FEN) {
      for (const move of moves) {
        const uci = moveToUci(move);
        policy.set(moveToActionId(move), uci === SHORT_UCI || uci === LONG_UCI ? 0.5 : 1e-6);
      }
    } else {
      for (const move of moves) policy.set(moveToActionId(move), 1 / moves.length);
    }
    // Side to move at the root is decisively winning; children mirror it.
    const wdl = evalBoard.turn === 'w' ? [0.95, 0.04, 0.01] : [0.01, 0.04, 0.95];
    return { policy, wdl, movesLeft: movesLeftFor(fen) };
  };
  return {
    evaluate,
    evaluateBatch: (boards, contexts = []) => Promise.all(boards.map((b, i) => evaluate(b, contexts[i]))),
  };
}

async function visitSplit(movesLeftMaxEffect) {
  const result = await searchRoot(parseFen(ROOT_FEN), makeMockEvaluator(), {
    visits: VISITS,
    temperature: 0,
    movesLeftMaxEffect,
  });
  const visitsOf = (uci) => result.policy.find((entry) => moveToUci(entry.move) === uci)?.visits ?? 0;
  return { short: visitsOf(SHORT_UCI), long: visitsOf(LONG_UCI), move: result.move ? moveToUci(result.move) : undefined };
}

test('moves-left utility prefers the shorter win when enabled', async () => {
  const off = await visitSplit(0);
  const on = await visitSplit(0.0345);
  // Disabled: the two symmetric candidates split visits about evenly.
  assert.ok(Math.abs(off.short - off.long) <= VISITS * 0.2,
    `expected near-even split with effect off, got short=${off.short} long=${off.long}`);
  // Enabled: the shorter win dominates visits and wins the move choice.
  assert.ok(on.short > on.long,
    `expected shorter win to receive more visits, got short=${on.short} long=${on.long}`);
  assert.equal(on.move, SHORT_UCI);
});
