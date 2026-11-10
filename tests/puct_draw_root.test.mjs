import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { automaticDrawReason } from '../src/chess/drawRules.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { advanceSearchRoot, searchRoot } from '../src/search/puct.ts';

// LC0's ExtendNode only short-circuits draws by rule below the root: a root that
// is already a claimable draw is searched like any other position.

class UniformEvaluator {
  evaluate(board, context = {}) {
    const legal = context.legalMoves ?? legalMoves(board);
    const policy = new Map(legal.map((move) => [moveToActionId(move), 1 / legal.length]));
    return { policy, wdl: [0.6, 0.3, 0.1] };
  }
  evaluateBatch(boards, contexts = []) {
    return boards.map((board, i) => this.evaluate(board, contexts[i] ?? {}));
  }
}

function play(fen, ucis) {
  let board = parseFen(fen);
  const historyFens = [];
  for (const uci of ucis) {
    historyFens.unshift(boardToFen(board));
    board = makeMove(
      board,
      legalMoves(board).find((move) => moveToUci(move) === uci),
    );
  }
  return { board, historyFens };
}

function assertSearched(result, visits, label) {
  assert.ok(result.move, `${label} returns a move`);
  assert.equal(result.visits, visits, `${label} runs every visit`);
  assert.ok(result.policy.length > 0, `${label} reports root children`);
}

test('PUCT searches a root already at the fifty-move limit', async () => {
  const board = parseFen('8/8/8/4k3/8/8/3Q4/4K3 w - - 100 90');
  assert.equal(automaticDrawReason(board), 'fiftyMove');
  const result = await searchRoot(board, new UniformEvaluator(), { visits: 48 });
  assertSearched(result, 48, 'halfmove-100 root');
  // Interior nodes past the limit are still draws by rule.
  assert.ok(result.stats.terminalHits > 0, 'children past the limit stay terminal');
});

test('PUCT searches a root that is a third repetition', async () => {
  const { board, historyFens } = play(START_FEN, ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8']);
  assert.equal(automaticDrawReason(board, historyFens), 'threefold');
  const result = await searchRoot(board, new UniformEvaluator(), { visits: 32, historyFens });
  assertSearched(result, 32, 'threefold root');
});

test('PUCT searches a drawn root with restricted root moves', async () => {
  const board = parseFen('8/8/8/4k3/8/8/3Q4/4K3 w - - 100 90');
  const rootMoves = legalMoves(board).slice(0, 3);
  const result = await searchRoot(board, new UniformEvaluator(), { visits: 16, rootMoves });
  assertSearched(result, 16, 'restricted drawn root');
  assert.ok(rootMoves.some((move) => moveToUci(move) === moveToUci(result.move)));
});

test('PUCT does not reuse a child cut off as a draw as the next root', async () => {
  const start = parseFen('8/8/8/4k3/8/8/3Q4/4K3 w - - 99 90');
  const evaluator = new UniformEvaluator();
  const first = await searchRoot(start, evaluator, { visits: 200 });
  const edge = first.root.edges.find((candidate) => candidate.child?.terminalValue !== null && candidate.child?.terminalValue !== undefined);
  assert.ok(edge, 'some child reached halfmove 100 and was cut off');
  const nextBoard = edge.child.board;
  const nextHistory = [boardToFen(start)];
  const reused = advanceSearchRoot(first.root, edge.move, nextBoard, nextHistory);
  assert.equal(reused, edge.child);
  const result = await searchRoot(nextBoard, evaluator, { visits: 16, root: reused, historyFens: nextHistory });
  assertSearched(result, 16, 'advanced drawn root');
  assert.equal(result.stats.rootReused, false);
});
