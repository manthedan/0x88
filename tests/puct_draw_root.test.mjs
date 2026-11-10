import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { automaticDrawReason } from '../src/chess/drawRules.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
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

class PawnPushEvaluator extends UniformEvaluator {
  evaluate(board, context = {}) {
    const result = super.evaluate(board, context);
    const pawnPush = (context.legalMoves ?? legalMoves(board)).find((move) => moveToUci(move) === 'e2e4');
    if (pawnPush) result.policy = new Map([[moveToActionId(pawnPush), 1]]);
    return { ...result, wdl: [1, 0, 0] };
  }
}

function drawChildCase(reason) {
  const parent =
    reason === 'fiftyMove'
      ? { board: parseFen('8/8/8/4k3/8/8/3QP3/4K3 b - - 99 90'), historyFens: [] }
      : play(START_FEN, ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']);
  const move = moveFromUci(reason === 'fiftyMove' ? 'e5e6' : 'f6g8');
  const board = makeMove(parent.board, move);
  const historyFens = [boardToFen(parent.board), ...parent.historyFens];
  assert.equal(automaticDrawReason(parent.board, parent.historyFens), null);
  assert.equal(automaticDrawReason(board, historyFens), reason);
  return { parent, move, board, historyFens };
}

test('PUCT still reuses an ordinary cached root as an interior node', async () => {
  const board = parseFen(START_FEN);
  const move = moveFromUci('g1f3');
  const evaluator = new UniformEvaluator();
  const transpositionTable = new Map();
  const previous = await searchRoot(makeMove(board, move), evaluator, { visits: 1, historyFens: [START_FEN], transpositionTable });
  const result = await searchRoot(board, evaluator, { visits: 1, rootMoves: [move], transpositionTable });
  assert.equal(result.root.edges[0].child, previous.root);
  assert.equal(result.stats.transpositionHits, 1);
});

for (const reason of ['fiftyMove', 'threefold']) {
  test(`PUCT adjudicates a cached drawn root expanded with restricted moves (${reason})`, async () => {
    const { parent, move, board, historyFens } = drawChildCase(reason);
    const evaluator = new PawnPushEvaluator();
    const transpositionTable = new Map();
    const previous = await searchRoot(board, evaluator, { visits: 1, historyFens, rootMoves: [moveFromUci('e2e4')], transpositionTable });
    assertSearched(previous, 1, 'restricted drawn root');
    const result = await searchRoot(parent.board, evaluator, { visits: 1, historyFens: parent.historyFens, rootMoves: [move], transpositionTable });
    assert.equal(Math.abs(result.value), 0);
    assert.equal(result.root.edges[0].child.terminalValue, 0);
    assert.equal(previous.root.terminalValue, null);
  });

  for (const batchSize of [1, 4]) {
    for (const useTt of [false, true]) {
      const label = `${reason}, batch ${batchSize}, TT ${useTt}`;

      test(`PUCT adjudicates a previous drawn root as an interior node (${label})`, async () => {
        const { parent, move, board, historyFens } = drawChildCase(reason);
        const evaluator = new PawnPushEvaluator();
        const transpositionTable = useTt ? new Map() : undefined;
        const previous = await searchRoot(board, evaluator, { visits: 1, historyFens, transpositionTable });
        assert.equal(previous.root.terminalValue, null, 'drawn root was expanded for search');
        const options = { visits: 1, historyFens: parent.historyFens, rootMoves: [move], batchSize };
        const fresh = await searchRoot(parent.board, evaluator, options);
        const result = await searchRoot(parent.board, evaluator, { ...options, transpositionTable });
        assert.equal(result.value, fresh.value, 'shared TT agrees with fresh draw adjudication');
        assert.equal(Math.abs(result.value), 0);
        assert.equal(result.root.edges[0].child.terminalValue, 0);
        assert.notEqual(result.root.edges[0].child, previous.root, 'root exemption stays with the previous root');
        assert.equal(previous.root.terminalValue, null, 'the previous root is not mutated');
        if (useTt) {
          const again = await searchRoot(parent.board, evaluator, { ...options, transpositionTable });
          assert.equal(again.root.edges[0].child, result.root.edges[0].child, 'ordinary interior draw remains reusable in the TT');
          assert.equal(again.stats.transpositionHits, 1);
        }
        const resumed = await searchRoot(board, evaluator, { visits: 2, historyFens, root: previous.root, transpositionTable });
        assertSearched(resumed, 2, 'resumed drawn root');
        assert.equal(resumed.stats.rootReused, true);
      });

      test(`PUCT searches an interior draw-terminal when advanced to root (${label})`, async () => {
        const { parent, move, board, historyFens } = drawChildCase(reason);
        const evaluator = new PawnPushEvaluator();
        const transpositionTable = useTt ? new Map() : undefined;
        const first = await searchRoot(parent.board, evaluator, {
          visits: 1,
          historyFens: parent.historyFens,
          rootMoves: [move],
          transpositionTable,
        });
        const child = advanceSearchRoot(first.root, move, board, historyFens);
        assert.equal(child.terminalValue, 0);
        const result = await searchRoot(board, evaluator, { visits: 8, historyFens, root: child, transpositionTable, batchSize });
        assertSearched(result, 8, 'promoted drawn root');
        assert.equal(result.stats.rootReused, false);
        assert.equal(child.terminalValue, 0, 'the old tree keeps its interior draw');
        const resumedParent = await searchRoot(parent.board, evaluator, {
          visits: 8,
          historyFens: parent.historyFens,
          root: first.root,
          transpositionTable,
          batchSize,
        });
        assert.equal(resumedParent.stats.rootReused, true);
        assert.equal(resumedParent.root.edges[0].child, child);
        assert.equal(Math.abs(resumedParent.value), 0);
      });

      test(`PUCT re-adjudicates a previous root attached inside a reused tree (${label})`, async () => {
        const { parent, move, board, historyFens } = drawChildCase(reason);
        const evaluator = new PawnPushEvaluator();
        const transpositionTable = useTt ? new Map() : undefined;
        const first = await searchRoot(parent.board, evaluator, { visits: 1, historyFens: parent.historyFens, rootMoves: [move] });
        const previous = await searchRoot(board, evaluator, { visits: 1, historyFens, transpositionTable });
        // The public Node/root API also permits a retained tree to contain a former root.
        first.root.edges[0].child = previous.root;
        const result = await searchRoot(parent.board, evaluator, {
          visits: 8,
          historyFens: parent.historyFens,
          root: first.root,
          transpositionTable,
          batchSize,
        });
        assert.equal(result.stats.rootReused, true);
        assert.equal(Math.abs(result.value), 0);
        assert.equal(result.root.edges[0].child.terminalValue, 0);
        assert.notEqual(result.root.edges[0].child, previous.root);
        assert.equal(previous.root.terminalValue, null);
      });
    }
  }
}
