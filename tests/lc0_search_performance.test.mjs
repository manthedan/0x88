import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { repetitionKey, rootRepetitionKey } from '../src/chess/drawRules.ts';
import { moveToActionId } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { prepareLc0EvaluatorInput } from '../src/lc0/onnxEvaluator.ts';
import { advanceSearchRoot, searchRoot } from '../src/search/puct.ts';

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

class StableEvaluator {
  evaluate(board, context = {}) {
    const moves = context.legalMoves ?? legalMoves(board);
    return {
      policy: new Map(moves.map((move, i) => [moveToActionId(move), i === 0 ? 1 : 0])),
      wdl: [0, 1, 0],
      movesLeft: 60,
      timing: { inputBuildMs: 1, totalEvalMs: 2 },
    };
  }
  evaluateBatch(boards, contexts = []) {
    return boards.map((board, i) => this.evaluate(board, contexts[i]));
  }
}

test('expanded nodes retain WDL for tree reuse without retaining policy or timing', async () => {
  const evaluator = new StableEvaluator();
  const first = await searchRoot(parseFen(START_FEN), evaluator, { visits: 32 });
  const seen = new Set();
  function inspect(node) {
    if (seen.has(node)) return;
    seen.add(node);
    if (node.evaluation) assert.deepEqual(Object.keys(node.evaluation), ['wdl']);
    for (const edge of node.edges) if (edge.child) inspect(edge.child);
  }
  inspect(first.root);
  const edge = first.root.edges.find((edge) => edge.child?.evaluation);
  assert.ok(edge);
  const child = edge.child;
  const root = advanceSearchRoot(first.root, edge.move, child.board, child.historyFens);
  assert.equal(root, child);
  const next = await searchRoot(child.board, evaluator, { visits: 40, root, historyFens: child.historyFens, historyBoards: child.historyBoards });
  assert.equal(next.stats.rootReused, true);
  assert.deepEqual(next.root.evaluation, { wdl: [0, 1, 0] });
  assert.ok(first.stats.evalBackendTimingSamples > 0, 'timing is still aggregated before dropping it');
});

test('prepared cache keys bound irrelevant ancient history while keeping recent and short histories', () => {
  const positions = buildBoardHistoryFromMoves([
    ...Array(4).fill(['g1f3', 'g8f6', 'f3g1', 'f6g8']).flat(),
    'e2e4',
    'e7e5',
    'd2d4',
    'd7d5',
    'c2c4',
    'c7c5',
    'b2b4',
    'b7b5',
    'a2a4',
  ]);
  function prepare(positions) {
    return prepareLc0EvaluatorInput(positions.at(-1), positions.slice(0, -1).reverse());
  }
  const original = prepare(positions);
  const ancientChanged = prepare([{ ...positions[0], fullmove: 99 }, ...positions.slice(1)]);
  assert.deepEqual(encodeLc0Classical112(original).planes, encodeLc0Classical112(ancientChanged).planes);
  assert.equal(original.prepared.cacheKey, ancientChanged.prepared.cacheKey);
  assert.ok(original.prepared.cacheKey.length < positions.map(boardToFen).join('\n').length / 2);
  const recent = positions.slice();
  recent[recent.length - 2] = { ...recent.at(-2), halfmove: 3 };
  assert.notEqual(prepare(recent).prepared.cacheKey, original.prepared.cacheKey);
  const shortA = prepare(positions.slice(-3));
  const differentOldest = positions.slice(-3);
  differentOldest[0] = positions[0];
  const shortB = prepare(differentOldest);
  assert.notEqual(shortA.prepared.cacheKey, shortB.prepared.cacheKey, 'oldest short-history board controls synthetic fill');
  assert.notDeepEqual(encodeLc0Classical112(shortA).planes, encodeLc0Classical112(shortB).planes);
});

test('shared search history exposes complete newest-first arrays with and without board context', async () => {
  const positions = buildBoardHistoryFromMoves(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5']);
  const board = positions.at(-1);
  const historyFens = positions.slice(0, -1).reverse().map(boardToFen);
  for (const historyBoards of [positions.slice(0, -1).reverse(), []]) {
    const result = await searchRoot(board, new StableEvaluator(), { visits: 40, batchSize: 4, historyFens, historyBoards });
    function inspect(node, fens, boards) {
      assert.deepEqual(node.historyFens, fens);
      assert.deepEqual(node.historyBoards, boards);
      for (const edge of node.edges) {
        if (edge.child) inspect(edge.child, [boardToFen(node.board), ...fens], [node.board, ...boards]);
      }
    }
    inspect(result.root, historyFens, historyBoards);
  }
});

test('prepared cache keys keep repetition history for slots before a recent pawn move', () => {
  const fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
  const moves = ['g8f6', 'g1f3', 'f6g8', 'f3g1', 'g8f6', 'g1f3', 'f6g8', 'f3g1', 'e7e5'];
  const positions = buildBoardHistoryFromMoves(moves, fen);
  const changed = [{ ...positions[0], epSquare: null }, ...positions.slice(1)];
  const prepare = (boards) => prepareLc0EvaluatorInput(boards.at(-1), boards.slice(0, -1).reverse());
  const a = prepare(positions);
  const b = prepare(changed);
  assert.equal(positions.at(-1).halfmove, 0);
  assert.notDeepEqual(encodeLc0Classical112(a).planes, encodeLc0Classical112(b).planes);
  assert.notEqual(a.prepared.cacheKey, b.prepared.cacheKey, 'previous slot still needs the original FEN-root en-passant key');
});

function stableStopOptions(earlyStop) {
  return {
    visits: 400,
    earlyStop,
    kldMinVisits: 50,
    kldCheckInterval: 50,
    kldThreshold: 1,
    kldStableChecks: 1,
    bestStableMinVisits: 50,
    bestStableCheckInterval: 50,
    bestStableChecks: 1,
    bestStableMinVisitLead: 0,
    bestStableMaxQDelta: 2,
  };
}

for (const earlyStop of ['kld-stable', 'best-stable']) {
  for (const [batchSize, batchPipelineDepth, reusedVisits, expected] of [
    [16, 1, 0, 112],
    [16, 2, 0, 128],
    [16, 1, 7, 103],
  ]) {
    test(`${earlyStop} checks crossed intervals with batch ${batchSize}, pipeline ${batchPipelineDepth}, reused ${reusedVisits}`, async () => {
      const board = parseFen(START_FEN);
      const evaluator = new StableEvaluator();
      const root = reusedVisits ? (await searchRoot(board, evaluator, { visits: reusedVisits })).root : undefined;
      const result = await searchRoot(board, evaluator, { ...stableStopOptions(earlyStop), batchSize, batchPipelineDepth, root });
      assert.equal(result.stats.rootReused, !!reusedVisits);
      assert.equal(result.stats.stopReason, earlyStop);
      assert.equal(result.visits, expected);
    });
  }
  test(`${earlyStop} serial checks preserve absolute multiples, minimum visits and reused-root timing`, async () => {
    for (const minVisits of [50, 64]) {
      for (const reusedVisits of [0, 7, 57]) {
        const board = parseFen(START_FEN);
        const evaluator = new StableEvaluator();
        const root = reusedVisits ? (await searchRoot(board, evaluator, { visits: reusedVisits })).root : undefined;
        const result = await searchRoot(board, evaluator, {
          ...stableStopOptions(earlyStop),
          root,
          kldMinVisits: minVisits,
          bestStableMinVisits: minVisits,
        });
        assert.equal(result.stats.stopReason, earlyStop);
        const firstCheck = Math.max(Math.ceil(minVisits / 50), Math.floor(reusedVisits / 50) + 1) * 50;
        assert.equal(result.visits, firstCheck + 50);
      }
    }
  });
}
