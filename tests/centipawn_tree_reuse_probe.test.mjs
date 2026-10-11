import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { makeMove } from '../src/chess/movegen.ts';
import { UniformEvaluator } from '../src/nn/evaluator.ts';
import { squareformerCompactInput } from '../src/nn/squareformerEvaluator.ts';
import { advanceSearchRoot, chooseMove, montyLitePuctPolicy } from '../src/search/puct.ts';

// Research guardrails, not an opt-in to product reuse. Standalone fixed-visit callers stay fresh.
test('full search identity advances beyond the old 16-position caller window', async () => {
  let board = parseFen(START_FEN);
  let historyFens = [];
  let root = null;
  const evaluator = new UniformEvaluator();
  for (let ply = 0; ply < 20; ply++) {
    const inherited = root?.visits ?? 0;
    const result = await chooseMove(board, evaluator, { visits: 64, historyFens, root, searchPolicy: montyLitePuctPolicy });
    assert.equal(result.stats.rootReused, ply > 0);
    assert.equal(result.visits, 64);
    assert.equal(result.stats.completedVisits, 64 - inherited, '64 is total visits, not 64 new visits');
    const next = makeMove(board, result.move);
    historyFens = [boardToFen(board), ...historyFens];
    root = advanceSearchRoot(result.root, result.move, next, historyFens);
    assert.ok(root);
    if (historyFens.length > 16) {
      assert.equal(advanceSearchRoot(result.root, result.move, next, historyFens.slice(0, 16)), null);
    }
    board = next;
  }
});

test('same FEN with a different game history never reuses the root', async () => {
  const board = parseFen(START_FEN);
  const evaluator = new UniformEvaluator();
  const first = await chooseMove(board, evaluator, { visits: 8 });
  const unrelated = await chooseMove(board, evaluator, { visits: 8, root: first.root, historyFens: [START_FEN] });
  assert.equal(unrelated.stats.rootReused, false);
  const fresh = await chooseMove(board, evaluator, { visits: 8 });
  assert.equal(fresh.stats.rootReused, false);
  assert.equal(fresh.stats.completedVisits, 8);
});

test('repetition uses full game history independently of the neural window', async () => {
  const board = parseFen(START_FEN);
  const otherFen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
  const historyFens = [...Array(16).fill(otherFen), START_FEN, START_FEN];
  const result = await chooseMove(board, new UniformEvaluator(), { visits: 8, historyFens });
  assert.equal(result.move, null, 'two occurrences outside the neural window still cause a draw');
  const meta = { history_plies: 7, token_features: 24 };
  assert.deepEqual(squareformerCompactInput(board, meta, historyFens), squareformerCompactInput(board, meta, historyFens.slice(0, 16)));
});
