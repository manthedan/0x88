import assert from 'node:assert/strict';
import test from 'node:test';
import { parsePgnGame } from '../src/chess/pgn.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { reviewWinWhite, runGameReview } from '../src/lc0/gameReviewRunner.ts';

function nodesFor(pgn) {
  const { tree } = parsePgnGame(pgn);
  return [tree.root, ...tree.mainlineFrom()].map((node) => ({
    fen: node.fen,
    san: node.san,
    uci: node.move ? moveToUci(node.move) : null,
  }));
}

test('reviewWinWhite converts side-to-move engine scores to White perspective', () => {
  const white = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const black = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1';
  assert.ok(reviewWinWhite(white, { scoreCp: 100 }) > 0.5);
  assert.ok(reviewWinWhite(black, { scoreCp: 100 }) < 0.5);
  assert.equal(reviewWinWhite(white, undefined), 0.5);
});

test('runGameReview analyzes each position and reports progress', async () => {
  const nodes = nodesFor('[Result "*"]\n\n1. e4 e5 *');
  const progress = [];
  const scores = [20, -10, 30];
  const engine = {
    async analyze(_fen, options) {
      assert.equal(options.depth, 9);
      return [{ scoreCp: scores.shift(), pvUci: ['a2a3'] }];
    },
  };
  const review = await runGameReview({ nodes, engine, depth: 9, onProgress: (value) => progress.push(value.position) });
  assert.equal(review.moves.length, 2);
  assert.deepEqual(progress, [1, 2, 3]);
  assert.equal(scores.length, 0);
});

test('runGameReview respects an already-aborted signal', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    runGameReview({ nodes: nodesFor('[Result "*"]\n\n1. e4 *'), engine: { analyze: async () => [] }, depth: 8, signal: controller.signal }),
    (error) => error.name === 'AbortError',
  );
});

test('runGameReview rejects a legal position without a scored engine line', async () => {
  await assert.rejects(
    runGameReview({ nodes: nodesFor('[Result "*"]\n\n1. e4 *'), engine: { analyze: async () => [] }, depth: 8 }),
    /no scored line for position 1/,
  );
});
