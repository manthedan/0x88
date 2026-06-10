import assert from 'node:assert/strict';
import { test } from 'node:test';
import { annotatedPgn, classifyMoverLoss, moveAccuracy, reviewGame } from '../src/lc0/gameReview.ts';

test('classification thresholds and overrides', () => {
  assert.equal(classifyMoverLoss(0, false, false), 'best');
  assert.equal(classifyMoverLoss(0.004, false, false), 'best');
  assert.equal(classifyMoverLoss(0.03, false, false), 'good');
  assert.equal(classifyMoverLoss(0.07, false, false), 'inaccuracy');
  assert.equal(classifyMoverLoss(0.15, false, false), 'mistake');
  assert.equal(classifyMoverLoss(0.25, false, false), 'blunder');
  // Playing the engine move is 'best' regardless of measured loss (eval noise).
  assert.equal(classifyMoverLoss(0.15, true, false), 'best');
  assert.equal(classifyMoverLoss(0.15, false, true), 'forced');
});

test('moveAccuracy follows the lichess curve anchors', () => {
  assert.ok(moveAccuracy(0) > 99);
  assert.ok(Math.abs(moveAccuracy(0.1) - (103.1668 * Math.exp(-0.4354) - 3.1669)) < 1e-9);
  assert.equal(moveAccuracy(2), moveAccuracy(1));
  assert.ok(moveAccuracy(1) >= 0);
});

test('reviewGame computes per-side losses, classes, and accuracy', () => {
  // White blunders on move 1 (60% -> 30%), black returns it (30% -> 55%).
  const positions = [
    { winWhite: 0.6, bestUci: 'e2e4' },
    { winWhite: 0.3, bestUci: 'd7d5' },
    { winWhite: 0.55, bestUci: 'g1f3' },
  ];
  const moves = [
    { san: 'a3', uci: 'a2a3' },
    { san: 'h6', uci: 'h7h6' },
  ];
  const review = reviewGame(positions, moves);
  assert.equal(review.moves[0].side, 'w');
  assert.ok(Math.abs(review.moves[0].moverLoss - 0.3) < 1e-9);
  assert.equal(review.moves[0].class, 'blunder');
  assert.equal(review.moves[1].side, 'b');
  assert.ok(Math.abs(review.moves[1].moverLoss - 0.25) < 1e-9);
  assert.equal(review.moves[1].class, 'blunder');
  assert.equal(review.counts.white.blunder, 1);
  assert.equal(review.counts.black.blunder, 1);
  assert.ok(review.accuracy.white < 50 && review.accuracy.black < 50);
  assert.deepEqual(review.criticalMoves.map((m) => m.ply), [1, 2]);
});

test('reviewGame honors best-move matches, forced moves, and black-to-move starts', () => {
  const positions = [
    { winWhite: 0.5, bestUci: 'e7e5' },
    { winWhite: 0.5, bestUci: 'g1f3', legalMoves: 1 },
    { winWhite: 0.48, bestUci: null },
  ];
  const moves = [
    { san: 'e5', uci: 'e7e5' },
    { san: 'Nf3', uci: 'g1f3' },
  ];
  const review = reviewGame(positions, moves, 'b');
  assert.equal(review.moves[0].side, 'b');
  assert.equal(review.moves[0].class, 'best');
  assert.equal(review.moves[1].side, 'w');
  assert.equal(review.moves[1].class, 'forced');
  // Forced-only side still reports an accuracy (defaults to 100).
  assert.equal(review.accuracy.white, 100);
  assert.equal(review.criticalMoves.length, 0);
});

test('criticalMoves excludes best/forced moves even with large measured swings', () => {
  // Mate delivered: huge nominal swing but the engine best move was played.
  const review = reviewGame([
    { winWhite: 0.0, bestUci: 'd4f3' },
    { winWhite: 0.0, bestUci: null },
  ], [{ san: 'Nf3#', uci: 'd4f3' }], 'b');
  assert.equal(review.moves[0].class, 'best');
  assert.equal(review.criticalMoves.length, 0);
});

test('reviewGame validates position/move count alignment', () => {
  assert.throws(() => reviewGame([{ winWhite: 0.5, bestUci: null }], [{ san: 'e4', uci: 'e2e4' }, { san: 'e5', uci: 'e7e5' }]), /one position per move/);
});

test('annotatedPgn emits NAGs, win comments, and best-move notes', () => {
  const review = reviewGame([
    { winWhite: 0.55, bestUci: 'e2e4' },
    { winWhite: 0.3, bestUci: 'd7d5' },
    { winWhite: 0.3, bestUci: 'b1c3' },
  ], [
    { san: 'a3', uci: 'a2a3' },
    { san: 'd5', uci: 'd7d5' },
  ]);
  const pgn = annotatedPgn(review, { tags: { White: 'A', Black: 'B' }, result: '0-1' });
  assert.ok(pgn.includes('[White "A"]'));
  assert.ok(pgn.includes('1. a3??'));
  assert.ok(pgn.includes('$4'));
  assert.ok(pgn.includes('best e2e4'));
  assert.ok(pgn.includes('[%win 30%]'));
  assert.ok(pgn.trim().endsWith('0-1'));
  // Black's best reply gets no suffix and no best-move note.
  assert.ok(pgn.includes(' d5 '));
});
