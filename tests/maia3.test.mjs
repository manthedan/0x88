import test from 'node:test';
import assert from 'node:assert/strict';

import { parseFen } from '../src/chess/board.ts';
import { boardToMaia3Tokens, chooseFromMaia3Policy, maia3InternalsForTests, maia3MoveIndex } from '../src/lc0/maia3.ts';

test('Maia3 move index matches 64x64 plus promotion layout', () => {
  assert.equal(maia3MoveIndex('a1a1'), 0);
  assert.equal(maia3MoveIndex('c2d7'), 691);
  assert.equal(maia3MoveIndex('h8h8'), 4095);
  assert.equal(maia3MoveIndex('a7a8q'), 4096);
  assert.equal(maia3MoveIndex('a7a8n'), 4099);
  assert.equal(maia3MoveIndex('h7h8n'), 4351);
  assert.equal(maia3MoveIndex('a2a1q'), undefined);
});

test('Maia3 tokens normalize black-to-move as white-to-move perspective', () => {
  const white = boardToMaia3Tokens(parseFen('8/8/8/8/8/8/P7/k6K w - - 0 1'));
  // a2 white pawn: square a2 index 8, channel 0.
  assert.equal(white[8 * 12], 1);
  // h1 white king: square h1 index 7, channel 5.
  assert.equal(white[7 * 12 + 5], 1);
  // a1 black king: square a1 index 0, channel 11.
  assert.equal(white[11], 1);

  const black = boardToMaia3Tokens(parseFen('k6K/p7/8/8/8/8/8/8 b - - 0 1'));
  // Original black pawn on a7 mirrors to a2 and becomes the side-to-move white pawn.
  assert.equal(black[8 * 12], 1);
  // Original black king on a8 mirrors to a1 and becomes a white king.
  assert.equal(black[5], 1);
  // Original white king on h8 mirrors to h1 and becomes a black king.
  assert.equal(black[7 * 12 + 11], 1);
});

test('Maia3 legal moves are mirrored back for black to move', () => {
  const board = parseFen('7k/8/8/8/8/8/1p6/4K3 b - - 0 1');
  const moves = maia3InternalsForTests.legalMaia3Moves(board);
  const promo = moves.find((entry) => entry.uci === 'b2b1q');
  assert.ok(promo);
  assert.equal(promo.modelUci, 'b7b8q');
  assert.equal(promo.index, 4132);
});

test('Maia3 choice helper supports argmax and top-p sampling pool', () => {
  const policy = [
    { uci: 'e2e4', prior: 0.7, logit: 3, index: 796 },
    { uci: 'd2d4', prior: 0.2, logit: 2, index: 731 },
    { uci: 'g1f3', prior: 0.1, logit: 1, index: 405 },
  ];
  assert.equal(chooseFromMaia3Policy(policy, { style: 'argmax' }), 'e2e4');
  for (let i = 0; i < 20; i++) {
    assert.equal(chooseFromMaia3Policy(policy, { style: 'sample', topP: 0.5 }), 'e2e4');
  }
});
