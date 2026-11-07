import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { makeMove } from '../src/chess/movegen.ts';
import { sanToMove } from '../src/chess/pgn.ts';
import { playMaterial } from '../src/lc0/playMaterial.ts';

const start = parseFen(START_FEN);
function replay(sans) {
  let board = start;
  for (const san of sans) {
    const move = sanToMove(board, san);
    assert.ok(move, `legal move: ${san}`);
    board = makeMove(board, move);
  }
  return board;
}

test('fresh game reserves an empty material display without claiming captures', () => {
  assert.deepEqual(playMaterial(start, start, 'w'), { html: '', label: 'No captures' });
});

test('a pawn capture displays the enemy glyph, material lead and accessible description', () => {
  const board = replay(['e4', 'd5', 'exd5']);
  const material = playMaterial(start, board, 'w');
  assert.match(material.html, /♟/);
  assert.match(material.html, /\+1/);
  assert.equal(material.label, 'Captured: 1 pawn; ahead by 1 material point');
  assert.equal(playMaterial(start, board, 'b').html, '');
});

test('recapture shows captures on both sides but no lead; browsing the start clears them', () => {
  const board = replay(['e4', 'd5', 'exd5', 'Qxd5']);
  assert.equal(playMaterial(start, board, 'w').html, '<span class="taken">♟</span>');
  assert.equal(playMaterial(start, board, 'b').html, '<span class="taken">♙</span>');
  assert.equal(playMaterial(start, start, 'w').html, '');
});

test('queen odds count the material lead but not a fictitious captured queen', () => {
  const odds = parseFen('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const material = playMaterial(odds, odds, 'w');
  assert.equal(material.html, '<span class="adv">+9</span>');
  assert.equal(material.label, 'No captures; ahead by 9 material points');
});
