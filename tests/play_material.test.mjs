import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { makeMove } from '../src/chess/movegen.ts';
import { sanToMove } from '../src/chess/pgn.ts';
import { playMaterial } from '../src/lc0/playMaterial.ts';

const start = parseFen(START_FEN);
function replay(sans, root = start) {
  const positions = [root];
  for (const san of sans) {
    const board = positions.at(-1);
    const move = sanToMove(board, san);
    assert.ok(move, `legal move: ${san}`);
    positions.push(makeMove(board, move));
  }
  return positions;
}

test('fresh game reserves an empty material display without claiming captures', () => {
  assert.deepEqual(playMaterial([start], 'w'), { html: '', label: 'No captures' });
});

test('a pawn capture displays the enemy glyph, material lead and accessible description', () => {
  const positions = replay(['e4', 'd5', 'exd5']);
  const material = playMaterial(positions, 'w');
  assert.match(material.html, /♟/);
  assert.match(material.html, /\+1/);
  assert.equal(material.label, 'Captured: 1 pawn; ahead by 1 material point');
  assert.equal(playMaterial(positions, 'b').html, '');
});

test('recapture shows captures on both sides but no lead; browsing the start clears them', () => {
  const positions = replay(['e4', 'd5', 'exd5', 'Qxd5']);
  assert.equal(playMaterial(positions, 'w').html, '<span class="taken">♟</span>');
  assert.equal(playMaterial(positions, 'b').html, '<span class="taken">♙</span>');
  assert.equal(playMaterial(positions, 'w', 0).html, '');
});

test('a non-capturing promotion is not a captured pawn and keeps the material lead accurate', () => {
  const root = parseFen('7k/P7/8/8/8/8/8/7K w - - 0 1');
  for (const [role, value] of [
    ['Q', 9],
    ['R', 5],
    ['B', 3],
    ['N', 3],
  ]) {
    const positions = replay([`a8=${role}`], root);
    assert.deepEqual(playMaterial(positions, 'b'), { html: '', label: 'No captures' });
    assert.equal(playMaterial(positions, 'w').label, `No captures; ahead by ${value} material points`);
  }
});

test('capturing a promoted piece shows its actual role rather than hiding it as a pawn', () => {
  const root = parseFen('1r5k/P7/8/8/8/8/8/7K w - - 0 1');
  const positions = replay(['a8=Q', 'Rxa8'], root);
  assert.deepEqual(playMaterial(positions, 'b'), {
    html: '<span class="taken">♕</span><span class="adv">+5</span>',
    label: 'Captured: 1 queen; ahead by 5 material points',
  });
  assert.deepEqual(playMaterial(positions, 'b', 1), { html: '', label: 'No captures' }, 'browsing before the capture excludes it');
  assert.deepEqual(playMaterial(positions.slice(0, 2), 'b'), { html: '', label: 'No captures' }, 'takeback excludes it');
});

test('promotion with capture records only the captured enemy piece', () => {
  const root = parseFen('1r5k/P7/8/8/8/8/8/7K w - - 0 1');
  const positions = replay(['axb8=Q'], root);
  assert.deepEqual(playMaterial(positions, 'w'), {
    html: '<span class="taken">♜</span><span class="adv">+9</span>',
    label: 'Captured: 1 rook; ahead by 9 material points',
  });
  assert.deepEqual(playMaterial(positions, 'b'), { html: '', label: 'No captures' });
});

test('en passant remains a pawn capture despite the empty destination square', () => {
  const positions = replay(['e4', 'a6', 'e5', 'd5', 'exd6']);
  assert.equal(playMaterial(positions, 'w').label, 'Captured: 1 pawn; ahead by 1 material point');
});

test('queen odds count the material lead but not a fictitious captured queen', () => {
  const odds = parseFen('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const material = playMaterial([odds], 'w');
  assert.equal(material.html, '<span class="adv">+9</span>');
  assert.equal(material.label, 'No captures; ahead by 9 material points');
});
