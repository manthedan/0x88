import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, squareIndex, START_FEN } from '../src/chess/board.ts';
import {
  LC0_AUX_PLANE_BASE,
  LC0_CLASSICAL_112_PLANES,
  LC0_PLANES_PER_HISTORY,
  encodeLc0Classical112,
} from '../src/lc0/encoder112.ts';

const ALL = (1n << 64n) - 1n;
const bit = (name) => 1n << BigInt(squareIndex(name));
const planeValue = (encoded, plane, square) => encoded.planes[plane * 64 + squareIndex(square)];

function mask(names) {
  return names.reduce((acc, name) => acc | bit(name), 0n);
}

test('LC0 classical 112 encoder matches LC0 startpos plane expectations', () => {
  const encoded = encodeLc0Classical112(START_FEN, { historyFill: 'no' });
  assert.equal(encoded.planes.length, 112 * 64);
  assert.deepEqual(encoded.shape, [1, 112, 8, 8]);
  assert.equal(encoded.masks.length, LC0_CLASSICAL_112_PLANES);

  assert.equal(encoded.masks[0], mask(['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']));
  assert.equal(encoded.masks[1], mask(['b1', 'g1']));
  assert.equal(encoded.masks[2], mask(['c1', 'f1']));
  assert.equal(encoded.masks[3], mask(['a1', 'h1']));
  assert.equal(encoded.masks[4], bit('d1'));
  assert.equal(encoded.masks[5], bit('e1'));
  assert.equal(encoded.masks[11], bit('e8'));

  for (let i = 1; i < 8; i++) {
    for (let j = 0; j < LC0_PLANES_PER_HISTORY; j++) {
      assert.equal(encoded.masks[i * LC0_PLANES_PER_HISTORY + j], 0n, `history plane ${i}.${j}`);
    }
  }

  for (let i = 0; i < 4; i++) assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + i], ALL);
  assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + 4], 0n);
  assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + 5], ALL);
  assert.equal(encoded.values[LC0_AUX_PLANE_BASE + 5], 0);
  assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + 6], 0n);
  assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + 7], ALL);
});

test('LC0 classical 112 fen_only leaves startpos history empty', () => {
  const encoded = encodeLc0Classical112(START_FEN);
  for (let plane = LC0_PLANES_PER_HISTORY; plane < LC0_AUX_PLANE_BASE; plane++) {
    assert.equal(encoded.masks[plane], 0n, `plane ${plane}`);
  }
});

test('LC0 classical 112 black-to-move encoding is side-to-move relative', () => {
  const encoded = encodeLc0Classical112('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', { historyFill: 'no' });
  // Black's pieces are ours and rank-mirrored onto the lower side.
  assert.equal(encoded.masks[0], mask(['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']));
  assert.equal(encoded.masks[1], mask(['b1', 'g1']));
  assert.equal(encoded.masks[5], bit('e1'));
  // White's advanced e4 pawn is theirs, mirrored to e5.
  assert.equal((encoded.masks[6] & bit('e5')) !== 0n, true);
  assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + 4], ALL);
  for (let i = 0; i < 4; i++) assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + i], ALL);
});

test('LC0 classical 112 fen_only synthesizes non-start histories by repeating FEN board', () => {
  const encoded = encodeLc0Classical112('4k3/8/8/8/8/8/8/4K2R w - - 37 20');
  for (let i = 0; i < 8; i++) {
    assert.equal(encoded.masks[i * 13 + 3], bit('h1'), `rook plane slot ${i}`);
    assert.equal(encoded.masks[i * 13 + 5], bit('e1'), `our king plane slot ${i}`);
    assert.equal(encoded.masks[i * 13 + 11], bit('e8'), `their king plane slot ${i}`);
  }
  assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + 5], ALL);
  assert.equal(encoded.values[LC0_AUX_PLANE_BASE + 5], 37);
  assert.equal(planeValue(encoded, LC0_AUX_PLANE_BASE + 5, 'a1'), 37);
});

test('LC0 classical 112 castling auxiliary planes are side-relative', () => {
  const white = encodeLc0Classical112('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', { historyFill: 'no' });
  const black = encodeLc0Classical112('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', { historyFill: 'no' });
  for (const encoded of [white, black]) {
    for (let i = 0; i < 4; i++) assert.equal(encoded.masks[LC0_AUX_PLANE_BASE + i], ALL);
  }
  assert.equal(white.masks[LC0_AUX_PLANE_BASE + 4], 0n);
  assert.equal(black.masks[LC0_AUX_PLANE_BASE + 4], ALL);
});

test('LC0 classical 112 synthetic en-passant history undoes the double pawn push', () => {
  const encoded = encodeLc0Classical112('rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3');
  // Current board has black pawn on d5 in their pawn plane.
  assert.equal((encoded.masks[6] & bit('d5')) !== 0n, true);
  // Synthetic missing history moves that pawn back to d7.
  assert.equal((encoded.masks[13 + 6] & bit('d5')) !== 0n, false);
  assert.equal((encoded.masks[13 + 6] & bit('d7')) !== 0n, true);
});
