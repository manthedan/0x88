import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN, squareIndex } from '../src/chess/board.ts';
import { encodeLc0Classical112, LC0_AUX_PLANE_BASE, LC0_CLASSICAL_112_PLANES, LC0_PLANES_PER_HISTORY } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';

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
  for (let i = 0; i < 8; i++) assert.equal(encoded.masks[i * 13 + 12], 0n, `synthetic history does not set repetition plane ${i}`);
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

test('LC0 classical 112 explicit move history uses current side perspective for prior boards', () => {
  const positions = buildBoardHistoryFromMoves(['e2e4', 'e7e5', 'g1f3', 'b8c6']);
  const encoded = encodeLc0Classical112({ positions }, { historyFill: 'fen_only' });
  // Current white-to-move slot after ...Nc6.
  assert.equal((encoded.masks[1] & bit('f3')) !== 0n, true, 'current white knight f3');
  assert.equal((encoded.masks[13 + 1] & bit('f3')) !== 0n, true, 'previous white knight f3 remains ours from current perspective');
  assert.equal((encoded.masks[6 * 13 + 1] & bit('g1')) !== 0n, false, 'unused history stays empty');
  // One ply back is before ...Nc6, so black knight is still on b8 in their knight plane.
  assert.equal((encoded.masks[13 + 7] & bit('b8')) !== 0n, true, 'previous black knight b8');
  assert.equal((encoded.masks[13 + 7] & bit('c6')) !== 0n, false, 'previous black knight not yet c6');
});

test('LC0 classical 112 explicit history preserves real en-passant prior boards', () => {
  const positions = buildBoardHistoryFromMoves(['e2e4', 'a7a6', 'e4e5', 'd7d5']);
  const encoded = encodeLc0Classical112({ positions }, { historyFill: 'fen_only' });
  assert.equal((encoded.masks[6] & bit('d5')) !== 0n, true, 'current black pawn d5');
  assert.equal((encoded.masks[13 + 6] & bit('d7')) !== 0n, true, 'explicit previous black pawn d7');
  assert.equal((encoded.masks[13 + 6] & bit('d5')) !== 0n, false, 'previous board has not yet pushed d-pawn');
  assert.equal((encoded.masks[13 + 0] & bit('e5')) !== 0n, true, 'one ply back includes white pawn e5');
  assert.equal((encoded.masks[2 * 13 + 0] & bit('e4')) !== 0n, true, 'two plies back includes white pawn e4');
});

test('LC0 classical 112 explicit history sets repetition planes for repeated positions', () => {
  const positions = buildBoardHistoryFromMoves(['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8']);
  const encoded = encodeLc0Classical112({ positions }, { historyFill: 'fen_only' });
  assert.equal(encoded.masks[12], ALL, 'current repeated startpos sets current repetition plane');
  assert.equal(encoded.masks[4 * 13 + 12], ALL, 'previous repeated startpos also sets its repetition plane');
  assert.equal(encoded.masks[5 * 13 + 12], 0n, 'older non-repeated position leaves repetition plane empty');
  assert.equal(planeValue(encoded, 12, 'a1'), 1, 'repetition plane is all ones');
});

test('LC0 classical 112 rejects corrupt history FENs', () => {
  assert.throws(() => encodeLc0Classical112({ positions: ['not-a-fen', START_FEN] }, { historyFill: 'fen_only' }), /Invalid FEN/);
});

const ITALIAN_FEN = 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7';

function slotMasks(encoded, slot) {
  return encoded.masks.slice(slot * LC0_PLANES_PER_HISTORY, slot * LC0_PLANES_PER_HISTORY + 12);
}

test('LC0 classical 112 fen_only fills slots past a short FEN-rooted history with the FEN root', () => {
  // LC0 encoder.cc keeps encoding history[0] once history_idx < 0, so search
  // nodes below a bare FEN root see the root in every slot the moves don't reach.
  const positions = buildBoardHistoryFromMoves(['c1g5', 'h7h6'], ITALIAN_FEN);
  const encoded = encodeLc0Classical112({ positions }, { historyFill: 'fen_only' });
  const root = slotMasks(encoded, 2);
  assert.ok(
    root.some((mask) => mask !== 0n),
    'oldest real slot is the FEN root',
  );
  for (let slot = 3; slot < 8; slot++) {
    assert.deepEqual(slotMasks(encoded, slot), root, `slot ${slot} repeats the FEN root`);
    assert.equal(encoded.masks[slot * LC0_PLANES_PER_HISTORY + 12], 0n, `slot ${slot} carries no repetition flag`);
  }
  assert.equal(encodeLc0Classical112({ positions }, { historyFill: 'no' }).masks[3 * LC0_PLANES_PER_HISTORY + 5], 0n, "'no' leaves the rest empty");
});

test('LC0 classical 112 bare FEN and single-position explicit history encode identically', () => {
  for (const fen of [
    ITALIAN_FEN,
    START_FEN,
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 6 9',
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
    'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3',
  ]) {
    for (const historyFill of ['fen_only', 'no']) {
      const bare = encodeLc0Classical112(fen, { historyFill });
      const explicit = encodeLc0Classical112({ positions: [fen] }, { historyFill });
      assert.deepEqual(explicit.masks, bare.masks, `${fen} ${historyFill}`);
      assert.deepEqual(explicit.planes, bare.planes, `${fen} ${historyFill}`);
    }
  }
});

test('LC0 classical 112 treats the startpos board as startpos regardless of move counters', () => {
  // ChessBoard == kStartposBoard ignores rule50/game ply, so LC0 does not fill.
  for (const input of [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 6 9',
    { positions: buildBoardHistoryFromMoves(['e2e4'], 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 5') },
  ]) {
    const encoded = encodeLc0Classical112(input);
    const real = typeof input === 'string' ? 1 : 2;
    for (let plane = real * LC0_PLANES_PER_HISTORY; plane < LC0_AUX_PLANE_BASE; plane++) assert.equal(encoded.masks[plane], 0n, `plane ${plane}`);
  }
  // Startpos pieces with reduced castling rights are a different board and do fill.
  const reduced = encodeLc0Classical112('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kkq - 0 1');
  assert.equal(reduced.masks[7 * LC0_PLANES_PER_HISTORY + 5], bit('e1'));
});

test("LC0 classical 112 'no' fill still writes the one pre-en-passant board LC0 can infer", () => {
  // encoder.cc: under NO, history_idx == -1 is still encoded when the oldest
  // position has an en-passant square, because the previous move is known.
  const encoded = encodeLc0Classical112('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', { historyFill: 'no' });
  // Black to move: white's pawn is "theirs" and mirrored, so e2 shows as e7.
  assert.equal((encoded.masks[6] & bit('e5')) !== 0n, true, 'current e4 pawn');
  assert.equal((encoded.masks[13 + 6] & bit('e7')) !== 0n, true, 'synthetic slot has the pawn back on e2');
  assert.equal((encoded.masks[13 + 6] & bit('e5')) !== 0n, false, 'synthetic slot has no e4 pawn');
  for (let plane = 2 * LC0_PLANES_PER_HISTORY; plane < LC0_AUX_PLANE_BASE; plane++) assert.equal(encoded.masks[plane], 0n, `plane ${plane}`);
  const positions = buildBoardHistoryFromMoves(['e7e5'], 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1');
  const history = encodeLc0Classical112({ positions }, { historyFill: 'no' });
  assert.equal((history.masks[2 * 13 + 0] & bit('e2')) !== 0n, true, 'one synthetic slot after the real FEN root');
  for (let plane = 3 * LC0_PLANES_PER_HISTORY; plane < LC0_AUX_PLANE_BASE; plane++) assert.equal(history.masks[plane], 0n, `plane ${plane}`);
});

test('LC0 classical 112 repetition flags ignore the synthetic copies of a repeated FEN root', () => {
  const positions = buildBoardHistoryFromMoves(['c3b1', 'c6b8', 'b1c3', 'b8c6'], ITALIAN_FEN);
  const encoded = encodeLc0Classical112({ positions });
  assert.equal(encoded.masks[12], ALL, 'current position repeats the root');
  assert.equal(encoded.masks[4 * 13 + 12], 0n, 'the root itself has no earlier occurrence');
  for (let slot = 5; slot < 8; slot++) {
    assert.deepEqual(slotMasks(encoded, slot), slotMasks(encoded, 4), `slot ${slot} repeats the root`);
    assert.equal(encoded.masks[slot * 13 + 12], 0n, `slot ${slot} carries no repetition flag`);
  }
});
