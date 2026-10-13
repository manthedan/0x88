import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { parseValidatedFen, validateFenPosition } from '../src/chess/fenValidation.ts';
import { parsePgnGame } from '../src/chess/pgn.ts';
import { gameOutcome } from '../src/lc0/engineBattle.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const BLACK_CHECKED = '4k3/8/8/8/8/8/8/K3R3 w - - 0 1';
const WHITE_CHECKED = '4k2r/8/8/8/8/8/8/7K b - - 0 1';

for (const [fen, color] of [
  [BLACK_CHECKED, 'black'],
  [WHITE_CHECKED, 'white'],
]) {
  test(`FEN position validation rejects the ${color} king in check when not to move`, () => {
    assert.throws(() => parseValidatedFen(fen), new RegExp(`${color} king is in check, but that side is not to move`));
    assert.doesNotThrow(() => parseFen(fen), 'the low-level syntax parser remains permissive');
  });
}

test('FEN position validation rejects adjacent kings and a checked opponent attacked by a knight or pawn', () => {
  for (const fen of ['8/8/8/8/8/8/4k3/4K3 w - - 0 1', '4k3/8/5N2/8/8/8/8/K7 w - - 0 1', '4k3/3P4/8/8/8/8/8/K7 w - - 0 1']) {
    assert.throws(() => parseValidatedFen(fen), /black king is in check, but that side is not to move/);
  }
});

test('FEN position validation requires exactly one king of each color', () => {
  for (const [placement, color, count] of [
    ['8/8/8/8/8/8/8/4K3', 'black', 0],
    ['4k3/8/8/8/8/8/8/8', 'white', 0],
    ['4k3/8/8/8/8/8/8/K3K3', 'white', 2],
    ['k3k3/8/8/8/8/8/8/4K3', 'black', 2],
  ]) {
    assert.throws(() => parseValidatedFen(`${placement} w - - 0 1`), new RegExp(`exactly one ${color} king \\(found ${count}\\)`));
  }
});

test('FEN position validation rejects pawns of either color on either back rank', () => {
  for (const placement of ['4k3/8/8/8/8/8/8/P3K3', '4k3/8/8/8/8/8/8/p3K3', 'P3k3/8/8/8/8/8/8/4K3', 'p3k3/8/8/8/8/8/8/4K3']) {
    assert.throws(() => parseValidatedFen(`${placement} w - - 0 1`), /pawns cannot be on rank 1 or 8/);
  }
});

test('FEN position validation preserves legal checks, blocked attacks, odds, and abbreviated FENs without mutation', () => {
  for (const fen of [
    START_FEN,
    BLACK_CHECKED.replace(' w ', ' b '),
    WHITE_CHECKED.replace(' b ', ' w '),
    '4k3/8/8/4p3/8/8/8/K3R3 w - - 0 1',
    '4k3/8/8/8/8/8/8/4K2R w - - 0 1', // h1 rook does not attack e8.
    START_FEN.replace('RNBQKBNR', 'RNB1KBNR'),
    '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
    START_FEN.split(' ').slice(0, 4).join(' '),
  ]) {
    const board = parseFen(fen);
    const before = structuredClone(board);
    validateFenPosition(board);
    assert.deepEqual(board, before);
    assert.deepEqual(parseValidatedFen(fen), board);
  }
});

test('FEN position validation accepts checkmate and stalemate roots', () => {
  const mate = parseValidatedFen('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1');
  const stalemate = parseValidatedFen('7k/5Q2/5K2/8/8/8/8/8 b - - 0 1');
  assert.equal(gameOutcome(mate).reason, 'checkmate');
  assert.equal(gameOutcome(stalemate).reason, 'stalemate');
});

test('Analysis game-tree and PGN setup roots share FEN position validation', () => {
  for (const fen of [BLACK_CHECKED, '8/8/8/8/8/8/8/4K3 w - - 0 1', '4k3/8/8/8/8/8/8/P3K3 w - - 0 1']) {
    assert.throws(() => new GameTree(fen), /Invalid FEN/);
    assert.throws(() => parsePgnGame(`[SetUp "1"]\n[FEN "${fen}"]\n\n*`), /Invalid FEN/);
  }
  const legal = BLACK_CHECKED.replace(' w ', ' b ');
  assert.equal(new GameTree(legal).root.fen, legal);
  const { tree } = parsePgnGame(`[SetUp "1"]\n[FEN "${legal}"]\n\n1... Kd7 *`);
  assert.equal(tree.root.fen, legal);
  assert.equal(tree.root.children[0].san, 'Kd7');
  assert.equal(boardToFen(parseValidatedFen(tree.root.children[0].fen)), tree.root.children[0].fen);
});
