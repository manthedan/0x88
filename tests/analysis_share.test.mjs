import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { positionShareUrl } from '../src/lc0/analysisShare.ts';

test('position links keep the origin and discard queries, fragments and private engine overrides', () => {
  const url = new URL(positionShareUrl(START_FEN, 'https://example.test/app/analysis/?dev&model=https://private.test/model&fen=old#private-game'));
  assert.equal(url.origin, 'https://example.test');
  assert.equal(url.pathname, '/app/analysis/');
  assert.equal(url.hash, '');
  assert.deepEqual([...url.searchParams.keys()], ['fen']);
  assert.equal(url.searchParams.get('fen'), START_FEN);
});

test('a copied current position round-trips independently from the game tree', () => {
  const tree = new GameTree();
  tree.addUci('e2e4');
  tree.addUci('c7c5');
  tree.addUci('g1f3');
  const fen = tree.current.fen;
  const url = new URL(positionShareUrl(fen, 'http://localhost:5198/app/analysis/'));
  const opened = new GameTree(url.searchParams.get('fen'));
  assert.equal(opened.current.fen, fen);
  assert.equal(opened.current.ply, 0);
  assert.equal(opened.current.children.length, 0);
  assert.equal(tree.current.ply, 3);
});

test('custom starts retain turn, castling, en passant and counters in a bounded FEN-only URL', () => {
  for (const fen of [
    'r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 42',
    '7k/8/8/8/8/8/p7/7K b - - 17 23',
  ]) {
    const url = positionShareUrl(fen, 'http://localhost:5198/app/analysis/?dev');
    assert.equal(new URL(url).searchParams.get('fen'), fen);
    assert.ok(url.length < 300);
  }
  assert.throws(() => positionShareUrl('not a fen', 'https://example.test'));
});
