import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { START_FEN } from '../src/chess/board.ts';
import { createPvPreview, insertPvPrefix } from '../src/lc0/analysisPv.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const pv = ['e2e4', 'e7e5', 'g1f3', 'b8c6'];

test('PV preview checks every move and leaves the game and source PV untouched', () => {
  const tree = new GameTree();
  tree.addUci('d2d4');
  const node = tree.current;
  const source = [...pv];
  const preview = createPvPreview(START_FEN, source);
  assert.equal(preview.steps.length, 4);
  assert.equal(preview.steps[0].label, '1. e4');
  assert.equal(preview.steps[1].label, '1… e5');
  assert.equal(preview.steps[3].label, '2… Nc6');
  assert.equal(preview.truncated, false);
  assert.equal(tree.current, node);
  assert.equal(tree.root.children.length, 1);
  source[0] = 'd2d4';
  assert.equal(preview.steps[0].uci, 'e2e4');
});

test('preview retains the full PV, including more than twelve plies', () => {
  const source = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  const preview = createPvPreview(START_FEN, Array.from({ length: 20 }, (_, i) => source[i % 4]));
  assert.equal(preview.steps.length, 20);
  assert.equal(preview.steps[19].label, '10… Ng8');
});

test('custom Black-to-move preview numbering and underpromotion are preserved', () => {
  const preview = createPvPreview('7k/8/8/8/8/8/p7/7K b - - 0 23', ['a2a1n']);
  assert.equal(preview.steps[0].label, '23… a1=N');
  assert.match(preview.steps[0].fen, /^7k\/8\/8\/8\/8\/8\/8\/n6K w - - 0 24$/);
  const tree = new GameTree(preview.rootFen);
  assert.ok(insertPvPrefix(tree, tree.root, ['a2a1n'], 1));
  assert.equal(tree.current.fen, preview.steps[0].fen);
});

test('invalid and mate-shortened tails stop at the last legal move', () => {
  assert.equal(createPvPreview(START_FEN, []).steps.length, 0);
  assert.equal(createPvPreview(START_FEN, ['a1a8']).steps.length, 0);
  const bad = createPvPreview(START_FEN, [...pv, 'a1a8', 'd2d4']);
  assert.equal(bad.steps.length, 4);
  assert.equal(bad.truncated, true);
  const mate = createPvPreview(START_FEN, ['f2f3', 'e7e5', 'g2g4', 'd8h4', 'a2a3']);
  assert.equal(mate.steps.length, 4);
  assert.equal(mate.steps[3].san, 'Qh4#');
  assert.equal(mate.truncated, true);
});

test('insertion follows existing nodes and adds a side branch without promoting it', () => {
  const tree = new GameTree();
  const main = tree.addUci('d2d4');
  tree.toStart();
  const e4 = tree.addUci('e2e4');
  const c5 = tree.addUci('c7c5');
  tree.toStart();
  const end = insertPvPrefix(tree, tree.root, pv, 3);
  assert.ok(end);
  assert.equal(tree.root.children.length, 2);
  assert.equal(tree.root.children[0], main);
  assert.equal(tree.root.children[1], e4);
  assert.equal(e4.children[0], c5);
  assert.equal(end.parent, e4.children[1]);
  tree.toStart();
  assert.equal(insertPvPrefix(tree, tree.root, pv, 3), end);
  assert.equal(e4.children.length, 2);
});

test('illegal prefixes and invalid counts are atomic', () => {
  const tree = new GameTree();
  for (const count of [0, -1, 1.5, 3, NaN]) {
    assert.equal(insertPvPrefix(tree, tree.root, ['e2e4', 'e7e5'], count), null);
    assert.equal(tree.current, tree.root);
    assert.equal(tree.root.children.length, 0);
  }
  assert.equal(insertPvPrefix(tree, tree.root, ['e2e4', 'e7e5', 'a1a8'], 3), null);
  assert.equal(tree.root.children.length, 0);
  assert.equal(tree.current, tree.root);
  assert.ok(insertPvPrefix(tree, tree.root, ['e2e4', 'e7e5', 'a1a8'], 2));
});

test('closing a preview retains its Chessground instance until page teardown', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const close = source.slice(source.indexOf('function closePvPreview()'), source.indexOf('function renderPvPreview()'));
  assert.doesNotMatch(close, /pvGround\?\.destroy\(|pvGround = null/);
  const dispose = source.slice(source.indexOf('function disposePageResources()'), source.indexOf('async function loadLc0Backend('));
  assert.match(dispose, /pvGround\?\.destroy\(\)/);
  assert.match(dispose, /pvGround = null/);
});

test('a changed root, equal-FEN replacement tree, or detached branch rejects stale insertions', () => {
  const tree = new GameTree();
  const root = tree.root;
  tree.addUci('d2d4');
  assert.equal(insertPvPrefix(tree, root, pv, 1), null);
  const replacement = new GameTree();
  assert.equal(insertPvPrefix(replacement, root, pv, 1), null);
  const detached = tree.current;
  tree.remove(detached);
  tree.goTo(detached);
  assert.equal(insertPvPrefix(tree, detached, ['d7d5'], 1), null);
  assert.equal(detached.children.length, 0);
});
