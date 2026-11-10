import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parsePgnGame } from '../src/chess/pgn.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { openingNameAt, openingPositionKey } from '../src/lc0/openingNames.ts';

const index = JSON.parse(readFileSync(new URL('../public/openings/eco.json', import.meta.url), 'utf8'));
const najdorf = '1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4';

test('bundled CC0 index contains all five ECO volumes and valid position records', () => {
  assert.equal(Object.keys(index).length, 3865);
  assert.deepEqual([...new Set(Object.values(index).map(([eco]) => eco[0]))].sort(), ['A', 'B', 'C', 'D', 'E']);
  for (const [key, entry] of Object.entries(index)) {
    assert.equal(key.split(' ').length, 4);
    assert.match(entry[0], /^[A-E]\d\d$/);
    assert.ok(entry[1].length > 0);
  }
});

test('current ply gets its deepest opening, never a future name, and keeps it off book', () => {
  const { tree } = parsePgnGame(najdorf);
  assert.equal(openingNameAt(index, tree.root), null);
  tree.toEnd();
  const opening = openingNameAt(index, tree.current);
  assert.deepEqual(opening, ['B90', 'Sicilian Defense: Najdorf Variation, English Attack']);
  const standalone = new GameTree(tree.current.fen);
  // The exact off-book FEN may not be named, but a known book position is independently matchable.
  const bookNode = tree.lineTo().find((node) => index[openingPositionKey(node.fen)]?.[1].includes('English Attack'));
  assert.ok(bookNode);
  assert.deepEqual(openingNameAt(index, new GameTree(bookNode.fen).root), index[openingPositionKey(bookNode.fen)]);
  const sicilian = tree.lineTo()[2];
  assert.equal(openingNameAt(index, sicilian)?.[1], 'Sicilian Defense');
  assert.ok(tree.addUci('h2h4'));
  assert.deepEqual(openingNameAt(index, tree.current), opening);
  assert.equal(standalone.root.parent, null);
});

test('transpositions ignore clocks and resolve by position, not move sequence', () => {
  const first = parsePgnGame('1. d4 d5 2. Nf3 Nf6').tree;
  const second = parsePgnGame('1. Nf3 Nf6 2. d4 d5').tree;
  first.toEnd();
  second.toEnd();
  assert.equal(openingPositionKey(first.current.fen), openingPositionKey(second.current.fen));
  assert.deepEqual(openingNameAt(index, first.current), openingNameAt(index, second.current));
  assert.ok(openingNameAt(index, first.current));
});

test('variation ancestry never borrows the mainline opening or another game’s match', () => {
  const tree = new GameTree();
  tree.addUci('e2e4');
  const e4 = tree.current;
  tree.addUci('e7e5');
  const index = { [openingPositionKey(e4.fen)]: ['B00', 'Test mainline'] };
  assert.deepEqual(openingNameAt(index, tree.current), ['B00', 'Test mainline']);
  tree.toStart();
  tree.addUci('d2d4');
  assert.equal(openingNameAt(index, tree.current), null);
  assert.equal(openingNameAt(index, new GameTree().root), null);
});

test('EPD strips non-capturable en passant targets, including pinned pawns', () => {
  const e4 = parsePgnGame('1. e4').tree;
  e4.toEnd();
  assert.ok(openingPositionKey(e4.current.fen).endsWith(' -'));
  assert.ok(openingPositionKey('4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1').endsWith(' -'));
  assert.ok(openingPositionKey('6k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1').endsWith(' d6'));
});
