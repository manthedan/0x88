import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { gameTreeToPgn, parsePgnGame, parsePgnGames, sanToMove, serializeMovetext } from '../src/chess/pgn.ts';

test('GameTree builds a mainline and navigates it', () => {
  const tree = new GameTree();
  for (const uci of ['e2e4', 'e7e5', 'g1f3']) assert.ok(tree.addUci(uci), `played ${uci}`);
  assert.equal(tree.current.san, 'Nf3');
  assert.equal(tree.current.ply, 3);
  assert.deepEqual(tree.mainlineFrom().map((n) => n.san), ['e4', 'e5', 'Nf3']);
  assert.equal(tree.historyBoards().length, 4, 'root + 3 plies');

  tree.back();
  assert.equal(tree.current.san, 'e5');
  tree.toStart();
  assert.equal(tree.current, tree.root);
  tree.toEnd();
  assert.equal(tree.current.san, 'Nf3');
});

test('GameTree reuses an existing child instead of duplicating', () => {
  const tree = new GameTree();
  const a = tree.addUci('e2e4');
  tree.toStart();
  const b = tree.addUci('e2e4');
  assert.equal(a, b, 'same node reused');
  assert.equal(tree.root.children.length, 1);
});

test('GameTree supports variations, promotion, and removal', () => {
  const tree = new GameTree();
  tree.addUci('e2e4');
  tree.addUci('e7e5');
  const mainline = tree.addUci('g1f3');      // mainline 2.Nf3
  tree.back();                                // back to 1...e5
  const variation = tree.addUci('f1c4');      // variation 2.Bc4
  const e5 = variation.parent;
  assert.equal(e5.children.length, 2);
  assert.equal(e5.children[0], mainline, 'Nf3 is mainline');
  assert.equal(e5.children[1], variation, 'Bc4 is the variation');

  tree.promote(variation);
  assert.equal(e5.children[0], variation, 'Bc4 promoted to mainline');

  tree.remove(variation);
  assert.equal(e5.children.length, 1);
  assert.equal(e5.children[0], mainline);
});

test('sanToMove resolves castling, promotion, en passant, and disambiguation', () => {
  assert.equal(moveToUci(sanToMove(parseFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'), 'O-O')), 'e1g1');
  assert.equal(moveToUci(sanToMove(parseFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'), 'O-O-O')), 'e1c1');
  assert.equal(moveToUci(sanToMove(parseFen('4k3/P7/8/8/8/8/8/6K1 w - - 0 1'), 'a8=Q')), 'a7a8q');
  assert.equal(moveToUci(sanToMove(parseFen('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'), 'exd6')), 'e5d6');
  assert.equal(moveToUci(sanToMove(parseFen('R6R/8/8/8/8/8/4k3/6K1 w - - 0 1'), 'Rad8')), 'a8d8');
  assert.equal(sanToMove(parseFen(START_FEN), 'Nf6'), null, 'illegal SAN -> null');
});

test('parsePgnGame reads mainline, a variation, comments, and result', () => {
  const pgn = `[Event "T"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 {good} (2. Bc4 Nc6) 2... Nc6 3. Bb5 a6 1-0`;
  const { tree, tags, result } = parsePgnGame(pgn);
  assert.equal(tags.White, 'A');
  assert.equal(result, '1-0');
  assert.deepEqual(tree.mainlineFrom().map((n) => n.san), ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6']);
  const nf3 = tree.mainlineFrom().find((n) => n.san === 'Nf3');
  assert.equal(nf3.comment, 'good');
  const e5 = nf3.parent;
  assert.equal(e5.children.length, 2, 'Nf3 plus the Bc4 variation');
  assert.equal(e5.children[1].san, 'Bc4');
  assert.equal(e5.children[1].children[0].san, 'Nc6', 'variation continues');
});

test('PGN round-trips the mainline through serialize + parse', () => {
  const pgn = `[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 *`;
  const { tree } = parsePgnGame(pgn);
  const out = gameTreeToPgn(tree, { White: 'A', Black: 'B' });
  const reparsed = parsePgnGame(out);
  assert.deepEqual(reparsed.tree.mainlineFrom().map((n) => n.san), ['d4', 'Nf6', 'c4', 'e6', 'Nc3', 'Bb4']);
  assert.match(serializeMovetext(tree), /1\. d4 Nf6 2\. c4 e6/);
});

test('parsePgnGames splits a multi-game PGN', () => {
  const pgn = `[Event "G1"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n[Event "G2"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n`;
  const games = parsePgnGames(pgn);
  assert.equal(games.length, 2);
  assert.equal(games[0].tags.Event, 'G1');
  assert.deepEqual(games[0].tree.mainlineFrom().map((n) => n.san), ['e4', 'e5']);
  assert.deepEqual(games[1].tree.mainlineFrom().map((n) => n.san), ['d4', 'd5']);
});
