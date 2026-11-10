import assert from 'node:assert/strict';
import { test } from 'node:test';
import { describePgnWarning, parseFirstPgnGame, parsePgnGame, splitPgnGames } from '../src/chess/pgn.ts';

const mainline = (game) => game.tree.mainlineFrom().map((node) => node.san);

test('tolerant PGN parsing stops the mainline at an unresolvable move instead of skipping it', () => {
  const pgn = '1. e4 e5 2. Qh5 Nc6x 3. Bc4 Nf6 4. Qxf7# 1-0';
  const game = parsePgnGame(pgn);
  assert.deepEqual(mainline(game), ['e4', 'e5', 'Qh5']);
  assert.deepEqual(game.warnings, [{ ply: 4, label: '2...', token: 'Nc6x', reason: 'invalid-move', variation: false }]);
  // The movetext result belongs to moves that were not loaded; it is not claimed.
  assert.equal(game.hasResult, false);
  assert.equal(game.result, '*');
  assert.match(describePgnWarning(game.warnings[0]), /stopped at move 2\.\.\.: Nc6x/i);
  assert.throws(() => parsePgnGame(pgn, { strict: true }), /Invalid PGN move: Nc6x/);
});

test('null moves stop a tolerant parse rather than shifting every later move to the other side', () => {
  for (const token of ['--', 'Z0', '0000', '@@@@']) {
    const game = parsePgnGame(`1. e4 e5 2. Nf3 ${token} 3. Bc4 Nc6 4. d4 *`);
    assert.deepEqual(mainline(game), ['e4', 'e5', 'Nf3'], token);
    assert.equal(game.warnings[0].reason, 'null-move', token);
    assert.equal(game.warnings[0].label, '2...', token);
  }
});

test('a stopped variation resumes the parent line after its closing parenthesis', () => {
  const game = parsePgnGame('1. e4 e5 (1... c5 2. Nf3 Zz9 (2. Nc3 Nc6) 2... d6 (2... e6)) 2. Nf3 Nc6 *');
  assert.deepEqual(mainline(game), ['e4', 'e5', 'Nf3', 'Nc6']);
  const c5 = game.tree.root.children[0].children[1];
  assert.equal(c5.san, 'c5');
  assert.deepEqual(
    game.tree.mainlineFrom(c5).map((node) => node.san),
    ['Nf3'],
  );
  assert.equal(c5.children[0].children.length, 0, 'nothing after the bad token is attached');
  assert.deepEqual(game.warnings, [{ ply: 4, label: '2...', token: 'Zz9', reason: 'invalid-move', variation: true }]);
  assert.equal(game.result, '*');
  assert.equal(game.hasResult, true);
});

test('pasted games glued together stop at the first result instead of merging', () => {
  const alice = '[White "Alice"]\n[Black "Bob"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0';
  const carol = '[White "Carol"]\n[Black "Dave"]\n[Result "0-1"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1';
  const game = parsePgnGame(`${alice}\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1`);
  assert.deepEqual(mainline(game), ['e4', 'e5', 'Nf3', 'Nc6']);
  assert.equal(game.result, '1-0');
  assert.equal(game.warnings[0].reason, 'after-result');
  assert.equal(game.warnings[0].token, 'd4');

  const first = parseFirstPgnGame(`${alice}\n\n${carol}\n`);
  assert.equal(first.gameCount, 2);
  assert.equal(first.game.tags.White, 'Alice');
  assert.equal(first.game.result, '1-0');
  assert.deepEqual(mainline(first.game), ['e4', 'e5', 'Nf3', 'Nc6']);
  assert.deepEqual(first.game.warnings, []);
});

test('interactive loading rejects text with no tags and no moves', () => {
  for (const text of ['This is not a chess game at all.', '%escape only', '*', '{just a comment}']) {
    assert.throws(() => parseFirstPgnGame(text), /PGN/, text);
  }
  assert.throws(() => parseFirstPgnGame('Hello 1. e4'), /stopped at move 1\.: Hello/i);
  assert.throws(() => parseFirstPgnGame('   \n'), /PGN/);
  // Tags alone (e.g. a set-up position) or a truncated game with tags still load.
  assert.equal(parseFirstPgnGame('[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/8/4K3 b - - 0 1"]\n\n*').game.tree.root.fen, '4k3/8/8/8/8/8/8/4K3 b - - 0 1');
  const truncated = parseFirstPgnGame('[White "A"]\n\n1. Nf6 *');
  assert.equal(truncated.game.tree.root.children.length, 0);
  assert.equal(truncated.game.warnings[0].label, '1.');
});

test('tolerant parsing still ignores annotation noise that is not a move', () => {
  const pgn = [
    '% escape line Qxf7',
    '[Event "noise"]',
    '',
    '{Intro with [brackets] (parens) ; and a semicolon} 1.e4 $1 e5!? 2. Nf3 ! ; rest of line Bb5',
    '2... Nc6 3. Bc4 { (not a variation) } 3. ... Bc5 4. 0-0 Nf6 5. d3 ?! d6 6. c3 +/= a6 7. a4 ±',
    '% another escape 1-0',
    '7... Ba7 8. h3 (8. Bg5 h6 0-1) 8… h6 9..Re1 O-O *',
  ].join('\n');
  const game = parsePgnGame(pgn);
  assert.deepEqual(game.warnings, []);
  assert.deepEqual(mainline(game), ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'O-O', 'Nf6', 'd3', 'd6', 'c3', 'a6', 'a4', 'Ba7', 'h3', 'h6', 'Re1', 'O-O']);
  assert.equal(game.tags.Event, 'noise');
  assert.equal(game.result, '*');
  assert.equal(game.hasResult, true);
  assert.deepEqual(mainline(parsePgnGame('1. e4 a6 2. e5 d5 3. exd6 e.p. Nf6 *')), ['e4', 'a6', 'e5', 'd5', 'exd6', 'Nf6']);
});

test('black-to-move start positions number their first move and stop labels with an ellipsis', () => {
  const fen = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3';
  const game = parsePgnGame(`[SetUp "1"]\n[FEN "${fen}"]\n\n3... Nf6 4. Ng5 Bb4x *`);
  assert.deepEqual(mainline(game), ['Nf6', 'Ng5']);
  assert.equal(game.warnings[0].label, '4...');
  assert.equal(game.warnings[0].ply, 3);
});

test('escape lines do not split a collection into an extra game', () => {
  assert.equal(splitPgnGames('% exported by tool\n[Event "a"]\n\n1. e4 *\n\n% next\n[Event "b"]\n\n1. d4 *').length, 2);
});
