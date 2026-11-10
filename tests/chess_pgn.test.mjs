import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { describePgnWarning, gameTreeToPgn, parseFirstPgnGame, parsePgnGame, sanToMove, serializeMovetext, splitPgnGames } from '../src/chess/pgn.ts';
import { moveToSan } from '../src/chess/san.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

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
  for (const word of ['constructor', 'toString', '__proto__', 'hasOwnProperty']) {
    assert.equal(parsePgnGame(`1. e4 ${word} 2. Nf3 *`).warnings[0]?.token, word, word);
  }
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
  assert.equal(splitPgnGames('[Event "a"]\n\n1. e4 [%clk 0:03:00] e5 *\n\n[Event "b"]\n\n1. d4 *').length, 2);
});

test('sanToMove accepts lenient real-world forms that identify exactly one legal move', () => {
  const uci = (fen, san) => {
    const move = sanToMove(parseFen(fen), san);
    return move && moveToUci(move);
  };
  const promo = '8/P6k/8/8/8/8/8/K7 w - - 0 1';
  assert.equal(uci(promo, 'a8Q'), 'a7a8q');
  assert.equal(uci(promo, 'a8=Q+'), 'a7a8q');
  assert.equal(uci(promo, 'a8N'), 'a7a8n');
  assert.equal(uci(promo, 'a7a8q'), 'a7a8q');
  assert.equal(uci(promo, 'a7-a8=R'), 'a7a8r');
  assert.equal(uci(START_FEN, 'e2e4'), 'e2e4');
  assert.equal(uci(START_FEN, 'e2-e4'), 'e2e4');
  assert.equal(uci(START_FEN, 'g1f3'), 'g1f3');
  assert.equal(uci(START_FEN, 'Ng1f3'), 'g1f3');
  assert.equal(uci(START_FEN, 'Ng1-f3'), 'g1f3');
  assert.equal(uci(START_FEN, 'Ngf3'), 'g1f3');
  assert.equal(uci(START_FEN, 'N1f3'), 'g1f3');
  assert.equal(uci(START_FEN, 'Nf3!?'), 'g1f3');
  const afterE4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
  assert.equal(uci(afterE4, 'Nbc6'), 'b8c6', 'unnecessary disambiguator');
  assert.equal(uci(afterE4, 'Nb8c6'), 'b8c6');
  const castles = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1';
  for (const [san, expected] of [
    ['O-O', 'e1g1'],
    ['0-0', 'e1g1'],
    ['O-O+', 'e1g1'],
    ['O-O-O', 'e1c1'],
    ['0-0-0#', 'e1c1'],
    ['e1g1', 'e1g1'],
  ])
    assert.equal(uci(castles, san), expected, san);
  assert.equal(uci('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'exd6e.p.'), 'e5d6');
  assert.equal(uci('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'ed6'), 'e5d6');
  assert.equal(uci('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6'), 'e5d6');
  assert.equal(uci('4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1', 'exd5'), 'e4d5');
});

test('sanToMove still rejects ambiguous, illegal and contradictory tokens', () => {
  const none = (fen, san) => assert.equal(sanToMove(parseFen(fen), san), null, `${san} in ${fen}`);
  const knights = '7k/8/8/8/8/2N1N3/8/7K w - - 0 1';
  none(knights, 'Nd5'); // two knights reach d5
  none(knights, 'Nd2f3'); // no knight on d2
  none('8/P6k/8/8/8/8/8/K7 w - - 0 1', 'a8'); // promotion piece missing: four legal moves
  none('8/P6k/8/8/8/8/8/K7 w - - 0 1', 'a8=K');
  none(START_FEN, 'Nf6');
  none(START_FEN, 'e5');
  none(START_FEN, 'Nxf3'); // claims a capture on an empty square
  none(START_FEN, 'e4=Q');
  none(START_FEN, 'O-O');
  none(START_FEN, 'Bb5');
  none(START_FEN, 'nf3'); // piece letters are upper case; `b` is a file
  none(START_FEN, 'e2e5');
  none(START_FEN, 'e9');
  none(START_FEN, '');
  none(START_FEN, 'Zz9');
  none(START_FEN, '--');
  // A pawn token without a from-file is a push; it must not turn into an en-passant capture.
  none('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'd6');
  none('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'xd6');
  // bxc3 is a pawn capture, Bxc3 a bishop capture; neither may be read as the other.
  const fileB = 'rnbqkbnr/pppppppp/8/8/8/2p5/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  assert.equal(moveToUci(sanToMove(parseFen(fileB), 'bxc3')), 'b2c3');
  none(fileB, 'Bxc3');
});

test('ambiguous tokens stop a tolerant parse with their own reason', () => {
  const game = parsePgnGame('[SetUp "1"]\n[FEN "7k/8/8/8/8/2N1N3/8/7K w - - 0 1"]\n\n1. Nd5 *');
  assert.equal(game.warnings[0].reason, 'ambiguous-move');
  assert.throws(() => parsePgnGame('[SetUp "1"]\n[FEN "7k/8/8/8/8/2N1N3/8/7K w - - 0 1"]\n\n1. Nd5 *', { strict: true }), /Invalid PGN move: Nd5/);
});

test('sanToMove resolves every legal move from its canonical SAN in a sweep of random games', () => {
  let seed = 11;
  const rand = (n) => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed % n;
  };
  for (const start of [START_FEN, 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', '8/2P1k3/8/8/8/8/1p2K3/8 w - - 0 1']) {
    for (let game = 0; game < 6; game++) {
      let board = parseFen(start);
      for (let ply = 0; ply < 80; ply++) {
        const moves = legalMoves(board);
        if (!moves.length) break;
        for (const move of moves) {
          const san = moveToSan(board, move);
          assert.equal(moveToUci(sanToMove(board, san)), moveToUci(move), `${san} in ${boardToFen(board)}`);
          assert.equal(moveToUci(sanToMove(board, moveToUci(move))), moveToUci(move), `${moveToUci(move)} in ${boardToFen(board)}`);
        }
        board = makeMove(board, moves[rand(moves.length)]);
      }
    }
  }
});

const shape = (node) => ({
  san: node.san,
  ...(node.beforeComment ? { beforeComment: node.beforeComment } : {}),
  ...(node.comment ? { comment: node.comment } : {}),
  ...(node.nags?.length ? { nags: node.nags } : {}),
  children: node.children.map(shape),
});
const countNodes = (node) => 1 + node.children.reduce((sum, child) => sum + countNodes(child), 0);
const roundTrip = (tree) => parsePgnGame(gameTreeToPgn(tree, tree.root.fen === START_FEN ? {} : { SetUp: '1', FEN: tree.root.fen }));

test('export keeps nested variations at every depth with correct move numbers', () => {
  const src = '1. e4 e5 (1... c5 2. Nf3 (2. Nc3 Nc6 (2... e6)) 2... d6) 2. Nf3 *';
  const { tree } = parsePgnGame(src);
  assert.equal(countNodes(tree.root), 10);
  assert.equal(serializeMovetext(tree), '1. e4 e5 (1... c5 2. Nf3 (2. Nc3 Nc6 (2... e6)) 2... d6) 2. Nf3');
  const again = roundTrip(tree);
  assert.equal(countNodes(again.tree.root), 10);
  assert.deepEqual(shape(again.tree.root), shape(tree.root));
});

test('export keeps the pre-game comment and NAGs and re-numbers black after a comment', () => {
  const { tree } = parsePgnGame('{Opening comment} 1. e4 $1 {good} e5 $2 2. Nf3 !? Nc6?! (2... d6!) 3. Bb5 *');
  assert.equal(tree.root.comment, 'Opening comment');
  const [e4, e5, nf3, nc6] = tree.mainlineFrom();
  assert.deepEqual(e4.nags, [1]);
  assert.deepEqual(e5.nags, [2]);
  assert.deepEqual(nf3.nags, [5]);
  assert.deepEqual(nc6.nags, [6]);
  assert.deepEqual(nf3.children[1].nags, [1]);
  assert.equal(serializeMovetext(tree), '{Opening comment} 1. e4 $1 {good} 1... e5 $2 2. Nf3 $5 Nc6 $6 (2... d6 $1) 3. Bb5');
  assert.deepEqual(shape(roundTrip(tree).tree.root), shape(tree.root));
});

test('export numbers a black-to-move start and keeps comments with PGN punctuation intact', () => {
  const fen = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3';
  const { tree } = parsePgnGame(`[SetUp "1"]\n[FEN "${fen}"]\n\n3... Nf6 {a ( b ) ; c [%clk 0:01:00]} (3... Bc5 4. c3) 4. Ng5 *`);
  assert.equal(serializeMovetext(tree), '3... Nf6 {a ( b ) ; c [%clk 0:01:00]} (3... Bc5 4. c3) 4. Ng5');
  assert.deepEqual(shape(roundTrip(tree).tree.root), shape(tree.root));
});

test('PGN round-trips random variation trees (seeded property test)', () => {
  let seed = 20261109;
  const rand = (n) => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed % n;
  };
  const COMMENTS = [
    'plan',
    '[Site "y"]',
    '[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]',
    'a (b) c',
    'x ; y',
    '[%eval 0.31]',
    'line\nbreak',
    '1. e4 is not a move here',
    '$1 ! ?',
    'tab\tand  spaces',
  ];
  const STARTS = [START_FEN, 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3', '8/2P1k3/8/8/8/8/1p2K3/8 w - - 0 40'];
  for (let round = 0; round < 40; round++) {
    const tree = new GameTree(STARTS[round % STARTS.length]);
    if (rand(3) === 0) tree.root.comment = COMMENTS[rand(COMMENTS.length)];
    const grow = (node, depth, budget) => {
      if (depth >= budget) return;
      const moves = legalMoves(parseFen(node.fen));
      if (!moves.length) return;
      const branches = depth > 0 && rand(4) === 0 ? 1 + rand(3) : 1;
      for (let b = 0; b < branches; b++) {
        tree.current = node;
        const child = tree.addMove(moves[rand(moves.length)]);
        if (rand(4) === 0) child.comment = COMMENTS[rand(COMMENTS.length)];
        if (rand(5) === 0) child.nags = [1 + rand(20), ...(rand(3) ? [] : [140 + rand(5)])];
        grow(child, depth + 1, b === 0 ? budget : Math.min(budget, depth + 1 + rand(6)));
      }
    };
    grow(tree.root, 0, 6 + rand(24));
    tree.current = tree.root;
    const reparsed = roundTrip(tree);
    assert.deepEqual(reparsed.warnings, [], `round ${round}`);
    assert.deepEqual(shape(reparsed.tree.root), shape(tree.root), `round ${round}: ${serializeMovetext(tree)}`);
    assert.equal(reparsed.tree.root.comment, tree.root.comment);
    assert.equal(serializeMovetext(reparsed.tree), serializeMovetext(tree));
  }
});

test('tag pairs are read only from the header, never from comments or later games', () => {
  const quoted = parsePgnGame('[White "O\\"Neil"]\n[Black "x"]\n\n1. e4 { see [Site "y"] } e5 *');
  assert.deepEqual(quoted.tags, { White: 'O"Neil', Black: 'x' });
  assert.equal(quoted.tree.root.children[0].comment, 'see [Site "y"]');
  assert.deepEqual(mainline(quoted), ['e4', 'e5']);

  const fen = parsePgnGame('1. e4 {[SetUp "1"] [FEN "4k3/8/8/8/8/8/8/4K3 w - - 0 1"]} e5 *');
  assert.deepEqual(fen.tags, {});
  assert.equal(fen.tree.root.fen, START_FEN);
  assert.deepEqual(mainline(fen), ['e4', 'e5']);

  const alice = '[White "Alice"]\n[Black "Bob"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0';
  const carol = '[White "Carol"]\n[Black "Dave"]\n[Result "0-1"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1';
  const glued = parsePgnGame(`${alice}\n\n${carol}`);
  assert.deepEqual(glued.tags, { White: 'Alice', Black: 'Bob', Result: '1-0' });
  assert.equal(glued.result, '1-0');
  assert.deepEqual(mainline(glued), ['e4', 'e5', 'Nf3', 'Nc6']);
  assert.equal(glued.warnings[0].reason, 'tag-pair');
});

test('the header tolerates escape lines, comments and a malformed quote before the movetext', () => {
  const game = parsePgnGame('% exported\n; note\n{intro}\n[Event "Rated "Blitz" game"]\n[White  "A"] [Black "B"]\n\n1. e4 [%clk 0:03:00] e5 *');
  assert.deepEqual(game.tags, { Event: 'Rated "Blitz" game', White: 'A', Black: 'B' });
  assert.equal(game.tree.root.comment, 'intro');
  assert.deepEqual(mainline(game), ['e4', 'e5']);
  assert.deepEqual(game.warnings, []);
});

test('a tag pair inside the movetext stops a tolerant line and fails a strict parse', () => {
  const pgn = '[Event "a"]\n\n1. e4 (1. d4 [Event "b"] d5) e5 *';
  const game = parsePgnGame(pgn);
  assert.equal(game.tags.Event, 'a');
  assert.deepEqual(mainline(game), ['e4', 'e5']);
  assert.equal(game.warnings[0].reason, 'tag-pair');
  assert.equal(game.warnings[0].variation, true);
  assert.throws(() => parsePgnGame(pgn, { strict: true }), /tag pair/i);
});

test('tag-shaped escape lines preserve the following moves in both parse modes', () => {
  for (const newline of ['\n', '\r\n', '\r']) {
    const pgn = ['1. e4', '% [Site "ignored"]', 'e5 2. Nf3 *'].join(newline);
    for (const strict of [false, true]) {
      const game = parsePgnGame(pgn, { strict });
      assert.deepEqual(mainline(game), ['e4', 'e5', 'Nf3']);
      assert.deepEqual(game.tags, {});
      assert.deepEqual(game.warnings, []);
    }
    assert.deepEqual(mainline(parseFirstPgnGame(pgn).game), ['e4', 'e5', 'Nf3']);
  }
});

test('unterminated bracket commands reject strictly and warn before truncating tolerantly', () => {
  const pgn = '1. e4 [%clk 0:03:00 e5 2. Nf3 Nc6 1-0';
  assert.throws(() => parsePgnGame(pgn, { strict: true }), /Unclosed PGN bracket command/);
  const game = parsePgnGame(pgn);
  assert.deepEqual(mainline(game), ['e4']);
  assert.equal(game.hasResult, false);
  assert.equal(game.warnings.length, 1);
  assert.equal(game.warnings[0].reason, 'unclosed-command');
  assert.equal(game.warnings[0].label, '1...');
  assert.match(describePgnWarning(game.warnings[0]), /unclosed bracket command/i);
  // A quoted closing bracket is not the end of a command.
  assert.throws(() => parsePgnGame('1. e4 [%note "]"', { strict: true }), /Unclosed PGN bracket command/);
  assert.deepEqual(mainline(parsePgnGame('1. e4 [%note "]"] e5 *', { strict: true })), ['e4', 'e5']);
});

test('bracket commands adjacent to moves are separate tokens', () => {
  const pgn = '1. e4[%clk 0:03:00]e5 2.Nf3[%eval 0.2] *';
  for (const strict of [false, true]) {
    const game = parsePgnGame(pgn, { strict });
    assert.deepEqual(mainline(game), ['e4', 'e5', 'Nf3']);
    assert.deepEqual(game.warnings, []);
  }
  assert.deepEqual(mainline(parseFirstPgnGame(pgn).game), ['e4', 'e5', 'Nf3']);
  // A stray opening bracket must also make progress, rather than looping forever.
  assert.equal(parsePgnGame('1. e4 [broken e5 *').warnings[0].token, '[broken');
});

test('variation-leading comments stay before their first move and out of the introduction', () => {
  const pgn = '{Intro} 1. e4 ({d4 only} 1. d4 d5) e5 *';
  const { tree } = parsePgnGame(pgn, { strict: true });
  assert.equal(tree.root.comment, 'Intro');
  assert.equal(tree.root.children[1].beforeComment, 'd4 only');
  assert.equal(serializeMovetext(tree), '{Intro} 1. e4 ({d4 only} 1. d4 d5) 1... e5');
  assert.deepEqual(shape(roundTrip(tree).tree.root), shape(tree.root));
  const nested = parsePgnGame('1. e4 e5 ({Sicilian} {plan} 1... c5 2. Nf3 ({other} 2. Nc3 {after} Nc6)) 2. Nf3 *');
  const c5 = nested.tree.root.children[0].children[1];
  assert.equal(c5.beforeComment, 'Sicilian plan');
  assert.equal(c5.children[1].beforeComment, 'other');
  assert.equal(c5.children[1].comment, 'after');
  assert.equal(nested.tree.root.comment, undefined);
  assert.equal(nested.tree.root.children[0].comment, undefined);
  assert.deepEqual(shape(roundTrip(nested.tree).tree.root), shape(nested.tree.root));
});

test('export sanitizes closing braces in every comment position so PGN re-parses', () => {
  const { tree } = parsePgnGame('1. e4 (1. d4) e5 *');
  tree.root.comment = 'Intro } tail';
  tree.root.children[0].comment = 'Move } tail';
  tree.root.children[1].beforeComment = 'Variation } tail';
  const exported = gameTreeToPgn(tree);
  const again = parsePgnGame(exported, { strict: true });
  assert.equal(again.tree.root.comment, 'Intro ] tail');
  assert.equal(again.tree.root.children[0].comment, 'Move ] tail');
  assert.equal(again.tree.root.children[1].beforeComment, 'Variation ] tail');
  assert.deepEqual(mainline(again), ['e4', 'e5']);
  assert.equal(tree.root.comment, 'Intro } tail', 'export must not mutate the tree');
});

test('unmatched closing comment braces warn tolerantly and reject strictly', () => {
  const pgn = '1. e4 } e5 *';
  assert.throws(() => parsePgnGame(pgn, { strict: true }), /Unmatched closing PGN comment brace/);
  const game = parsePgnGame(pgn);
  assert.deepEqual(mainline(game), ['e4']);
  assert.equal(game.warnings[0].reason, 'unmatched-comment');
  assert.equal(game.warnings[0].token, '}');
  assert.match(describePgnWarning(game.warnings[0]), /unmatched closing comment brace/i);
  const variation = parsePgnGame('1. e4 (1. d4 } d5) e5 *');
  assert.deepEqual(mainline(variation), ['e4', 'e5']);
  assert.equal(variation.warnings[0].variation, true);
});

test('5000 nested PGN variations hit a clear depth limit instead of overflowing', () => {
  const alternatives = ['Nc3', 'Nc6', 'Nc3', 'Nc6'];
  const cycle = ['Nf3', 'Nf6', 'Ng1', 'Ng8'];
  const pgn = `${Array.from({ length: 5000 }, (_, i) => `${alternatives[i % 4]} (${cycle[i % 4]} `).join('')}${')'.repeat(5000)} Nf6 *`;
  assert.throws(() => parsePgnGame(pgn, { strict: true }), /PGN variation depth exceeds 128/);
  const game = parsePgnGame(pgn);
  assert.equal(game.warnings.length, 1);
  assert.equal(game.warnings[0].reason, 'variation-depth');
  assert.match(describePgnWarning(game.warnings[0]), /variation depth exceeds 128/i);
  assert.deepEqual(mainline(game), ['Nc3', 'Nf6']);
  assert.throws(() => parsePgnGame(pgn, { strict: true, mainlineOnly: true }), /PGN variation depth exceeds 128/);
});

test('export round-trips 128 variation levels and bounds a 5000-level tree', () => {
  const alternatives = ['b1c3', 'b8c6', 'b1c3', 'b8c6'];
  const cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  for (const limit of [128, 5000]) {
    const tree = new GameTree();
    for (let depth = 0; depth < limit; depth++) {
      const parent = tree.current;
      assert.ok(tree.addUci(alternatives[depth % 4]));
      tree.goTo(parent);
      assert.ok(tree.addUci(cycle[depth % 4]));
    }
    if (limit === 128) {
      const game = parsePgnGame(serializeMovetext(tree), { strict: true });
      assert.deepEqual(shape(game.tree.root), shape(tree.root));
    } else {
      assert.throws(() => serializeMovetext(tree), /PGN variation depth exceeds 128/);
    }
  }
});
