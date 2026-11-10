import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { parsePgnGame } from '../src/chess/pgn.ts';
import {
  adjacentCriticalPly,
  annotatedPgn,
  criticalReviewPlies,
  reviewGame,
  reviewRepetitionCount,
  reviewRepetitionKey,
  terminalReviewPosition,
} from '../src/lc0/gameReview.ts';
import { reviewChartSvg, reviewMoveLabel, reviewPlyAtRatio, reviewSummaryHtml } from '../src/lc0/gameReviewPresentation.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const review = () =>
  reviewGame(
    [
      { winWhite: 0.5, bestUci: 'd2d4', scoreCp: 0 },
      { winWhite: 0.2, bestUci: 'e7e5', scoreCp: -350 },
      { winWhite: 0.35, bestUci: 'e2e4', scoreCp: -150 },
      { winWhite: 0.28, bestUci: null, scoreCp: -200 },
    ],
    [
      { san: 'f3', uci: 'f2f3' },
      { san: 'f6', uci: 'f7f6' },
      { san: 'g4', uci: 'g2g4' },
    ],
  );

function terminal(fen, history = []) {
  const board = parseFen(fen);
  return terminalReviewPosition(board, history, legalMoves(board).length);
}

test('review rejects NaN, infinite and out-of-range probabilities instead of scoring a draw', () => {
  for (const value of [NaN, Infinity, -0.01, 1.1]) {
    assert.throws(() => reviewGame([{ winWhite: value }, { winWhite: 0.5 }], [{ san: 'e4', uci: 'e2e4' }]), /finite score/);
  }
});

test('terminal review detects both colors of mate and stalemate without asking the engine', () => {
  const blackMated = terminal('7k/6Q1/6K1/8/8/8/8/8 b - - 0 25');
  assert.equal(blackMated.winWhite, 1);
  assert.equal(blackMated.mateIn, 0);
  const whiteMated = terminal('8/8/8/8/8/6k1/6q1/7K w - - 100 25');
  assert.equal(whiteMated.winWhite, 0, 'mate takes precedence over the fifty-move rule');
  assert.equal(terminal('7k/5Q2/6K1/8/8/8/8/8 b - - 0 25').scoreCp, 0);
});

test('terminal review accounts for automatic 75-move and insufficient-material draws', () => {
  assert.equal(terminal(START_FEN.replace('0 1', '150 76')).winWhite, 0.5);
  assert.equal(terminal('7k/8/6K1/8/8/8/8/8 w - - 0 25').scoreCp, 0);
  assert.equal(terminal(START_FEN), undefined);
});

test('review draw uses the supplied history, not only the final FEN', () => {
  const game = new GameTree();
  for (const uci of Array(4).fill(['g1f3', 'g8f6', 'f3g1', 'f6g8']).flat()) game.addUci(uci);
  const fens = game.lineTo().map((node) => node.fen);
  assert.equal(terminal(game.current.fen, fens.slice(0, -1)).winWhite, 0.5);
  assert.equal(terminal(game.current.fen, []), undefined);
});

test('claimable threefold repetitions remain engine-scored when the mainline continues', () => {
  const game = new GameTree();
  for (const uci of Array(2).fill(['g1f3', 'g8f6', 'f3g1', 'f6g8']).flat()) game.addUci(uci);
  assert.equal(
    terminal(
      game.current.fen,
      game
        .lineTo()
        .slice(0, -1)
        .map((node) => node.fen),
    ),
    undefined,
  );
});

test('claimable fifty-move positions do not fabricate a draw evaluation', () => {
  assert.equal(terminal(START_FEN.replace('0 1', '100 51')), undefined);
  assert.equal(terminal(START_FEN.replace('0 1', '149 75')), undefined);
});

test('review detects fivefold when the first occurrence has an uncapturable en-passant target', () => {
  const game = new GameTree();
  game.addUci('e2e4');
  for (const uci of Array(4).fill(['g8f6', 'g1f3', 'f6g8', 'f3g1']).flat()) game.addUci(uci);
  const history = game
    .lineTo()
    .slice(0, -1)
    .map((node) => node.fen);
  assert.equal(reviewRepetitionCount(parseFen(game.current.fen), history), 5);
  assert.equal(terminal(game.current.fen, history).scoreCp, 0);
});

test('repetition identity keeps a legal en-passant opportunity distinct', () => {
  const fen = 'k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1';
  assert.notEqual(reviewRepetitionKey(fen), reviewRepetitionKey(fen.replace(' d6 ', ' - ')));
});

test('repetition identity ignores a pinned, illegal en-passant capture', () => {
  const fen = 'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1';
  assert.equal(reviewRepetitionKey(fen), reviewRepetitionKey(fen.replace(' d6 ', ' - ')));
});

test('critical ply hook is ranked and next/previous navigation walks game order with wrap', () => {
  const result = review();
  assert.deepEqual(criticalReviewPlies(result), [1, 2, 3]);
  assert.deepEqual(criticalReviewPlies(result, 2), [1, 2]);
  assert.equal(adjacentCriticalPly(result, 0, 1), 1);
  assert.equal(adjacentCriticalPly(result, 1, 1), 2);
  assert.equal(adjacentCriticalPly(result, 3, 1), 1);
  assert.equal(adjacentCriticalPly(result, 1, -1), 3);
  assert.equal(adjacentCriticalPly({ moves: [] }, 0, 1), undefined);
});

test('critical hook limits the ranked losses to five and does not mutate the move list', () => {
  const moves = Array.from({ length: 9 }, (_, i) => ({ ply: i + 1, class: 'mistake', moverLoss: i / 10 }));
  assert.deepEqual(criticalReviewPlies({ moves }), [9, 8, 7, 6, 5]);
  assert.equal(moves[0].ply, 1);
});

test('custom Black-to-move starts label the actual move number', () => {
  const result = reviewGame(
    [{ winWhite: 0.5 }, { winWhite: 0.8 }, { winWhite: 0.5 }],
    [
      { san: 'f6', uci: 'f7f6' },
      { san: 'g4', uci: 'g2g4' },
    ],
    'b',
  );
  const fen = START_FEN.replace('w KQkq - 0 1', 'b KQkq - 0 37');
  assert.equal(reviewMoveLabel(fen, result.moves[0]), '37… f6');
  assert.equal(reviewMoveLabel(fen, result.moves[1]), '38. g4');
});

test('review graph includes starting position and moves with current marker, no marker on variations', () => {
  const svg = reviewChartSvg(review(), 0);
  assert.match(svg, /viewBox="0 0 360 118"/);
  assert.match(svg, /class="review-cursor"/);
  assert.match(svg, /x1="36" x2="36"/);
  assert.match(svg, /aria-hidden="true"/);
  assert.doesNotMatch(reviewChartSvg(review(), -1), /class="review-cursor"/);
  assert.match(reviewChartSvg(review(), 3), /x1="352" x2="352"/);
});

test('graph hit testing scales with width and clamps to both endpoints', () => {
  assert.equal(reviewPlyAtRatio(36 / 360, 100), 0);
  assert.equal(reviewPlyAtRatio(352 / 360, 100), 100);
  assert.equal(reviewPlyAtRatio(194 / 360, 100), 50);
  assert.equal(reviewPlyAtRatio(-1, 100), 0);
  assert.equal(reviewPlyAtRatio(2, 100), 100);
});

test('summary does not invent 100% accuracy for a side with no informative moves', () => {
  const result = reviewGame([{ winWhite: 0.5, legalMoves: 1 }, { winWhite: 0.5 }], [{ san: 'a3', uci: 'a2a3' }]);
  const markup = reviewSummaryHtml(result);
  assert.equal((markup.match(/class="review-accuracy">—/g) ?? []).length, 2);
});

test('best-move score noise does not lower an otherwise best move accuracy', () => {
  const result = reviewGame([{ winWhite: 0.8, bestUci: 'e2e4' }, { winWhite: 0.5 }], [{ san: 'e4', uci: 'e2e4' }]);
  assert.equal(result.moves[0].class, 'best');
  assert.ok(result.accuracy.white > 99);
});

test('review PGN retains FEN start, Black numbering, headers, result and original evals', () => {
  const fen = START_FEN.replace('w KQkq - 0 1', 'b KQkq - 0 37');
  const result = reviewGame(
    [{ winWhite: 0.5 }, { winWhite: 0.8, scoreCp: 346 }, { winWhite: 1, mateIn: 3 }],
    [
      { san: 'f6', uci: 'f7f6' },
      { san: 'g4', uci: 'g2g4' },
    ],
    'b',
  );
  const pgn = annotatedPgn(result, {
    tags: { White: 'Alice', Black: 'Bob', FEN: fen, SetUp: '1', Result: '0-1' },
    startTurn: 'b',
    startFullmove: 37,
    result: '1-0',
  });
  assert.match(pgn, /37\.\.\. f6 \$4/);
  assert.match(pgn, /38\. g4/);
  assert.match(pgn, /\[%eval 3\.46\]/);
  assert.match(pgn, /\[%eval #3\]/);
  assert.match(pgn, /\[Result "1-0"\]/);
  const parsed = parsePgnGame(pgn);
  assert.equal(parsed.tree.root.fen, fen);
  assert.equal(parsed.tags.White, 'Alice');
  assert.equal(parsed.result, '1-0');
  assert.equal(parsed.tree.mainlineFrom().length, 2);
});

test('probability-only review cannot fabricate centipawns, and header text is escaped', () => {
  const result = reviewGame([{ winWhite: 0.5 }, { winWhite: 0.6 }], [{ san: 'e4', uci: 'e2e4' }]);
  const pgn = annotatedPgn(result, { tags: { White: 'A"B\\C\nD', 'bad]': 'injection' }, result: 'not valid' });
  assert.doesNotMatch(pgn, /\[%eval/);
  assert.match(pgn, /\[White "A\\"B\\\\C D"\]/);
  assert.doesNotMatch(pgn, /injection/);
  assert.ok(pgn.trim().endsWith('*'));
});
