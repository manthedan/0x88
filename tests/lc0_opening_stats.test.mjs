import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { parsePgnGames } from '../src/chess/pgn.ts';
import { buildOpeningPositionIndex, mergeOpeningMoveStats, openingStatsForPosition, openingStatsFromIndex, openingSummary, positionKey } from '../src/lc0/openingStats.ts';

const PGN = [
  '[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0',
  '[Result "0-1"]\n\n1. e4 e5 2. Nf3 Nc6 0-1',
  '[Result "1/2-1/2"]\n\n1. e4 c5 1/2-1/2',
  '[Result "1-0"]\n\n1. d4 d5 1-0',
].join('\n\n');

test('positionKey drops the move clocks', () => {
  assert.equal(positionKey(START_FEN), 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -');
});

test('openingStatsForPosition tallies first moves with W/D/L', () => {
  const games = parsePgnGames(PGN);
  const stats = openingStatsForPosition(games, START_FEN);
  const e4 = stats.find((s) => s.san === 'e4');
  const d4 = stats.find((s) => s.san === 'd4');
  assert.equal(e4.count, 3, '3 games opened 1.e4');
  assert.equal(e4.whiteWins, 1);
  assert.equal(e4.blackWins, 1);
  assert.equal(e4.draws, 1);
  assert.equal(d4.count, 1);
  assert.equal(stats[0].san, 'e4', 'most frequent first');
});

test('openingStatsForPosition follows transpositions to a deeper position', () => {
  const games = parsePgnGames(PGN);
  // Position after 1.e4 e5 2.Nf3 — both e5 games reached it; next move Nc6.
  const afterNf3 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2';
  const stats = openingStatsForPosition(games, afterNf3);
  assert.equal(stats.length, 1);
  assert.equal(stats[0].san, 'Nc6');
  assert.equal(stats[0].count, 2);
  const summary = openingSummary(stats);
  assert.deepEqual(summary, { total: 2, whiteWins: 1, blackWins: 1, draws: 0 });
});

test('buildOpeningPositionIndex precomputes lookup stats by position key', () => {
  const games = parsePgnGames(PGN);
  const index = buildOpeningPositionIndex(games);
  const startStats = openingStatsFromIndex(index, START_FEN);
  assert.equal(Object.keys(index).includes(positionKey(START_FEN)), true);
  assert.equal(startStats[0].san, 'e4');
  assert.equal(startStats[0].count, 3);
  assert.deepEqual(openingStatsFromIndex(index, '8/8/8/8/8/8/8/8 w - - 0 1'), []);
});

test('mergeOpeningMoveStats aggregates saved collection search hits', () => {
  const merged = mergeOpeningMoveStats([
    [{ uci: 'e2e4', san: 'e4', count: 2, whiteWins: 1, blackWins: 1, draws: 0 }],
    [{ uci: 'e2e4', san: 'e4', count: 1, whiteWins: 0, blackWins: 0, draws: 1 }, { uci: 'd2d4', san: 'd4', count: 2, whiteWins: 2, blackWins: 0, draws: 0 }],
  ]);
  assert.deepEqual(merged.map((stat) => [stat.san, stat.count]), [['e4', 3], ['d4', 2]]);
  assert.equal(merged[0].draws, 1);
});
