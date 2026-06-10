import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  TournamentStandings,
  buildSchedule,
  eloError95,
  eloFromScore,
  gauntletPairings,
  roundRobinPairings,
  tournamentPairings,
} from '../src/lc0/tournament.ts';

test('round-robin generates all unordered pairs in seat order', () => {
  assert.deepEqual(roundRobinPairings(['a', 'b', 'c']), [
    { a: 'a', b: 'b' },
    { a: 'a', b: 'c' },
    { a: 'b', b: 'c' },
  ]);
  assert.deepEqual(roundRobinPairings(['a']), []);
});

test('gauntlet pairs seat 1 against the field', () => {
  assert.deepEqual(gauntletPairings(['hero', 'x', 'y', 'z']), [
    { a: 'hero', b: 'x' },
    { a: 'hero', b: 'y' },
    { a: 'hero', b: 'z' },
  ]);
});

test('mode dispatch: match uses only the first two seats', () => {
  assert.deepEqual(tournamentPairings('match', ['a', 'b', 'c']), [{ a: 'a', b: 'b' }]);
  assert.deepEqual(tournamentPairings('round-robin', ['a', 'b']), [{ a: 'a', b: 'b' }]);
  assert.deepEqual(tournamentPairings('gauntlet', ['a']), []);
});

test('schedule expands pairings over openings with per-game color alternation', () => {
  const games = buildSchedule([{ a: 'p', b: 'q' }], ['open1', 'open2'], 2);
  assert.equal(games.length, 4);
  // game index 0 within the pairing: p is white for every opening
  assert.deepEqual(games.slice(0, 2).map((g) => [g.whiteId, g.blackId, g.opening]), [
    ['p', 'q', 'open1'],
    ['p', 'q', 'open2'],
  ]);
  // game index 1: colors flip
  assert.deepEqual(games.slice(2).map((g) => [g.whiteId, g.blackId, g.opening]), [
    ['q', 'p', 'open1'],
    ['q', 'p', 'open2'],
  ]);
  assert.ok(games.every((g) => g.pairingKey === 'p|q'));
});

test('schedule covers a 3-engine double round robin evenly', () => {
  const games = buildSchedule(roundRobinPairings(['a', 'b', 'c']), ['o'], 2);
  assert.equal(games.length, 6);
  const whiteCounts = new Map();
  const perEngine = new Map();
  for (const g of games) {
    whiteCounts.set(g.whiteId, (whiteCounts.get(g.whiteId) ?? 0) + 1);
    for (const id of [g.whiteId, g.blackId]) perEngine.set(id, (perEngine.get(id) ?? 0) + 1);
  }
  for (const id of ['a', 'b', 'c']) {
    assert.equal(perEngine.get(id), 4, `engine ${id} plays 4 games`);
    assert.equal(whiteCounts.get(id), 2, `engine ${id} gets white twice`);
  }
});

test('elo helpers behave at the anchors and clamp degenerate scores', () => {
  assert.equal(eloFromScore(0.5), 0);
  assert.ok(Math.abs(eloFromScore(0.75) - 190.85) < 0.1);
  assert.equal(eloFromScore(1), 1200);
  assert.equal(eloFromScore(0), -1200);
  assert.equal(eloError95(0.5, 0), null);
  assert.equal(eloError95(1, 100), null);
  // More games shrink the error bar.
  assert.ok(eloError95(0.6, 100) < eloError95(0.6, 25));
});

test('standings track points, records, and elo across a small tournament', () => {
  const standings = new TournamentStandings([
    { id: 'a', name: 'Alpha' },
    { id: 'b', name: 'Beta' },
    { id: 'c', name: 'Gamma' },
  ]);
  standings.record('a', 'b', '1-0');
  standings.record('b', 'a', '1/2-1/2');
  standings.record('a', 'c', '1-0');
  standings.record('c', 'a', '0-1');
  standings.record('b', 'c', '1-0');
  standings.record('c', 'b', '1-0');

  const table = standings.table();
  assert.deepEqual(table.map((row) => row.name), ['Alpha', 'Beta', 'Gamma']);

  const alpha = table[0];
  assert.equal(alpha.games, 4);
  assert.equal(alpha.wins, 3);
  assert.equal(alpha.draws, 1);
  assert.equal(alpha.losses, 0);
  assert.equal(alpha.points, 3.5);
  assert.equal(alpha.score, 0.875);
  assert.ok(alpha.eloDiff > 300);
  assert.ok(alpha.eloError > 0);
  assert.deepEqual(alpha.perOpponent.get('b'), { wins: 1, draws: 1, losses: 0 });
  assert.deepEqual(alpha.perOpponent.get('c'), { wins: 2, draws: 0, losses: 0 });

  const beta = table[1];
  assert.equal(beta.points, 1.5);
  const gamma = table[2];
  assert.equal(gamma.points, 1);
  assert.equal(standings.totalGames(), 6);
});

test('standings ignore self-pairings and unknown ids', () => {
  const standings = new TournamentStandings([{ id: 'a', name: 'Alpha' }]);
  standings.record('a', 'a', '1-0');
  standings.record('a', 'ghost', '1-0');
  assert.equal(standings.table()[0].games, 0);
  assert.equal(standings.totalGames(), 0);
});
