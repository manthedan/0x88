import assert from 'node:assert/strict';
import { test } from 'node:test';
import { applyGameResult, gauntletPairings, initStandings, rankedStandings, roundRobinPairings } from '../src/lc0/arena.ts';

test('roundRobinPairings makes every pair once with alternating colors', () => {
  const pairings = roundRobinPairings(['a', 'b', 'c'], 2);
  // 3 pairs * 2 games = 6.
  assert.equal(pairings.length, 6);
  assert.deepEqual(pairings[0], { white: 'a', black: 'b' });
  assert.deepEqual(pairings[1], { white: 'b', black: 'a' });
  // single-game-per-pair keeps each pair once
  assert.equal(roundRobinPairings(['a', 'b', 'c', 'd'], 1).length, 6);
});

test('gauntletPairings pits champions against challengers only', () => {
  const pairings = gauntletPairings(['champ'], ['x', 'y'], 2);
  assert.equal(pairings.length, 4);
  assert.ok(pairings.every((p) => p.white === 'champ' || p.black === 'champ'));
  assert.ok(!pairings.some((p) => p.white === p.black));
  // a champion that is also a challenger does not self-pair
  assert.equal(gauntletPairings(['a'], ['a', 'b'], 2).length, 2);
});

test('applyGameResult and rankedStandings track scores and order', () => {
  const standings = initStandings([{ id: 'a', name: 'A' }, { id: 'b', name: 'B' }, { id: 'c', name: 'C' }]);
  applyGameResult(standings, 'a', 'b', '1-0');   // A beats B
  applyGameResult(standings, 'b', 'a', '1/2-1/2'); // A draws B
  applyGameResult(standings, 'c', 'a', '1-0');   // C beats A

  const a = standings.get('a');
  assert.equal(a.wins, 1); assert.equal(a.draws, 1); assert.equal(a.losses, 1);
  assert.equal(a.score, 1.5); assert.equal(a.games, 3);
  assert.equal(standings.get('b').score, 0.5);
  assert.equal(standings.get('c').score, 1);

  const ranked = rankedStandings(standings);
  assert.equal(ranked[0].id, 'a', 'A leads with 1.5');
  assert.equal(ranked[0].score, 1.5);
});
