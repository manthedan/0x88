import assert from 'node:assert/strict';
import { test } from 'node:test';
import { arenaOpeningPairs, arenaPointsForSeat, formatArenaPoints } from '../src/lc0/arenaResults.ts';
import { buildSchedule, roundRobinPairings } from '../src/lc0/tournament.ts';

test('Arena half-point notation includes zero and totals', () => {
  assert.deepEqual([0, 0.5, 1, 1.5, 2].map(formatArenaPoints), ['0', '½', '1', '1½', '2']);
  assert.equal(arenaPointsForSeat({ whiteId: 'b' }, 'a', '0-1'), 1);
});

test('opening pairs use reversed colours and schedule indices, not completion order', () => {
  const schedule = buildSchedule([{ a: 'a', b: 'b' }], [{ name: 'One' }, { name: 'Two' }], 2);
  const pairs = arenaOpeningPairs(schedule, [
    { scheduleIndex: 2, result: '1/2-1/2' },
    { scheduleIndex: 0, result: '1-0' },
    { scheduleIndex: 3, result: '1-0' },
  ]);
  assert.deepEqual(pairs[0].indices, [0, 2]);
  assert.deepEqual(pairs[0].points, [1.5, 0.5]);
  assert.equal(pairs[0].played, 2);
  assert.deepEqual(pairs[1].results, [null, '1-0']);
  assert.deepEqual(pairs[1].points, [0, 1]);
});

test('duplicate suite slots, repetitions, and same-named seats stay separate', () => {
  const opening = { name: 'Duplicate' };
  const schedule = buildSchedule(roundRobinPairings(['0', '1', '2']), [opening, opening], 5);
  const pairs = arenaOpeningPairs(schedule, []);
  assert.equal(pairs.length, 18);
  assert.deepEqual(pairs.slice(0, 6).map((pair) => [pair.openingIndex, pair.round, pair.indices]), [
    [0, 1, [0, 2]], [1, 1, [1, 3]], [0, 2, [4, 6]], [1, 2, [5, 7]], [0, 3, [8]], [1, 3, [9]],
  ]);
  assert.deepEqual([...new Set(pairs.map((pair) => pair.pairingKey))], ['0|1', '0|2', '1|2']);
  assert.ok(pairs.every((pair) => pair.played === 0));
});

test('duplicate result updates cannot double-count pair scores', () => {
  const schedule = buildSchedule([{ a: 'a', b: 'b' }], ['o'], 2);
  const pairs = arenaOpeningPairs(schedule, [{ scheduleIndex: 0, result: '1-0' }, { scheduleIndex: 0, result: '1-0' }]);
  assert.equal(pairs[0].played, 1);
  assert.deepEqual(pairs[0].points, [1, 0]);
});
