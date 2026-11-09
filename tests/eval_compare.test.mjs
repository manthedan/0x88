import assert from 'node:assert/strict';
import { test } from 'node:test';
import { arenaLaneOutput, compareArenaEvals, compareEvals, evalPercent, evalTickBadgeShift, evalTickFocusKey, evalTooltipShift, fenPly, groupEvalTicks, searchAge, searchAgeLabel } from '../src/lc0/evalCompare.ts';

const fen = (ply, pieces = '8/8/8/8/8/8/8/K6k') => `${pieces} ${ply % 2 ? 'b' : 'w'} - - 0 ${1 + Math.floor(ply / 2)}`;
const score = (ply, cp = 20) => ({ fen: fen(ply), scoreCp: cp });

test('bar spread uses the nonlinear scale, not a fixed pawn threshold', () => {
  assert.equal(compareEvals([{ scoreCp: 500 }, { scoreCp: 600 }]).meaningful, false);
  const spread = compareEvals([{ scoreCp: 20 }, { scoreCp: 180 }]);
  assert.equal(spread.meaningful, true);
  assert.equal(spread.label, '1.60 pawns');
  assert.ok(spread.spread > 15 && spread.spread < 16);
});
test('side disagreement excludes near-zero sign noise', () => {
  assert.equal(compareEvals([{ scoreCp: -5 }, { scoreCp: 5 }]).oppositeSides, false);
  assert.equal(compareEvals([{ scoreCp: -60 }, { scoreCp: 110 }]).oppositeSides, true);
});
test('absent and non-finite scores never become equal results', () => {
  assert.equal(evalPercent({}), undefined);
  assert.equal(evalPercent({ scoreCp: NaN }), undefined);
  assert.equal(evalPercent({ scoreCp: Infinity }), undefined);
  assert.equal(compareEvals([{}, { scoreCp: 10 }]), undefined);
  assert.equal(compareEvals([]), undefined);
  assert.equal(evalPercent({ scoreCp: 0 }), 50);
});
test('mates take precedence, pin to endpoints and are never subtracted as pawns', () => {
  assert.equal(evalPercent({ scoreCp: -100, mateIn: 7 }), 100);
  assert.equal(evalPercent({ mateIn: -7 }), 0);
  assert.equal(evalPercent({ mateIn: 0 }), 100); // same convention as evalBarWhitePercent
  const spread = compareEvals([{ mateIn: -3 }, { mateIn: 5 }]);
  assert.equal(spread.label, '100 points on bar');
  assert.equal(spread.oppositeSides, true);
  assert.equal(compareEvals([{ mateIn: 3 }, { mateIn: 7 }]).meaningful, false);
});
test('FEN age includes fullmove and side to move, also with custom roots', () => {
  assert.equal(fenPly(fen(0)), 0);
  assert.equal(fenPly(fen(31)), 31);
  assert.equal(searchAge(fen(30), fen(32)), 2);
  assert.equal(searchAge(fen(31), fen(32)), 1);
  assert.equal(searchAge(fen(32), fen(32)), 0);
  assert.equal(fenPly('bad fen'), undefined);
  assert.equal(fenPly('8/8 w - - 0 0'), undefined);
  assert.equal(searchAge(undefined, fen(32)), undefined);
});
test('freshness follows the most recent completed search, not board turn', () => {
  const result = compareArenaEvals([score(30), score(31)], fen(32));
  assert.deepEqual(result.ages, [2, 1]);
  assert.deepEqual(result.fresh, [false, true]);
  assert.equal(result.fair, true);
  assert.equal(result.adjacent, true);
  assert.deepEqual(compareArenaEvals([score(32), score(31)], fen(32)).fresh, [true, false]);
});
test('Arena does not claim a comparison until both engines have searched', () => {
  const result = compareArenaEvals([{}, score(31)], fen(32));
  assert.equal(result.fair, false);
  assert.equal(result.comparison, undefined);
  assert.deepEqual(result.fresh, [false, true]);
});
test('old, widely separated, or future review results cannot claim agreement', () => {
  for (const scores of [[score(28), score(29)], [score(29), score(31)], [score(32), score(33)]]) {
    assert.equal(compareArenaEvals(scores, fen(32)).fair, false);
  }
  assert.deepEqual(compareArenaEvals([score(28), score(29)], fen(32)).fresh, [false, false]);
});
test('same-ply results require matching positions; current FEN mismatch is stale', () => {
  const other = { fen: fen(31, '8/8/8/8/8/8/K7/7k'), scoreCp: 50 };
  assert.equal(compareArenaEvals([score(31), other], fen(32)).fair, false);
  assert.equal(compareArenaEvals([score(31), score(31)], fen(32)).fair, true);
  assert.equal(searchAge(other.fen, fen(31)), undefined);
});
test('search age labels distinguish current, previous, missing and future', () => {
  assert.equal(searchAgeLabel(0), 'Current position');
  assert.equal(searchAgeLabel(1), '1 ply ago');
  assert.equal(searchAgeLabel(2), '2 plies ago');
  assert.match(searchAgeLabel(-2), /later than the board/);
  assert.match(searchAgeLabel(undefined), /No search/);
});
test('tick groups carry every identity and anchor to the chosen engine', () => {
  const ticks = [{ id: 'a', scoreCp: 20 }, { id: 'b', scoreCp: 22 }, { id: 'c', scoreCp: 180 }, { id: 'empty' }];
  const groups = groupEvalTicks(ticks, 'b', 700);
  assert.equal(groups.length, 2);
  assert.deepEqual(groups[0].members.map((tick) => tick.id), ['a', 'b']);
  assert.equal(groups[0].selected, true);
  assert.equal(groups[0].percent, evalPercent(ticks[1]));
  assert.equal(groups[1].selected, false);
});
test('tick collision tolerance is in pixels and changes with track length', () => {
  const ticks = [{ id: 'a', scoreCp: 0 }, { id: 'b', scoreCp: 10 }];
  assert.equal(groupEvalTicks(ticks, undefined, 390).length, 1);
  assert.equal(groupEvalTicks(ticks, undefined, 700).length, 2);
});
test('count badges coalesce instead of overlapping other counts or individual ticks', () => {
  const ticks = [22, 25, 36, 40].map((cp, index) => ({ id: String(index), scoreCp: cp }));
  const groups = groupEvalTicks(ticks, '1', 700);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].members.length, 4);
  assert.equal(groups[0].percent, evalPercent(ticks[1]));
});
test('endpoint badge placement and collision detection share pixel bounds', () => {
  assert.equal(evalTickBadgeShift(0, 382, true), 8.5);
  assert.equal(evalTickBadgeShift(100, 382, true), -8.5);
  assert.equal(evalTickBadgeShift(8, 382, true), 0, 'do not shift badges that already fit');
  assert.equal(evalTickBadgeShift(92, 382, false), 0);
  const ticks = [-620, -610, -510].map((scoreCp, index) => ({ id: String(index), scoreCp }));
  assert.equal(groupEvalTicks(ticks, '1', 382).length, 1, 'leave no overlapping count and adjacent tick');
  const groups = groupEvalTicks([-10000, -1400, -1000, -800, -620, -610, -510, -200, -50, 0, 100, 1000, 1300, 10000]
    .map((scoreCp, index) => ({ id: String(index), scoreCp })), '5', 382);
  for (let i = 1; i < groups.length; i++) {
    const bounds = (group) => {
      const half = group.members.length > 1 ? (group.selected ? 8.5 : 7.5) : group.selected ? 4 : 3;
      const center = group.percent * 382 / 100 + (group.members.length > 1 ? evalTickBadgeShift(group.percent, 382, group.selected) : 0);
      return { min: center - half, max: center + half };
    };
    assert.ok(bounds(groups[i]).min - bounds(groups[i - 1]).max >= 2);
  }
});
test('separate single ticks do not chain into a misleading count group', () => {
  const ticks = [0, 10, 20].map((cp, index) => ({ id: String(index), scoreCp: cp }));
  assert.equal(groupEvalTicks(ticks, undefined, 700).length, 3);
});
test('self-play keeps distinct side results even when the engine ID is shared', () => {
  const white = { engineId: 'lc0-small', ...score(0, 20) };
  const black = { engineId: 'lc0-small', ...score(1, -60) };
  assert.equal(arenaLaneOutput(undefined, white, white.engineId, 'w'), white);
  assert.equal(arenaLaneOutput(undefined, white, white.engineId, 'b'), undefined);
  assert.equal(arenaLaneOutput(white, black, white.engineId, 'w'), white);
  assert.equal(arenaLaneOutput(undefined, black, black.engineId, 'b'), black);
  const next = { engineId: 'lc0-small', ...score(2, 40) };
  assert.equal(arenaLaneOutput(white, next, next.engineId, 'w'), next);
  assert.equal(arenaLaneOutput(black, next, next.engineId, 'b'), black);
});
test('lane results reset at game boundaries and when the playing engine changes', () => {
  const previous = { engineId: 'lc0-small', ...score(0) };
  assert.equal(arenaLaneOutput(previous, undefined, previous.engineId, 'w'), undefined);
  assert.equal(arenaLaneOutput(previous, previous, null, 'w'), undefined);
  assert.equal(arenaLaneOutput(previous, previous, 'stockfish', 'w'), undefined);
});
test('focused engine stays the focus key when a tick joins a group or splits out', () => {
  const before = evalTickFocusKey(['b'], undefined, undefined);
  const joined = evalTickFocusKey(['a', 'b', 'c'], 'a', before);
  assert.equal(joined, 'b');
  assert.equal(evalTickFocusKey(['a'], 'a', joined), 'a');
  assert.equal(evalTickFocusKey(['b', 'c'], 'a', joined), 'b');
});
test('mobile tooltips stay within gutters at endpoints and at +/-5 pawn positions', () => {
  for (const cp of [-10000, -500, -200, 0, 200, 500, 10000]) {
    const anchor = 4 + 382 * evalPercent({ scoreCp: cp }) / 100;
    const center = anchor + evalTooltipShift(anchor, 390);
    assert.ok(center - 90 >= 8 - 0.001, `left bound at ${cp}`);
    assert.ok(center + 90 <= 382 + 0.001, `right bound at ${cp}`);
  }
  assert.equal(evalTooltipShift(195, 390), 0);
});

test('mate endpoints and exact overlaps remain visible in groups', () => {
  const groups = groupEvalTicks([{ id: 'a', mateIn: 1 }, { id: 'b', mateIn: 7 }, { id: 'c', mateIn: -3 }], 'a', 390);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].percent, 0);
  assert.equal(groups[1].percent, 100);
  assert.equal(groups[1].members.length, 2);
  assert.deepEqual(groupEvalTicks([], undefined, 0), []);
});
