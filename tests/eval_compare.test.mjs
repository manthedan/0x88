import assert from 'node:assert/strict';
import { test } from 'node:test';
import { arenaComparisonStatus, arenaLaneOutput, compactEvalSpread, compareArenaEvals, compareEvals, evalPercent, fenPly, searchAge, searchAgeLabel } from '../src/lc0/evalCompare.ts';
const fen = (ply, pieces = '8/8/8/8/8/8/8/K6k') => `${pieces} ${ply % 2 ? 'b' : 'w'} - - 0 ${1 + Math.floor(ply / 2)}`;
const score = (ply, cp = 20) => ({ fen: fen(ply), scoreCp: cp });

test('meaningful spread uses the nonlinear scale, not a fixed pawn threshold', () => {
  assert.equal(compareEvals([{ scoreCp: 500 }, { scoreCp: 600 }]).meaningful, false);
  const spread = compareEvals([{ scoreCp: 20 }, { scoreCp: 180 }]);
  assert.equal(spread.meaningful, true);
  assert.equal(spread.label, '1.60 pawns');
  assert.ok(spread.spread > 15 && spread.spread < 16);
  assert.equal(compactEvalSpread(spread), '1.60');
});
test('side disagreement excludes near-zero sign noise', () => {
  assert.equal(compareEvals([{ scoreCp: -5 }, { scoreCp: 5 }]).oppositeSides, false);
  assert.equal(compareEvals([{ scoreCp: -60 }, { scoreCp: 110 }]).oppositeSides, true);
});
test('absent and non-finite scores never become equal results', () => {
  for (const score of [{}, { scoreCp: NaN }, { scoreCp: Infinity }]) assert.equal(evalPercent(score), undefined);
  assert.equal(compareEvals([{}, { scoreCp: 10 }]), undefined);
  assert.equal(compareEvals([]), undefined);
  assert.equal(evalPercent({ scoreCp: 0 }), 50);
});
test('mates take precedence, pin to endpoints and are never subtracted as pawns', () => {
  assert.equal(evalPercent({ scoreCp: -100, mateIn: 7 }), 100);
  assert.equal(evalPercent({ mateIn: -7 }), 0);
  assert.equal(evalPercent({ mateIn: 0 }), 100);
  const spread = compareEvals([{ mateIn: -3 }, { mateIn: 5 }]);
  assert.equal(spread.label, '100 points on bar');
  assert.equal(compactEvalSpread(spread), '100 pts');
  assert.equal(spread.oppositeSides, true);
  assert.equal(compareEvals([{ mateIn: 3 }, { mateIn: 7 }]).meaningful, false);
});
test('FEN age includes fullmove and side to move, also with custom roots', () => {
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
});
test('missing, old, widely separated and future results are not comparable', () => {
  for (const scores of [[{}, score(31)], [score(28), score(29)], [score(29), score(31)], [score(32), score(33)]]) {
    assert.equal(compareArenaEvals(scores, fen(32)).fair, false);
    assert.equal(compareArenaEvals(scores, fen(32)).comparison, undefined);
  }
});
test('only a completed not-comparable pair dims its stale lanes', () => {
  assert.deepEqual(compareArenaEvals([score(30), score(31)], fen(32)).stale, [false, false]);
  assert.deepEqual(compareArenaEvals([score(26), score(31)], fen(32)).stale, [true, false]);
  assert.deepEqual(compareArenaEvals([score(30), score(25)], fen(32)).stale, [false, true]);
  assert.deepEqual(compareArenaEvals([score(27), score(28)], fen(32)).stale, [true, true]);
  assert.deepEqual(compareArenaEvals([score(32), score(33)], fen(32)).stale, [false, true]);
  assert.deepEqual(compareArenaEvals([score(26), {}], fen(32)).stale, [false, false]);
  const unrelated = { ...score(31), fen: fen(31, '8/8/8/8/8/8/K7/7k') };
  assert.deepEqual(compareArenaEvals([score(31), unrelated], fen(32)).stale, [true, true]);
});
test('same-ply results require matching positions; current FEN mismatch is stale', () => {
  const other = { fen: fen(31, '8/8/8/8/8/8/K7/7k'), scoreCp: 50 };
  assert.equal(compareArenaEvals([score(31), other], fen(32)).fair, false);
  assert.equal(compareArenaEvals([score(31), score(31)], fen(32)).fair, true);
  assert.equal(searchAge(other.fen, fen(31)), undefined);
});
test('age labels preserve exact ply units in accessible details', () => {
  assert.equal(searchAgeLabel(0), 'Current position');
  assert.equal(searchAgeLabel(1), '1 ply ago');
  assert.equal(searchAgeLabel(2), '2 plies ago');
  assert.match(searchAgeLabel(-2), /later than the board/);
  assert.match(searchAgeLabel(undefined), /No search/);
});
test('Arena status states adjacent positions rather than claiming agreement', () => {
  const status = (scores) => arenaComparisonStatus(compareArenaEvals(scores, fen(2)), scores.map((s) => evalPercent(s) !== undefined), ['Lc0', 'SF']);
  assert.equal(status([{}, {}]).text, 'Waiting for first search');
  assert.match(status([score(0), {}]).text, /Black.*search/);
  assert.match(status([{}, score(1)]).text, /White.*search/);
  assert.deepEqual(status([score(0, 30), score(1, 40)]), { text: 'similar, one move apart', emphasis: false });
  assert.deepEqual(status([score(0, 0), score(1, 51)]), { text: 'similar, one move apart', emphasis: false });
  assert.deepEqual(status([score(0, 20), score(1, 180)]), { text: '1.60 apart', emphasis: true });
  assert.deepEqual(status([score(0, -60), score(1, 110)]), { text: 'disagree on who’s better', emphasis: true });
  assert.equal(status([score(1), score(1)]).text, 'similar, same position');
});
test('unfair age is explained only in the not-comparable status', () => {
  assert.deepEqual(arenaComparisonStatus(compareArenaEvals([score(2), score(5)], fen(6)), [true, true], ['Lc0', 'SF']), { text: 'not comparable, Lc0’s eval is 4 plies old', emphasis: false });
});
test('self-play retains each side’s own result and resets at game/engine boundaries', () => {
  const white = { engineId: 'lc0-small', ...score(0, 20) };
  const black = { engineId: 'lc0-small', ...score(1, -60) };
  assert.equal(arenaLaneOutput(undefined, white, white.engineId, 'w'), white);
  assert.equal(arenaLaneOutput(undefined, white, white.engineId, 'b'), undefined);
  assert.equal(arenaLaneOutput(white, black, white.engineId, 'w'), white);
  assert.equal(arenaLaneOutput(undefined, black, black.engineId, 'b'), black);
  assert.equal(arenaLaneOutput(white, undefined, white.engineId, 'w'), undefined);
  assert.equal(arenaLaneOutput(white, white, null, 'w'), undefined);
  assert.equal(arenaLaneOutput(white, white, 'stockfish', 'w'), undefined);
});
