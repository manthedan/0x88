import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import {
  evalBarWhitePercent,
  formatScore,
  lc0AnalysisLines,
  qToCentipawns,
} from '../src/lc0/analysisFormat.ts';

test('qToCentipawns is monotonic and signed like the LC0 mapping', () => {
  assert.equal(qToCentipawns(0), 0);
  assert.ok(qToCentipawns(0.5) > 0);
  assert.ok(qToCentipawns(-0.5) < 0);
  assert.ok(qToCentipawns(0.9) > qToCentipawns(0.5));
  assert.ok(Number.isFinite(qToCentipawns(1)), 'q=1 is clamped, not Infinity');
});

test('formatScore renders mate and centipawns', () => {
  assert.equal(formatScore(123, undefined), '+1.23');
  assert.equal(formatScore(-50, undefined), '-0.50');
  assert.equal(formatScore(0, undefined), '+0.00');
  assert.equal(formatScore(999, 5), '#5', 'mate wins over cp');
  assert.equal(formatScore(undefined, -3), '#-3');
  assert.equal(formatScore(undefined, undefined), '—');
});

test('evalBarWhitePercent flips with side to move and clamps to [0,100]', () => {
  assert.equal(evalBarWhitePercent(0, undefined, 'w'), 50);
  assert.ok(evalBarWhitePercent(300, undefined, 'w') > 50, 'white ahead, white to move');
  // Same side-to-move score but black to move means white is behind.
  assert.ok(evalBarWhitePercent(300, undefined, 'b') < 50);
  assert.equal(evalBarWhitePercent(undefined, 5, 'w'), 100, 'white mating');
  assert.equal(evalBarWhitePercent(undefined, 5, 'b'), 0, 'black mating');
  const p = evalBarWhitePercent(100000, undefined, 'w');
  assert.ok(p <= 100 && p >= 0);
});

test('lc0AnalysisLines builds MultiPV lines with SAN and root-mover score', () => {
  const result = {
    value: 0.1,
    visits: 40,
    pv: ['d2d4', 'd7d5'],
    multiPv: [['d2d4', 'd7d5', 'c2c4'], ['g1f3', 'g8f6']],
    children: [
      { uci: 'd2d4', visits: 21, q: -0.12 }, // child q is from black's view; root (white) score = +0.12
      { uci: 'g1f3', visits: 12, q: -0.05 },
    ],
  };
  const lines = lc0AnalysisLines(result, START_FEN, 'LC0');
  assert.equal(lines.length, 2);
  assert.equal(lines[0].multipv, 1);
  assert.equal(lines[0].engine, 'LC0');
  assert.ok(lines[0].scoreCp > 0, 'white is slightly better on the d4 line');
  assert.match(lines[0].pvSan, /^d4 d5 c4/);
  assert.equal(lines[0].detail, '21 visits');
  assert.match(lines[1].pvSan, /^Nf3 Nf6/);
});

test('lc0AnalysisLines falls back to the single PV when multiPv is absent', () => {
  const result = { value: -0.2, visits: 16, pv: ['e2e4', 'e7e5'], children: [{ uci: 'e2e4', visits: 16, q: 0.2 }] };
  const lines = lc0AnalysisLines(result, START_FEN);
  assert.equal(lines.length, 1);
  assert.match(lines[0].pvSan, /^e4 e5/);
});
