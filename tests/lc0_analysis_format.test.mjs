import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import {
  engineBrushes,
  engineColorKey,
  evalBarWhitePercent,
  formatScore,
  lc0AnalysisLines,
  qToCentipawns,
  stockfishAnalysisLines,
} from '../src/lc0/analysisFormat.ts';
import { parseStockfishInfo } from '../src/lc0/stockfishEngine.ts';

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
      { uci: 'd2d4', visits: 21, q: 0.12 }, // child q is the move's value for the root mover (white)
      { uci: 'g1f3', visits: 12, q: 0.05 },
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

test('engineColorKey assigns stable per-engine color families', () => {
  assert.equal(engineColorKey('LC0'), 'green');
  assert.equal(engineColorKey('LC0 search 400'), 'green');
  assert.equal(engineColorKey('SF d14'), 'blue');
  assert.equal(engineColorKey('Stockfish lite'), 'blue');
  assert.equal(engineColorKey('Komodo'), 'yellow');
  const lc0 = engineBrushes('LC0');
  assert.equal(lc0.primary, 'green');
  assert.equal(lc0.alt, 'paleGreen');
  assert.equal(engineBrushes('SF d8').primary, 'blue');
});

test('parseStockfishInfo extracts multipv, score, mate, and PV', () => {
  const a = parseStockfishInfo('info depth 18 seldepth 24 multipv 1 score cp 35 nodes 1000 nps 5 pv e2e4 e7e5 g1f3');
  assert.deepEqual(a, { multipv: 1, depth: 18, scoreCp: 35, mateIn: undefined, pvUci: ['e2e4', 'e7e5', 'g1f3'] });
  const b = parseStockfishInfo('info depth 10 multipv 2 score mate -3 pv f1c4 d7d5');
  assert.equal(b.mateIn, -3);
  assert.equal(b.multipv, 2);
  assert.equal(parseStockfishInfo('info depth 1 seldepth 1 score cp 0 nodes 20'), null, 'no PV -> null');
  assert.equal(parseStockfishInfo('info string NNUE evaluation using net'), null);
  assert.equal(parseStockfishInfo('bestmove e2e4'), null);
});

test('stockfishAnalysisLines converts info lines to SAN-rendered analysis lines', () => {
  const lines = stockfishAnalysisLines(
    [{ multipv: 1, depth: 18, scoreCp: 35, pvUci: ['e2e4', 'e7e5'] }, { multipv: 2, depth: 18, mateIn: 4, pvUci: ['d2d4', 'd7d5'] }],
    START_FEN,
    'SF 18',
  );
  assert.equal(lines.length, 2);
  assert.equal(lines[0].engine, 'SF 18');
  assert.equal(lines[0].scoreText, '+0.35');
  assert.match(lines[0].pvSan, /^e4 e5/);
  assert.equal(lines[1].scoreText, '#4');
  assert.equal(lines[1].detail, 'depth 18');
});

test('lc0AnalysisLines falls back to the single PV when multiPv is absent', () => {
  const result = { value: -0.2, visits: 16, pv: ['e2e4', 'e7e5'], children: [{ uci: 'e2e4', visits: 16, q: 0.2 }] };
  const lines = lc0AnalysisLines(result, START_FEN);
  assert.equal(lines.length, 1);
  assert.match(lines[0].pvSan, /^e4 e5/);
});
