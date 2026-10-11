import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { analysisResultCacheKey, cachedReviewPosition } from '../src/lc0/analysisCache.ts';
import { evalBarWhitePercent, stockfishAnalysisLines } from '../src/lc0/analysisFormat.ts';

const row = { family: 'sf', variant: 'lite', strength: 12 };
const settings = { multiPv: 1, runtime: '', cpu: 'balanced/2/sf-lite' };
const key = (r = row, s = settings, history = [START_FEN]) => analysisResultCacheKey(START_FEN, history, r, s);

test('review cache identity requires exact engine, depth, history, MultiPV and runtime settings', () => {
  assert.equal(key(), key({ ...row }, { ...settings }, [START_FEN]));
  assert.notEqual(key(), key({ ...row, variant: 'full' }));
  assert.notEqual(key(), key({ ...row, strength: 13 }));
  assert.notEqual(key(), key({ ...row, family: 'berserk' }));
  assert.notEqual(key(), key(row, { ...settings, multiPv: 3 }));
  assert.notEqual(key(), key(row, { ...settings, runtime: 'wasm' }));
  assert.notEqual(key(), key(row, { ...settings, cpu: 'eco/1/sf-lite' }));
  assert.notEqual(key(), key(row, settings, [START_FEN, START_FEN]));
});

test('cached review scores preserve White perspective on Black-to-move boards', () => {
  const fen = START_FEN.replace(' w ', ' b ');
  for (const score of [{ scoreCp: 120 }, { mateIn: 3 }, { mateIn: -2 }]) {
    const infos = [{ multipv: 1, depth: 12, pvUci: ['e7e5'], ...score }];
    const lines = stockfishAnalysisLines(infos, fen, 'SF Lite');
    const cached = cachedReviewPosition(lines, 20);
    assert.deepEqual(cached, {
      winWhite: evalBarWhitePercent(score.scoreCp === undefined ? undefined : -score.scoreCp, score.mateIn === undefined ? undefined : -score.mateIn) / 100,
      bestUci: 'e7e5',
      legalMoves: 20,
    });
  }
});

test('review ignores errors, missing top lines and unscored cache entries', () => {
  assert.equal(cachedReviewPosition(undefined, 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 2, scoreCp: 10 }], 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 1, error: 'failed', scoreCp: 10 }], 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 1 }], 20), undefined);
});
