import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, START_FEN } from '../src/chess/board.ts';
import { analysisResultCacheKey, cachedAnalysisSelection, cachedReviewPosition } from '../src/lc0/analysisCache.ts';
import { evalBarWhitePercent, stockfishAnalysisLines } from '../src/lc0/analysisFormat.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

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

test('completed selections restore the right history after a same-FEN transposition', () => {
  const histories = [
    ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'],
    ['g1f3', 'g8f6', 'f3h4', 'f6h5', 'h4f3', 'h5f6', 'f3g1', 'f6g8'],
  ].map((moves) => {
    const game = new GameTree();
    for (const move of moves) game.addUci(move);
    return game;
  });
  assert.equal(histories[0].current.fen, histories[1].current.fen);
  const keys = histories.map((game) => analysisResultCacheKey(game.current.fen, game.historyBoards().map(boardToFen), row, settings));
  assert.notEqual(keys[0], keys[1]);
  const a = stockfishAnalysisLines([{ multipv: 1, depth: 12, scoreCp: 0, pvUci: ['g1f3'] }], histories[0].current.fen, 'SF Lite');
  const b = stockfishAnalysisLines([{ multipv: 1, depth: 12, scoreCp: 16, pvUci: ['e2e4'] }], histories[1].current.fen, 'SF Lite');
  const cache = new Map([
    [keys[0], a],
    [keys[1], b],
  ]);
  assert.deepEqual(cachedAnalysisSelection(cache, [keys[0]]), a);
  assert.deepEqual(cachedAnalysisSelection(cache, [keys[1]]), b);
  assert.deepEqual(cachedAnalysisSelection(cache, [keys[1], keys[0]]), [...b, ...a]);
  cache.delete(keys[0]);
  assert.equal(cachedAnalysisSelection(cache, [keys[0]]), undefined, 'a completed marker must not hide an evicted result');
  assert.equal(cachedAnalysisSelection(cache, [keys[1], keys[0]]), undefined, 'do not restore a partial engine selection');
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
