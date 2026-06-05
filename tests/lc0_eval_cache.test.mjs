import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

function evaluation(fen, bestMove = 'e2e4') {
  return {
    fen,
    wdl: [0.4, 0.3, 0.3],
    q: 0.1,
    mlh: 80,
    legalPriors: [{ uci: bestMove, index: 0, logit: 1, prior: 1 }],
    bestMove,
  };
}

test('CachedLc0Evaluator reuses evaluations and exposes hit/miss metrics', async () => {
  const calls = [];
  const inner = {
    async evaluateBatch(inputs) {
      calls.push(inputs.map(String));
      return inputs.map((input) => evaluation(String(input)));
    },
  };
  const cached = new CachedLc0Evaluator(inner, { maxEntries: 8 });

  const first = await cached.evaluate(START_FEN);
  const second = await cached.evaluate(START_FEN);

  assert.equal(first.bestMove, 'e2e4');
  assert.equal(second.bestMove, 'e2e4');
  assert.equal(calls.length, 1, 'second evaluation is served from cache');
  assert.deepEqual(cached.metrics(), { hits: 1, misses: 1, entries: 1, maxEntries: 8 });
});

test('CachedLc0Evaluator preserves sequence batches for uncached misses', async () => {
  const sequenceCalls = [];
  const inner = {
    async evaluateBatchSequence(batches) {
      sequenceCalls.push(batches.map((batch) => batch.length));
      return batches.map((batch) => batch.map((input) => evaluation(String(input))));
    },
    async evaluateBatch(inputs) {
      return inputs.map((input) => evaluation(String(input)));
    },
  };
  const cached = new CachedLc0Evaluator(inner, { maxEntries: 8 });
  const batches = [['8/8/8/8/8/8/8/K6k w - - 0 1'], ['8/8/8/8/8/8/8/K5k1 w - - 0 1', '8/8/8/8/8/8/8/K4k2 w - - 0 1']];

  const first = await cached.evaluateBatchSequence(batches);
  const second = await cached.evaluateBatchSequence(batches);

  assert.deepEqual(first.map((batch) => batch.map((entry) => entry.bestMove)), [['e2e4'], ['e2e4', 'e2e4']]);
  assert.deepEqual(second.map((batch) => batch.map((entry) => entry.bestMove)), [['e2e4'], ['e2e4', 'e2e4']]);
  assert.deepEqual(sequenceCalls, [[1, 2]], 'cache forwards the first miss set as one sequence call');
  assert.deepEqual(cached.metrics(), { hits: 3, misses: 3, entries: 3, maxEntries: 8 });
});

test('CachedLc0Evaluator keeps bare FEN and explicit one-position history separate', async () => {
  let calls = 0;
  const fen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2';
  const inner = {
    async evaluate(input) {
      calls += 1;
      const explicit = typeof input === 'object' && input !== null && 'positions' in input;
      return evaluation(fen, explicit ? 'g1f3' : 'd2d4');
    },
  };
  const cached = new CachedLc0Evaluator(inner, { maxEntries: 8 });

  assert.equal((await cached.evaluate(fen)).bestMove, 'd2d4');
  assert.equal((await cached.evaluate({ positions: [fen] })).bestMove, 'g1f3');
  assert.equal(calls, 2, 'explicit history uses a distinct cache key from bare FEN input');
});

test('CachedLc0Evaluator evicts least-recently-used entries when resized', async () => {
  let calls = 0;
  const inner = {
    async evaluate(input) {
      calls += 1;
      return evaluation(String(input));
    },
  };
  const cached = new CachedLc0Evaluator(inner, { maxEntries: 2 });

  await cached.evaluate('8/8/8/8/8/8/8/K6k w - - 0 1');
  await cached.evaluate('8/8/8/8/8/8/8/K5k1 w - - 0 1');
  await cached.evaluate('8/8/8/8/8/8/8/K4k2 w - - 0 1');
  assert.equal(cached.metrics().entries, 2, 'cache is capped');

  await cached.evaluate('8/8/8/8/8/8/8/K6k w - - 0 1');
  assert.equal(calls, 4, 'oldest entry was evicted and recomputed');

  cached.setMaxEntries(0);
  assert.equal(cached.metrics().entries, 0, 'resizing to zero clears cached entries');
});

test('CachedLc0Evaluator clears entries and forwards dispose to the wrapped evaluator', async () => {
  let disposed = 0;
  const inner = {
    async evaluate(input) { return evaluation(String(input)); },
    async dispose() { disposed += 1; },
  };
  const cached = new CachedLc0Evaluator(inner, { maxEntries: 8 });
  await cached.evaluate(START_FEN);
  assert.equal(cached.metrics().entries, 1);

  await cached.dispose();

  assert.equal(disposed, 1);
  assert.deepEqual(cached.metrics(), { hits: 0, misses: 0, entries: 0, maxEntries: 8 });
});

test('Lc0OnnxEvaluator releases its ORT session at most once', async () => {
  let releases = 0;
  const session = {
    inputMetadata: [{ name: '/input/planes', type: 'float32', shape: [1, 112, 8, 8] }],
    async run() { throw new Error('not used'); },
    async release() { releases += 1; },
  };
  const evaluator = new Lc0OnnxEvaluator(session);

  await evaluator.dispose();
  await evaluator.dispose();

  assert.equal(releases, 1);
  await assert.rejects(() => evaluator.evaluate(START_FEN), /disposed/);
});
