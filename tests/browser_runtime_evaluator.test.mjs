import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { RuntimeFallbackEvaluator } from '../src/nn/runtimeFallbackEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function pendingEvaluation() {
  let reject;
  const promise = new Promise((_resolve, rejectPromise) => {
    reject = rejectPromise;
  });
  return { promise, reject };
}

test('runtime fallback does not create ORT after destruction', async () => {
  const pending = pendingEvaluation();
  let primaryDestroyed = false;
  let fallbackCreates = 0;
  const evaluator = new RuntimeFallbackEvaluator(
    {
      evaluate: () => pending.promise,
      destroy() { primaryDestroyed = true; },
    },
    async () => {
      fallbackCreates += 1;
      return { evaluate: async () => ({ policy: new Map(), wdl: [0, 1, 0] }) };
    },
    { source: 'test' },
  );
  const evaluation = evaluator.evaluate(parseFen(START_FEN));
  evaluator.destroy();
  pending.reject(new Error('device lost'));
  await assert.rejects(evaluation, (error) => error?.name === 'AbortError');
  assert.equal(primaryDestroyed, true);
  assert.equal(fallbackCreates, 0);
});

test('runtime fallback destroys ORT when creation completes after destruction', async () => {
  const pending = pendingEvaluation();
  let resolveFallback;
  let fallbackDestroyed = false;
  const fallbackCreated = new Promise((resolve) => {
    resolveFallback = resolve;
  });
  const evaluator = new RuntimeFallbackEvaluator(
    {
      evaluate: () => pending.promise,
      destroy() {},
    },
    () => fallbackCreated,
    { source: 'test' },
  );
  const evaluation = evaluator.evaluate(parseFen(START_FEN));
  pending.reject(new Error('device lost'));
  await Promise.resolve();
  evaluator.destroy();
  resolveFallback({
    evaluate: async () => ({ policy: new Map(), wdl: [0, 1, 0] }),
    destroy() { fallbackDestroyed = true; },
  });
  await assert.rejects(evaluation, (error) => error?.name === 'AbortError');
  assert.equal(fallbackDestroyed, true);
});

test('SquareFormer evaluator releases its ORT session once', async () => {
  let releases = 0;
  const evaluator = new SquareFormerEvaluator({
    release() { releases += 1; },
  }, {});
  evaluator.destroy();
  evaluator.destroy();
  await Promise.resolve();
  assert.equal(releases, 1);
});
