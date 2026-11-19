import assert from 'node:assert/strict';
import test from 'node:test';
import { BrokeredEvaluator } from '../src/nn/evaluator.ts';

test('broker timing samples retain the latest window while counters remain cumulative', async (t) => {
  let clock = 0;
  t.mock.method(performance, 'now', () => clock);
  let runs = 0;
  const broker = new BrokeredEvaluator(
    {
      evaluate() {},
      evaluateBatch(boards) {
        clock += ++runs;
        return boards.map(() => ({ policy: new Map(), wdl: [0, 1, 0] }));
      },
    },
    { maxBatchSize: 1 },
  );
  for (let i = 0; i < 1100; i++) await broker.evaluate({});
  const metrics = broker.metrics();
  assert.equal(metrics.waitMs.length, 1024);
  assert.deepEqual(
    metrics.runMs,
    Array.from({ length: 1024 }, (_, i) => 77 + i),
  );
  assert.equal(metrics.logicalRequests, 1100);
  assert.equal(metrics.positionsEvaluated, 1100);
  assert.equal(metrics.flushes, 1100);
  assert.equal(metrics.batchSizes[1], 1100);
  assert.equal(metrics.avgBatch, 1);
  metrics.waitMs.length = 0;
  metrics.runMs.length = 0;
  assert.equal(broker.metrics().runMs.length, 1024, 'metrics snapshots do not mutate the window');
});

test('broker bounds queue timing even when every evaluation fails', async () => {
  const broker = new BrokeredEvaluator(
    {
      evaluate() {
        throw new Error('evaluation failed');
      },
    },
    { maxBatchSize: 1 },
  );
  for (let i = 0; i < 1100; i++) await assert.rejects(broker.evaluate({}), /evaluation failed/);
  assert.equal(broker.metrics().waitMs.length, 1024);
  assert.equal(broker.metrics().runMs.length, 0);
  assert.equal(broker.metrics().flushes, 0);
});

test('a request larger than maxBatchSize is flushed alone, whole', async () => {
  const batches = [];
  const broker = new BrokeredEvaluator(
    {
      evaluate() {},
      async evaluateBatch(boards) {
        batches.push(boards.map((board) => board.id));
        await new Promise((resolve) => setTimeout(resolve, 0));
        return boards.map((board) => ({ policy: new Map(), wdl: [board.id, 0, 0] }));
      },
    },
    { maxBatchSize: 4, maxWaitMs: 0 },
  );
  const boards = (ids) => ids.map((id) => ({ id }));
  const small = broker.evaluateBatch(boards([1, 2]));
  const oversized = broker.evaluateBatch(boards([3, 4, 5, 6, 7, 8]));
  const tail = broker.evaluateBatch(boards([9]));
  const results = await Promise.all([small, oversized, tail]);
  assert.deepEqual(batches, [[1, 2], [3, 4, 5, 6, 7, 8], [9]]);
  assert.deepEqual(
    results.map((evals) => evals.map((e) => e.wdl[0])),
    [[1, 2], [3, 4, 5, 6, 7, 8], [9]],
  );
});
