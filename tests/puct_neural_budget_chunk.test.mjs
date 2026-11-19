import assert from 'node:assert/strict';
import { test } from 'node:test';
import { neuralBudgetChunk } from '../src/search/puct.ts';

test('neural budget chunk finishes one physical batch but never spends a second pipelined round', () => {
  assert.equal(neuralBudgetChunk(16, 4, 1000, 3), 16, 'a nearly spent budget still completes one batch');
  assert.equal(neuralBudgetChunk(16, 4, 1000, 40), 40);
  assert.equal(neuralBudgetChunk(16, 4, 1000, 500), 64, 'capped at batchSize * pipelineDepth');
  assert.equal(neuralBudgetChunk(16, 1, 1000, 500), 16);
  assert.equal(neuralBudgetChunk(16, 4, 5, 500), 5, 'capped by remaining root room');
  assert.equal(neuralBudgetChunk(16, 4, 0, 500), 1, 'always at least one visit');
});
