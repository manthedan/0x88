import assert from 'node:assert/strict';
import test from 'node:test';
import { workerFallbackReplayAbort } from '../src/lc0/ortWorkerFallback.ts';

test('search cancellation prevents replay after ORT worker fallback', () => {
  const error = workerFallbackReplayAbort('search', true);
  assert.equal(error?.name, 'AbortError');
  assert.match(error?.message ?? '', /cancelled during ORT worker fallback/);
});

test('non-search requests and active searches remain replayable', () => {
  assert.equal(workerFallbackReplayAbort('evaluate', true), undefined);
  assert.equal(workerFallbackReplayAbort('evaluateBatch', true), undefined);
  assert.equal(workerFallbackReplayAbort('search', false), undefined);
});
