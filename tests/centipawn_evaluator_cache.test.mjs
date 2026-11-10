import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createCentipawnEvaluatorCache } from '../src/lc0/centipawnEvaluatorCache.ts';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};
const evaluator = () => ({
  destroyed: 0,
  destroy() {
    this.destroyed++;
  },
});
const request = { spec: { onnx: 'model', meta: 'meta' }, options: { audit: { surface: 'analysis' } }, maxEntries: 2, label: 'test' };

test('an old rejection never deletes a newer in-flight load for the same key', async () => {
  const loads = [deferred(), deferred()];
  let index = 0;
  const cache = createCentipawnEvaluatorCache(() => loads[index++].promise);
  const old = cache.get('same', request, () => true);
  const rejected = assert.rejects(old, /old aborted/);
  await Promise.resolve();
  cache.retain(new Set());
  const current = cache.get('same', request, () => true);
  await Promise.resolve();
  loads[0].reject(Error('old aborted'));
  await rejected;
  assert.equal(
    cache.get('same', request, () => true),
    current,
  );
  const inner = evaluator();
  loads[1].resolve({ evaluator: inner });
  assert.equal((await current).inner, inner);
  assert.equal(index, 2);
  cache.clear();
  assert.equal(inner.destroyed, 1);
});

test('a late successful old load is destroyed and cannot overwrite the current evaluator', async () => {
  const loads = [deferred(), deferred()];
  const inner = [evaluator(), evaluator()];
  let index = 0;
  const cache = createCentipawnEvaluatorCache(() => loads[index++].promise);
  const old = cache.get('same', request, () => true);
  const rejected = assert.rejects(old, { name: 'AbortError' });
  await Promise.resolve();
  cache.clear();
  const current = cache.get('same', request, () => true);
  await Promise.resolve();
  loads[1].resolve({ evaluator: inner[1] });
  await current;
  loads[0].resolve({ evaluator: inner[0] });
  await rejected;
  assert.equal(
    cache.get('same', request, () => true),
    current,
  );
  assert.deepEqual(
    inner.map((item) => item.destroyed),
    [1, 0],
  );
  cache.clear();
  cache.clear();
  assert.deepEqual(
    inner.map((item) => item.destroyed),
    [1, 1],
  );
});

test('retention preserves active entries and Analysis run snapshots; inactive resolutions are destroyed', async () => {
  const inner = evaluator();
  const cache = createCentipawnEvaluatorCache(async () => ({ evaluator: inner }));
  const current = cache.get('held-by-run', request, () => true);
  await current;
  cache.retain(new Set(['held-by-run']));
  assert.equal(
    cache.get('held-by-run', request, () => true),
    current,
  );
  cache.retain(new Set());
  assert.equal(inner.destroyed, 1);
  const inactive = evaluator();
  const other = createCentipawnEvaluatorCache(async () => ({ evaluator: inactive }));
  await assert.rejects(
    other.get('removed', request, () => false),
    { name: 'AbortError' },
  );
  assert.equal(inactive.destroyed, 1);
});
