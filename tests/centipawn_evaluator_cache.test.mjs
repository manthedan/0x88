import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createCentipawnEvaluatorCache } from '../src/lc0/centipawnEvaluatorCache.ts';
import { awaitEngineRetirements } from '../src/lc0/engineRetirement.ts';

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
  const rejected = assert.rejects(old, { name: 'AbortError' });
  await new Promise(setImmediate);
  cache.retain(new Set());
  const current = cache.get('same', request, () => true);
  await new Promise(setImmediate);
  loads[0].reject(Error('old aborted'));
  await rejected;
  await awaitEngineRetirements();
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
  await new Promise(setImmediate);
  cache.clear();
  const current = cache.get('same', request, () => true);
  await new Promise(setImmediate);
  loads[1].resolve({ evaluator: inner[1] });
  await current;
  loads[0].resolve({ evaluator: inner[0] });
  await rejected;
  await awaitEngineRetirements();
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

for (const pending of [false, true])
  test(`shared Centipawn cache tracks loaded and late native releases (${pending})`, async () => {
    const native = deferred(),
      release = deferred();
    let calls = 0,
      signal;
    const inner = {
      dispose: () => {
        calls++;
        return release.promise;
      },
      destroy: () => assert.fail('must await dispose, not void destroy'),
    };
    const cache = createCentipawnEvaluatorCache(async (_spec, options) => {
      signal = options.signal;
      return native.promise;
    });
    const loading = cache.get('owned', request, () => true),
      canceled = pending ? assert.rejects(loading, { name: 'AbortError' }) : null;
    await new Promise(setImmediate);
    if (!pending) {
      native.resolve({ evaluator: inner });
      await loading;
    }
    cache.clear();
    assert.equal(cache.has('owned'), false);
    assert.equal(signal.aborted, true);
    let retired = false;
    const waiting = awaitEngineRetirements().then(() => {
      retired = true;
    });
    if (pending) {
      await canceled;
      native.resolve({ evaluator: inner });
    }
    await new Promise(setImmediate);
    assert.equal(calls, 1);
    assert.equal(retired, false);
    release.resolve();
    await waiting;
    assert.equal(retired, true);
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
