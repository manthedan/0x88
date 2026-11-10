import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadArenaResource } from '../src/lc0/arenaResourceLoad.ts';

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('Arena resource loading returns a live resource without disposing it', async () => {
  let disposed = false;
  const resource = {
    dispose: () => {
      disposed = true;
    },
  };
  assert.equal(await loadArenaResource(async () => resource, new AbortController().signal), resource);
  assert.equal(disposed, false);
});

test('Stop rejects immediately and disposes an evaluator that finishes loading late', async () => {
  const controller = new AbortController();
  let finish;
  let disposed = 0;
  const pending = loadArenaResource(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
    controller.signal,
  );
  await tick();
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  finish({
    dispose: () => {
      disposed++;
    },
  });
  await tick();
  assert.equal(disposed, 1);
});

test('pre-aborted or immediately stopped initialization never starts a loader', async () => {
  const controller = new AbortController();
  controller.abort();
  assert.throws(() => loadArenaResource(() => assert.fail(), controller.signal), { name: 'AbortError' });
  const another = new AbortController();
  const pending = loadArenaResource(() => assert.fail('must not start after cancellation'), another.signal);
  another.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  await tick();
});

test('Arena initialization errors propagate without an unhandled late rejection', async () => {
  const failure = new Error('Model unavailable');
  await assert.rejects(
    loadArenaResource(async () => {
      throw failure;
    }, new AbortController().signal),
    (error) => error === failure,
  );
  const controller = new AbortController();
  let fail;
  const pending = loadArenaResource(
    () =>
      new Promise((_, reject) => {
        fail = reject;
      }),
    controller.signal,
  );
  await tick();
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  fail(failure);
  await tick();
});
