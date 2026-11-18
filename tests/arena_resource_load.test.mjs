import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadRetirableResource } from '../src/lc0/arenaResourceLoad.ts';
import { awaitEngineRetirements, hasEngineRetirements } from '../src/lc0/engineRetirement.ts';

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('Arena resource loading returns a live resource without disposing it', async () => {
  let disposed = false;
  const resource = {
    dispose: () => {
      disposed = true;
    },
  };
  assert.equal(await loadRetirableResource(async () => resource, new AbortController().signal), resource);
  assert.equal(disposed, false);
});

test('Stop rejects immediately and disposes an evaluator that finishes loading late', async () => {
  const controller = new AbortController();
  let finish;
  let disposed = 0;
  const pending = loadRetirableResource(
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

test('a canceled native creation and its asynchronous release block the next model until both settle', async () => {
  const creation = Promise.withResolvers(),
    release = Promise.withResolvers(),
    controller = new AbortController();
  const loading = loadRetirableResource(() => creation.promise, controller.signal);
  await tick();
  controller.abort();
  await assert.rejects(loading, { name: 'AbortError' });
  assert.equal(hasEngineRetirements(), true, 'cancellation publishes cleanup synchronously');
  let admitted = false,
    disposed = false;
  const waiting = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  await tick();
  assert.equal(admitted, false);
  creation.resolve({
    dispose: () => {
      disposed = true;
      return release.promise;
    },
  });
  await tick();
  assert.equal(disposed, true);
  assert.equal(admitted, false);
  release.resolve();
  await waiting;
  assert.equal(admitted, true);
});

test('pre-aborted or immediately stopped initialization never starts a loader', async () => {
  const controller = new AbortController();
  controller.abort();
  assert.throws(() => loadRetirableResource(() => assert.fail(), controller.signal), { name: 'AbortError' });
  const another = new AbortController();
  const pending = loadRetirableResource(() => assert.fail('must not start after cancellation'), another.signal);
  another.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  await tick();
});

test('Arena initialization errors propagate without an unhandled late rejection', async () => {
  const failure = new Error('Model unavailable');
  await assert.rejects(
    loadRetirableResource(async () => {
      throw failure;
    }, new AbortController().signal),
    (error) => error === failure,
  );
  const controller = new AbortController();
  let fail;
  const pending = loadRetirableResource(
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
