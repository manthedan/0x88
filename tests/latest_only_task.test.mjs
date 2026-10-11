import assert from 'node:assert/strict';
import test from 'node:test';
import { LatestOnlyTask } from '../src/lc0/latestOnlyTask.ts';

const turn = () => new Promise((resolve) => setImmediate(resolve));
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}

test('latest-only inference keeps one active and replaces the pending board', async () => {
  const started = [];
  const displayed = [];
  const first = deferred();
  let active = 0;
  let peak = 0;
  const task = new LatestOnlyTask(async (value, isCurrent) => {
    started.push(value);
    peak = Math.max(peak, ++active);
    if (value === 1) await first.promise;
    if (isCurrent()) displayed.push(value);
    active -= 1;
  }, assert.fail);
  task.request(1);
  task.request(2);
  task.request(3);
  assert.deepEqual(started, [1]);
  first.resolve();
  await turn();
  assert.deepEqual(started, [1, 3]);
  assert.deepEqual(displayed, [3]);
  assert.equal(peak, 1);
});

test('navigation invalidates results immediately, before debounce dispatch', async () => {
  const first = deferred();
  const displayed = [];
  const task = new LatestOnlyTask(async (value, current) => {
    await first.promise;
    if (current()) displayed.push(value);
  }, assert.fail);
  task.request('old');
  task.invalidate();
  first.resolve();
  await turn();
  assert.deepEqual(displayed, []);
  task.request('new');
  await turn();
  assert.deepEqual(displayed, ['new']);
});

test('a rejected stale inference always drains the latest board', async () => {
  const first = deferred();
  const started = [];
  const errors = [];
  const task = new LatestOnlyTask(
    async (value) => {
      started.push(value);
      if (value === 1) await first.promise;
    },
    (error) => errors.push(error),
  );
  task.request(1);
  task.request(2);
  first.reject(Error('old GPU error'));
  await turn();
  assert.deepEqual(started, [1, 2]);
  assert.deepEqual(errors, []);
});

test('a current inference error does not strand later requests, even if the error handler throws', async () => {
  const started = [];
  const task = new LatestOnlyTask(
    async (value) => {
      started.push(value);
      if (value === 1) throw Error('GPU error');
    },
    () => {
      throw Error('UI gone');
    },
  );
  task.request(1);
  await turn();
  task.request(2);
  await turn();
  assert.deepEqual(started, [1, 2]);
});

test('disposal drops pending work and a fresh task is independent of the old completion', async () => {
  const old = deferred();
  const started = [];
  const displayed = [];
  const task = new LatestOnlyTask(async (value, current) => {
    started.push(value);
    await old.promise;
    if (current()) displayed.push(value);
  }, assert.fail);
  task.request('old-active');
  task.request('old-pending');
  task.dispose();
  task.request('after-dispose');
  const fresh = new LatestOnlyTask(async (value, current) => {
    if (current()) displayed.push(value);
  }, assert.fail);
  fresh.request('fresh');
  old.resolve();
  await turn();
  assert.deepEqual(started, ['old-active']);
  assert.deepEqual(displayed, ['fresh']);
});
