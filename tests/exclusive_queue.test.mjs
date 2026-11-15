import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ExclusiveQueue } from '../src/lc0/exclusiveQueue.ts';

const flush = async () => {
  for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function observe(promise) {
  const state = { status: 'pending' };
  promise.then(
    (value) => Object.assign(state, { status: 'resolved', value }),
    (error) => Object.assign(state, { status: 'rejected', error }),
  );
  return state;
}

test('tasks run one at a time, in call order, whether earlier ones succeed or fail', async () => {
  const queue = new ExclusiveQueue();
  const log = [];
  const first = deferred();
  const a = queue.run(async () => {
    log.push('a:start');
    await first.promise;
    log.push('a:end');
  });
  const b = queue.run(async () => {
    log.push('b');
    throw new Error('b failed');
  });
  const c = queue.run(async () => {
    log.push('c');
    return 'c';
  });
  await flush();
  assert.deepEqual(log, ['a:start']);
  first.resolve();
  await a;
  await assert.rejects(b, /b failed/);
  assert.equal(await c, 'c');
  assert.deepEqual(log, ['a:start', 'a:end', 'b', 'c']);
});

test('a task aborted while it waits leaves at once; the tasks behind it keep their order', async () => {
  const queue = new ExclusiveQueue(() => Object.assign(new Error('engine aborted'), { name: 'AbortError' }));
  const first = deferred();
  const log = [];
  const active = observe(queue.run(() => first.promise));
  const controller = new AbortController();
  const cancelled = observe(queue.run(async () => log.push('cancelled ran'), controller.signal));
  const after = queue.run(async () => log.push('after'));
  await flush();
  controller.abort();
  await flush();
  assert.equal(cancelled.status, 'rejected');
  assert.equal(cancelled.error.name, 'AbortError');
  assert.equal(cancelled.error.message, 'engine aborted');
  assert.equal(active.status, 'pending', 'the running task is not interrupted');
  assert.deepEqual(log, [], 'nothing behind the running task has started');
  first.resolve();
  await after;
  assert.deepEqual(log, ['after']);
});

test('an already-aborted signal rejects without running the task', async () => {
  const queue = new ExclusiveQueue();
  const controller = new AbortController();
  controller.abort();
  let ran = false;
  await assert.rejects(
    queue.run(async () => {
      ran = true;
    }, controller.signal),
    { name: 'AbortError' },
  );
  assert.equal(ran, false);
});

test('aborting a running task does not reject it from the queue side', async () => {
  const queue = new ExclusiveQueue();
  const controller = new AbortController();
  const gate = deferred();
  const running = observe(queue.run(() => gate.promise, controller.signal));
  await flush();
  controller.abort();
  await flush();
  assert.equal(running.status, 'pending');
  gate.resolve('done');
  await flush();
  assert.equal(running.value, 'done');
});

test('idle() settles after every task queued so far, including failed ones', async () => {
  const queue = new ExclusiveQueue();
  const gate = deferred();
  const work = queue.run(() => gate.promise);
  const idle = observe(queue.idle());
  await flush();
  assert.equal(idle.status, 'pending');
  gate.reject(new Error('task failed'));
  await assert.rejects(work, /task failed/);
  await flush();
  assert.equal(idle.status, 'resolved');
});
