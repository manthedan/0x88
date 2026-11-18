import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { disposeOrtWorker } from '../src/lc0/disposeOrtWorker.ts';

class FakeSearcher {
  constructor(config) {
    this.config = config;
    this.disposeCalls = 0;
  }

  dispose() {
    this.disposeCalls += 1;
  }
}

function fakeScheduler() {
  const timers = [];
  return {
    timers,
    scheduler: {
      setTimeout(fn, delay) {
        const timer = { fn, delay, cleared: false };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        timer.cleared = true;
      },
    },
    runNext() {
      const timer = timers.find((entry) => !entry.cleared);
      assert(timer, 'expected a live timer');
      timer.cleared = true;
      timer.fn();
    },
  };
}

test('BigNetSessionPool keeps a released searcher alive until the idle TTL fires', () => {
  const { scheduler, timers, runNext } = fakeScheduler();
  const pool = new BigNetSessionPool((config) => new FakeSearcher(config), scheduler, 500);
  const searcher = pool.acquire('bt4');
  pool.release('bt4');

  assert.equal(pool.has('bt4'), true);
  assert.equal(timers[0].delay, 500);
  assert.equal(searcher.disposeCalls, 0);

  runNext();
  assert.equal(pool.has('bt4'), false);
  assert.equal(searcher.disposeCalls, 1);
});

test('BigNetSessionPool reacquire cancels pending idle disposal and reuses the session', () => {
  const { scheduler, timers } = fakeScheduler();
  const pool = new BigNetSessionPool((config) => new FakeSearcher(config), scheduler, 500);
  const first = pool.acquire('t3');
  pool.release('t3');
  const second = pool.acquire('t3');

  assert.equal(second, first);
  assert.equal(timers[0].cleared, true);
  assert.equal(first.disposeCalls, 0);
});

test('BigNetSessionPool peek does not cancel pending idle disposal', () => {
  const { scheduler, timers, runNext } = fakeScheduler();
  const pool = new BigNetSessionPool((config) => new FakeSearcher(config), scheduler, 500);
  const searcher = pool.acquire('bt4');
  pool.release('bt4');

  assert.equal(pool.peek('bt4'), searcher);
  assert.equal(timers[0].cleared, false);

  runNext();
  assert.equal(pool.has('bt4'), false);
  assert.equal(searcher.disposeCalls, 1);
});

test('BigNetSessionPool releaseUnused schedules only inactive retained nets', () => {
  const { scheduler, timers, runNext } = fakeScheduler();
  const pool = new BigNetSessionPool((config) => new FakeSearcher(config), scheduler, 250);
  const bt4 = pool.acquire('bt4');
  const t3 = pool.acquire('t3');

  pool.releaseUnused(['bt4']);

  assert.equal(pool.has('bt4'), true);
  assert.equal(pool.has('t3'), true);
  assert.equal(timers.length, 1);
  runNext();
  assert.equal(pool.has('bt4'), true);
  assert.equal(pool.has('t3'), false);
  assert.equal(bt4.disposeCalls, 0);
  assert.equal(t3.disposeCalls, 1);
});

test('detaching the owned instance cancels its TTL and a destination acquires a fresh worker', () => {
  const { scheduler, timers } = fakeScheduler();
  const pool = new BigNetSessionPool((config) => new FakeSearcher(config), scheduler, 500);
  const retired = pool.acquire('bt4');
  pool.release('bt4');
  assert.equal(pool.detach('bt4', retired), true);
  assert.equal(timers[0].cleared, true);
  assert.equal(retired.disposeCalls, 0, 'old owner still drains its in-flight work');
  const destination = pool.acquire('bt4');
  assert.notEqual(destination, retired);
  assert.equal(pool.detach('bt4', retired), false, 'late cleanup cannot detach a replacement');
  retired.dispose();
  assert.equal(destination.disposeCalls, 0);
  assert.equal(pool.peek('bt4'), destination);
  pool.disposeAllNow();
});

test('identity-checked detach cannot remove another owner or a different key', () => {
  const pool = new BigNetSessionPool((config) => new FakeSearcher(config));
  const bt4 = pool.acquire('bt4');
  const t3 = pool.acquire('t3');
  assert.equal(pool.detach('t3', bt4), false);
  assert.equal(pool.peek('t3'), t3);
  assert.equal(pool.detach('bt4', bt4), true);
  assert.equal(pool.detach('bt4', bt4), false);
  pool.disposeAllNow();
  assert.equal(bt4.disposeCalls, 0);
  assert.equal(t3.disposeCalls, 1);
  bt4.dispose();
});

test('constrained exclusive acquire disposes the previous net before creating a different one', () => {
  const events = [];
  const { scheduler, timers } = fakeScheduler();
  const pool = new BigNetSessionPool(
    (config) => {
      events.push(`create:${config.key}`);
      const searcher = new FakeSearcher(config);
      searcher.dispose = () => events.push(`dispose:${config.key}`);
      return searcher;
    },
    scheduler,
    500,
    () => true,
  );
  pool.acquire('bt4', true);
  pool.acquire('t3', true);
  pool.acquire('lqo', true);
  assert.deepEqual(events, ['create:bt4', 'dispose:bt4', 'create:t3', 'dispose:t3', 'create:lqo']);
  assert.equal(pool.has('bt4'), false);
  assert.equal(pool.has('t3'), false);
  assert.equal(timers.length, 0);
});

test('constrained pool reuses its one slot, but release never retains a retired net', () => {
  const { scheduler, timers } = fakeScheduler();
  const pool = new BigNetSessionPool(
    (config) => new FakeSearcher(config),
    scheduler,
    500,
    () => true,
  );
  const searcher = pool.acquire('bt4');
  assert.equal(pool.acquire('bt4'), searcher);
  pool.releaseUnused(['bt4']);
  assert.equal(searcher.disposeCalls, 0);
  pool.release('bt4', 180000);
  assert.equal(searcher.disposeCalls, 1);
  assert.equal(pool.has('bt4'), false);
  assert.equal(timers.length, 0);
});

test('constrained shared callers retain active sessions until they retire them (Analysis parallel init)', () => {
  const pool = new BigNetSessionPool(
    (config) => new FakeSearcher(config),
    undefined,
    undefined,
    () => true,
  );
  const bt4 = pool.acquire('bt4');
  const t3 = pool.acquire('t3');
  assert.equal(pool.acquire('bt4'), bt4);
  assert.equal(pool.acquire('t3'), t3);
  assert.equal(bt4.disposeCalls, 0);
  assert.equal(t3.disposeCalls, 0);
  pool.releaseUnused(['t3']);
  assert.equal(bt4.disposeCalls, 1);
  assert.equal(t3.disposeCalls, 0);
  pool.disposeAllNow();
});

test('desktop retains active sessions even when Arena requests exclusive residency', () => {
  const pool = new BigNetSessionPool(
    (config) => new FakeSearcher(config),
    undefined,
    undefined,
    () => false,
  );
  const bt4 = pool.acquire('bt4', true);
  const t3 = pool.acquire('t3', true);
  assert.equal(pool.peek('bt4'), bt4);
  assert.equal(pool.peek('t3'), t3);
  assert.equal(bt4.disposeCalls, 0);
  pool.disposeAllNow();
});

test('constrained acquire also clears an inherited idle-TTL net before replacement', () => {
  let constrained = false;
  const { scheduler, timers } = fakeScheduler();
  let first;
  const pool = new BigNetSessionPool(
    (config) => {
      if (config.key === 't3') assert.equal(first.disposeCalls, 1, 'old worker must be terminated before replacement');
      return new FakeSearcher(config);
    },
    scheduler,
    180000,
    () => constrained,
  );
  first = pool.acquire('bt4');
  pool.release('bt4');
  constrained = true;
  pool.acquire('t3');
  assert.equal(timers[0].cleared, true);
  assert.equal(pool.has('bt4'), false);
});

for (const ack of [true, false]) {
  test(`phone exclusive replacement awaits graceful disposal (${ack ? 'ack' : 'bounded timeout'})`, async () => {
    const events = [];
    const disposing = Promise.withResolvers();
    class Worker extends EventTarget {
      postMessage(message) {
        events.push(message.type);
        if (message.type === 'dispose') disposing.resolve();
      }
      terminate() {
        events.push('terminate');
      }
    }
    const worker = new Worker();
    const pool = new BigNetSessionPool(
      (config) => {
        events.push(`create:${config.key}`);
        return { dispose: () => disposeOrtWorker(worker, 25) };
      },
      undefined,
      undefined,
      () => true,
    );
    const first = pool.acquire('bt4');
    const next = pool.acquireAsync('t3', true);
    await disposing.promise;
    assert.equal(pool.has('bt4'), false, 'detach remains synchronous at retirement');
    assert.equal(pool.has('t3'), false, 'replacement is not constructed while the old worker drains');
    assert.deepEqual(events, ['create:bt4', 'cancel', 'dispose']);
    if (ack) worker.dispatchEvent(new MessageEvent('message', { data: { id: -1, type: 'disposed' } }));
    const replacement = await next;
    assert.deepEqual(events, ['create:bt4', 'cancel', 'dispose', 'terminate', 'create:t3']);
    assert.notEqual(replacement, first);
    assert.equal(pool.detach('bt4', first), false);
    assert.equal(pool.peek('t3'), replacement);
  });
}

test('async phone acquire waits for an already released net and preserves active Analysis rows', async () => {
  const retirement = Promise.withResolvers();
  const pool = new BigNetSessionPool(
    (config) => ({ config, dispose: () => retirement.promise }),
    undefined,
    undefined,
    () => true,
  );
  const bt4 = pool.acquire('bt4');
  const t3 = await pool.acquireAsync('t3');
  assert.equal(pool.peek('bt4'), bt4, 'shared acquisition does not evict an active row');
  pool.release('bt4', 0);
  let acquired = false;
  const next = pool.acquireAsync('lqo').then((value) => {
    acquired = true;
    return value;
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(acquired, false);
  retirement.resolve();
  await next;
  assert.equal(pool.peek('t3'), t3);
});

test('cancelled graceful retirement does not construct a replacement; a failed one ends the wait', async () => {
  for (const cancel of [false, true]) {
    const retirement = Promise.withResolvers();
    const pool = new BigNetSessionPool(
      () => ({ dispose: () => retirement.promise }),
      undefined,
      undefined,
      () => true,
    );
    pool.acquire('bt4');
    const abort = new AbortController();
    const next = pool.acquireAsync('t3', true, abort.signal);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(pool.has('t3'), false, 'no replacement while the old worker drains');
    if (cancel) {
      abort.abort();
      retirement.resolve();
      await assert.rejects(next, { name: 'AbortError' });
      assert.equal(pool.has('t3'), false);
    } else {
      // Worker disposal terminates even when release reports an error (disposeOrtWorker).
      retirement.reject(new Error('release failed'));
      await next;
      assert.equal(pool.has('t3'), true);
    }
  }
});

test('default scheduler calls the timer globals unbound (browsers reject a foreign receiver)', () => {
  const realSet = globalThis.setTimeout;
  const realClear = globalThis.clearTimeout;
  const calls = [];
  globalThis.setTimeout = function (_fn, delay) {
    if (this !== undefined && this !== globalThis) throw new TypeError('Illegal invocation');
    calls.push(['set', delay]);
    return 7;
  };
  globalThis.clearTimeout = function (handle) {
    if (this !== undefined && this !== globalThis) throw new TypeError('Illegal invocation');
    calls.push(['clear', handle]);
  };
  try {
    const pool = new BigNetSessionPool((config) => new FakeSearcher(config));
    pool.acquire('t3');
    pool.release('t3', 1000);
    pool.acquire('t3');
    assert.deepEqual(calls, [
      ['set', 1000],
      ['clear', 7],
    ]);
  } finally {
    globalThis.setTimeout = realSet;
    globalThis.clearTimeout = realClear;
  }
});
