import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';

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
