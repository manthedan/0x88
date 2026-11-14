import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { createWakeLockManager } from '../src/lc0/wakeLock.ts';

const deferred = () => {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
};
const sentinel = () => ({
  releases: 0,
  async release() {
    this.releases++;
  },
});

test('overlapping wake-lock requests resolve out of order and release their own tokens', async () => {
  const requests = [deferred(), deferred()];
  const locks = [sentinel(), sentinel()];
  let index = 0;
  const manager = createWakeLockManager(() => requests[index++].promise);
  const first = new AbortController(),
    second = new AbortController();
  const a = manager.acquire(first.signal),
    b = manager.acquire(second.signal);
  requests[1].resolve(locks[1]);
  const releaseB = await b;
  first.abort();
  requests[0].resolve(locks[0]);
  const releaseA = await a;
  assert.deepEqual(
    locks.map((lock) => lock.releases),
    [1, 0],
  );
  releaseA();
  assert.deepEqual(
    locks.map((lock) => lock.releases),
    [1, 0],
  );
  releaseB();
  releaseB();
  second.abort();
  assert.deepEqual(
    locks.map((lock) => lock.releases),
    [1, 1],
  );
});

test('abort releases an acquired sentinel immediately, with no request for an already aborted turn', async () => {
  const lock = sentinel();
  let requests = 0;
  const manager = createWakeLockManager(async () => {
    requests++;
    return lock;
  });
  const controller = new AbortController();
  const release = await manager.acquire(controller.signal);
  controller.abort();
  release();
  await manager.acquire(controller.signal);
  assert.equal(lock.releases, 1);
  assert.equal(requests, 1);
});

test('missing and denied APIs remain best effort', async () => {
  for (const request of [
    () => undefined,
    async () => {
      throw Error('denied');
    },
  ]) {
    const release = await createWakeLockManager(request).acquire(new AbortController().signal);
    assert.doesNotThrow(release);
  }
});

test('a turn aborted or disposed while acquiring a wake lock never renders or starts an engine search', async () => {
  const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const turn = source.slice(source.indexOf('async function ctxEngineTurn('), source.indexOf('function ctxOnUserMove('));
  for (const disposed of [false, true]) {
    const pending = deferred();
    let renders = 0,
      searches = 0,
      releases = 0;
    const ctx = { gameOver: null, board: { turn: 'b' }, humanColor: 'w', abort: new AbortController(), disposed: false };
    const dependencies = {
      wakeLocks: { acquire: () => pending.promise },
      el: () => ({ textContent: '' }),
      ctxRender: () => {
        renders++;
      },
      ctxRequestEngineMove: () => {
        searches++;
      },
      ctxSetEngineThinking: (target, thinking) => {
        target.engineThinking = thinking;
      },
    };
    const engineTurn = new Function(...Object.keys(dependencies), `${stripTypeScriptTypes(turn)}; return ctxEngineTurn;`)(...Object.values(dependencies));
    const running = engineTurn(ctx);
    if (disposed) ctx.disposed = true;
    else ctx.abort.abort();
    pending.resolve(() => {
      releases++;
    });
    await running;
    assert.equal(renders, 0);
    assert.equal(searches, 0);
    assert.equal(releases, 1);
  }
});

test('the thinking indicator only shows when an engine turn outlasts the show delay', () => {
  const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const helper = source.slice(source.indexOf('const THINKING_SHOW_DELAY_MS'), source.indexOf('function ctxCancelEngineTurn('));
  const timers = new Map();
  let nextTimer = 0,
    renders = 0;
  const dependencies = {
    setTimeout: (fn, ms) => {
      timers.set(++nextTimer, { fn, ms });
      return nextTimer;
    },
    clearTimeout: (id) => timers.delete(id),
    ctxRender: () => {
      renders++;
    },
  };
  const setThinking = new Function(...Object.keys(dependencies), `${stripTypeScriptTypes(helper)}; return ctxSetEngineThinking;`)(...Object.values(dependencies));
  const fire = () => {
    const pending = [...timers.values()];
    timers.clear();
    for (const timer of pending) timer.fn();
  };
  const ctx = { engineThinking: false, thinkingShown: false, thinkingShowTimer: null, disposed: false };

  // Fast reply: the turn ends before the delay, so nothing ever shows.
  setThinking(ctx, true);
  assert.equal(ctx.engineThinking, true);
  assert.equal(ctx.thinkingShown, false);
  setThinking(ctx, false);
  assert.equal(timers.size, 0);
  assert.equal(renders, 0);

  // Slow reply: shows once the delay elapses, and hides with the turn.
  setThinking(ctx, true);
  assert.equal([...timers.values()][0].ms, 400);
  fire();
  assert.equal(ctx.thinkingShown, true);
  assert.equal(renders, 1);
  setThinking(ctx, false);
  assert.equal(ctx.thinkingShown, false);

  // A disposed page never renders from a late timer.
  setThinking(ctx, true);
  ctx.disposed = true;
  fire();
  assert.equal(ctx.thinkingShown, false);
  assert.equal(renders, 1);
});
