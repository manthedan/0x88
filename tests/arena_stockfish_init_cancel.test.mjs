import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';

// Arena's serial/mixed and threaded pools use bestMove for warmup; private
// CPU games additionally exercise prewarm/newGame. No real worker is started.
for (const phase of ['uci', 'isready']) {
  for (const command of ['prewarm', 'newGame', 'bestMove', 'analyze']) {
    test(`Stop during Stockfish ${command}/${phase} settles without waiting for startup`, async (t) => {
      const workers = [];
      replaceGlobal(
        t,
        'Worker',
        class {
          terminated = false;
          constructor() {
            workers.push(this);
          }
          postMessage(message) {
            if (message === 'uci' && phase === 'isready') queueMicrotask(() => this.onmessage?.({ data: 'uciok' }));
          }
          terminate() {
            this.terminated = true;
          }
        },
      );
      replaceGlobal(t, 'location', { href: 'https://arena.test/app/arena/', origin: 'https://arena.test' });
      const engine = new StockfishEngine({}, '/stockfish/stockfish-19-lite.js');
      t.after(() => engine.dispose());
      const controller = new AbortController();
      let outcome = 'pending';
      const pending = (
        command === 'analyze'
          ? engine.analyze(START_FEN, { signal: controller.signal })
          : command === 'bestMove'
            ? engine.bestMove(START_FEN, controller.signal)
            : engine[command](controller.signal)
      ).then(
        () => {
          outcome = 'resolved';
        },
        (error) => {
          outcome = error.name;
        },
      );
      await new Promise(setImmediate);
      assert.equal(workers.length, 1);
      controller.abort();
      await new Promise(setImmediate);
      assert.equal(outcome, 'AbortError', 'Stop must not wait for the 120-second startup timeout');
      // Startup is shared by every later request. Analysis cancels on each
      // navigation, so a cancelled waiter must not kill a cold start.
      assert.ok(
        workers.every((worker) => !worker.terminated),
        'a cancelled waiter does not kill the startup in progress',
      );
      await pending;
      const retry = engine.prewarm().catch(() => {});
      await new Promise(setImmediate);
      assert.equal(workers.length, 1, 'the next request reuses the startup in progress');
      engine.dispose();
      await retry;
      assert.equal(workers[0].terminated, true, 'dispose() releases the startup worker');
    });
  }
}

test('a pre-aborted Stockfish warmup never creates a worker', async (t) => {
  let workers = 0;
  replaceGlobal(
    t,
    'Worker',
    class {
      constructor() {
        workers++;
      }
      postMessage() {}
      terminate() {}
    },
  );
  replaceGlobal(t, 'location', { href: 'https://arena.test/app/arena/', origin: 'https://arena.test' });
  const engine = new StockfishEngine();
  t.after(() => engine.dispose());
  const controller = new AbortController();
  controller.abort();
  const pending = engine.prewarm(controller.signal).catch((error) => error);
  await new Promise(setImmediate);
  assert.equal(workers, 0);
  assert.equal((await pending).name, 'AbortError');
});

test('successful Stockfish readiness detaches its old startup signal', async (t) => {
  const workers = [];
  replaceGlobal(
    t,
    'Worker',
    class {
      terminated = false;
      constructor() {
        workers.push(this);
      }
      postMessage(message) {
        const line = message === 'uci' ? 'uciok' : message === 'isready' ? 'readyok' : null;
        if (line) queueMicrotask(() => this.onmessage?.({ data: line }));
      }
      terminate() {
        this.terminated = true;
      }
    },
  );
  replaceGlobal(t, 'location', { href: 'https://arena.test/app/arena/', origin: 'https://arena.test' });
  const engine = new StockfishEngine();
  t.after(() => engine.dispose());
  const controller = new AbortController();
  await engine.prewarm(controller.signal);
  controller.abort();
  await engine.newGame();
  assert.equal(workers.length, 1);
  assert.equal(workers[0].terminated, false, 'a completed startup must not own future worker operations');
});

function replaceGlobal(t, key, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  });
}
