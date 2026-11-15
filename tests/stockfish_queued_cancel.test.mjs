import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';

// One StockfishEngine runs one request at a time. A request cancelled while it
// waits behind another caller's request must settle at once, not when the
// other caller's cold startup (up to the init timeout) and search finish.
// Found by the Quint worker-lifecycle model (stage 2, C-P3). No real worker.

const tick = () => new Promise(setImmediate);

function fakeWorkers(t) {
  const workers = [];
  replaceGlobal(
    t,
    'Worker',
    class {
      terminated = false;
      messages = [];
      constructor() {
        workers.push(this);
      }
      postMessage(message) {
        this.messages.push(message);
      }
      emit(line) {
        this.onmessage?.({ data: line });
      }
      terminate() {
        this.terminated = true;
      }
    },
  );
  replaceGlobal(t, 'location', { href: 'https://arena.test/app/arena/', origin: 'https://arena.test' });
  return workers;
}

/** Answer the UCI handshake and any pending isready on `worker`. */
async function finishStartup(worker) {
  worker.emit('uciok');
  await tick();
  worker.emit('readyok');
  await tick();
}

function track(promise) {
  const state = { outcome: 'pending', value: undefined };
  state.done = promise.then(
    (value) => {
      state.outcome = 'resolved';
      state.value = value;
    },
    (error) => {
      state.outcome = error.name;
    },
  );
  return state;
}

const queuedCalls = {
  prewarm: (engine, signal) => engine.prewarm(signal),
  newGame: (engine, signal) => engine.newGame(signal),
  bestMove: (engine, signal) => engine.bestMove(START_FEN, signal),
  analyze: (engine, signal) => engine.analyze(START_FEN, { signal }),
};

for (const [command, call] of Object.entries(queuedCalls)) {
  test(`a queued ${command} cancelled during another caller's startup settles at once`, async (t) => {
    const workers = fakeWorkers(t);
    const engine = new StockfishEngine({ depth: 1 }, '/stockfish/stockfish-19-lite.js');
    t.after(() => engine.dispose());

    // Caller A owns the gate and is cold-starting the worker; it is not cancelled.
    const owner = track(engine.bestMove(START_FEN));
    await tick();
    assert.equal(workers.length, 1);

    // Caller B queues behind A with its own signal, then cancels.
    const controller = new AbortController();
    const queued = track(call(engine, controller.signal));
    await tick();
    controller.abort();
    await tick();
    assert.equal(queued.outcome, 'AbortError', 'a cancelled queued request must not wait for another caller');

    // Caller A keeps its startup and its result.
    assert.equal(workers[0].terminated, false, "B's cancel does not kill A's startup");
    assert.equal(owner.outcome, 'pending');
    await finishStartup(workers[0]);
    workers[0].emit('bestmove e2e4');
    await owner.done;
    assert.equal(owner.outcome, 'resolved');
    assert.equal(owner.value, 'e2e4');
    assert.equal(workers.length, 1, 'one startup serves A');
    assert.equal(workers[0].messages.filter((m) => m.startsWith('go')).length, 1, 'the cancelled request never reached the worker');
  });
}

test('a cancelled queued request does not let the next request overtake the gate owner', async (t) => {
  const workers = fakeWorkers(t);
  const engine = new StockfishEngine({ depth: 1 }, '/stockfish/stockfish-19-lite.js');
  t.after(() => engine.dispose());

  const owner = track(engine.bestMove(START_FEN));
  await tick();
  const controller = new AbortController();
  const cancelled = track(engine.bestMove(START_FEN, controller.signal));
  const next = track(engine.analyze(START_FEN));
  await tick();
  controller.abort();
  await tick();
  assert.equal(cancelled.outcome, 'AbortError');

  await finishStartup(workers[0]);
  assert.equal(workers[0].messages.filter((m) => m.startsWith('go')).length, 1, 'only the owner is searching; the next request still waits its turn');
  assert.equal(next.outcome, 'pending');

  workers[0].emit('bestmove e2e4');
  await owner.done;
  assert.equal(owner.value, 'e2e4');
  await tick();
  assert.equal(workers[0].messages.filter((m) => m.startsWith('go')).length, 2, 'the next request runs after the owner');
  workers[0].emit('info depth 1 multipv 1 score cp 20 pv e2e4');
  workers[0].emit('bestmove e2e4');
  await next.done;
  assert.equal(next.outcome, 'resolved');
});

function replaceGlobal(t, key, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  });
}
