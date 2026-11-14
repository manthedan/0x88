import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';

// No engine assets or browser: explicitly deliver each UCI event, including
// callbacks already queued when a worker is retired.
const turn = () => new Promise((resolve) => setImmediate(resolve));

function transport(t) {
  const workers = [];
  class Worker {
    messages = [];
    terminated = false;
    autoReady = true;
    constructor() {
      workers.push(this);
    }
    postMessage(command) {
      this.messages.push(command);
      if (command === 'uci') queueMicrotask(() => this.emit('uciok'));
      if (command === 'isready' && this.autoReady) queueMicrotask(() => this.emit('readyok'));
    }
    emit(data) {
      this.onmessage?.({ data });
    }
    crash(message) {
      this.onerror?.({ message, preventDefault() {} });
    }
    terminate() {
      this.terminated = true;
    }
  }
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  globalThis.Worker = Worker;
  t.after(() => {
    if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
    else delete globalThis.Worker;
  });
  const engine = new StockfishEngine({}, '/stockfish/stockfish-19-lite-single.js');
  t.after(() => engine.dispose());
  return { engine, workers };
}

test('cancelled newGame readiness cannot acknowledge the next newGame', async (t) => {
  const { engine, workers } = transport(t);
  await engine.prewarm();
  const old = workers[0];
  old.autoReady = false;
  const controller = new AbortController();
  const cancelled = assert.rejects(engine.newGame(controller.signal), { name: 'AbortError' });
  await turn();
  assert.equal(old.messages.at(-1), 'isready');
  controller.abort();
  await cancelled;

  let completed = false;
  const replacement = engine.newGame().then(() => {
    completed = true;
  });
  await turn();
  // A fix may drain the old handshake or discard the transport. Either way,
  // the first readyok is not the second newGame's acknowledgement.
  if (workers.length === 1) {
    old.emit('readyok');
    await turn();
    assert.equal(completed, false, 'old readyok completed the replacement reset');
    old.emit('readyok');
  } else {
    assert.equal(old.terminated, true);
    await replacement;
  }
  await replacement;
  assert.equal(completed, true);
});

for (const staleEvent of ['info/bestmove', 'error']) {
  test(`retired worker ${staleEvent} cannot settle or terminate replacement analysis`, async (t) => {
    const { engine, workers } = transport(t);
    await engine.prewarm();
    const old = workers[0];
    const failed = assert.rejects(engine.analyze(START_FEN), /first worker failed/);
    await turn();
    old.crash('first worker failed');
    await failed;
    assert.equal(old.terminated, true);

    let outcome = 'pending';
    const result = engine.analyze(START_FEN).then(
      (lines) => {
        outcome = 'resolved';
        return lines;
      },
      (error) => {
        outcome = 'rejected';
        return error;
      },
    );
    await turn();
    const current = workers[1];
    assert.ok(current.messages.some((command) => command.startsWith('go ')));
    if (staleEvent === 'error') old.crash('late old error');
    else {
      old.emit('info depth 99 score cp -999 pv d2d4');
      old.emit('bestmove d2d4');
    }
    await turn();
    assert.equal(outcome, 'pending', 'retired worker settled replacement work');
    assert.equal(current.terminated, false);
    current.emit('info depth 8 score cp 25 pv e2e4');
    current.emit('bestmove e2e4');
    const lines = await result;
    assert.equal(lines.length, 1);
    assert.equal(lines[0].scoreCp, 25);
    assert.deepEqual(lines[0].pvUci, ['e2e4']);
  });
}
