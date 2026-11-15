import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate as turn } from 'node:timers/promises';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';
import { PlentyChessEngine } from '../src/lc0/plentychessEngine.ts';
import { StormphraxEngine } from '../src/lc0/stormphraxEngine.ts';

// The Emscripten counterpart of stockfish_queued_cancel.test.mjs. A request
// cancelled while it waits for its turn must settle with AbortError at once,
// not behind another caller's work. The worker's init RPC has no timeout, so
// before the fix a cancelled queued call could wait forever behind a silent
// init. Found by the round 4 model-hot-spot run (E4).

const engines = [
  ['Berserk', BerserkEngine],
  ['PlentyChess', PlentyChessEngine],
  ['Stormphrax', StormphraxEngine],
];
const FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

async function drain() {
  for (let i = 0; i < 30; i++) await turn();
}

function observe(promise) {
  const state = { status: 'pending' };
  state.done = promise.then(
    (value) => Object.assign(state, { status: 'resolved', value }),
    (error) => Object.assign(state, { status: 'rejected', error }),
  );
  return state;
}

/** Mock Emscripten worker: replies to every RPC unless the init is held. */
function install(t, Engine, { holdInit = false } = {}) {
  const workers = [];
  const descriptors = ['Worker', 'location'].map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  globalThis.location = { href: 'https://example.test/play/', origin: 'https://example.test' };
  globalThis.Worker = class {
    messages = [];
    terminated = false;
    constructor() {
      workers.push(this);
    }
    emit(data) {
      this.onmessage?.({ data });
    }
    line(line) {
      this.emit({ type: 'line', line, stream: 'stdout' });
    }
    releaseInit() {
      const init = this.messages.find((message) => message.type === 'init');
      this.emit({ type: 'ready', id: init.id });
    }
    postMessage(message) {
      this.messages.push(message);
      void (async () => {
        await turn();
        if (this.terminated) return;
        // Hold only the first init; every request starts with its own init RPC.
        if (message.type === 'init' && holdInit && this.messages.filter((m) => m.type === 'init').length === 1) return;
        this.emit({ type: 'ready', id: message.id });
        await turn();
        if (this.terminated || message.type === 'init') return;
        const command = message.command;
        if (command === 'uci') this.line('uciok');
        if (command === 'isready') this.line('readyok');
        if (command.startsWith('go ')) {
          this.line('info depth 4 multipv 1 score cp 10 nodes 40 pv e2e4');
          await turn();
          if (!this.terminated) this.line('bestmove e2e4');
        }
        await turn();
        if (!this.terminated) this.emit({ type: 'commandDone', id: message.id });
      })();
    }
    terminate() {
      this.terminated = true;
    }
  };
  const engine = new Engine({ depth: 4, hashMb: 16, threads: 1 });
  t.after(async () => {
    engine.dispose();
    await drain();
    for (const [name, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  return { engine, workers };
}

function invoke(engine, kind, signal) {
  if (kind === 'prewarm') return engine.prewarm(signal);
  if (kind === 'newGame') return engine.newGame(signal);
  if (kind === 'bestMove') return engine.bestMove(FEN, signal);
  return engine.analyze(FEN, { signal, multipv: 2, depth: 4 });
}

for (const [label, Engine] of engines) {
  for (const kind of ['prewarm', 'newGame', 'bestMove', 'analyze']) {
    test(`${label}: a cancelled queued ${kind} settles without waiting for another caller's init`, async (t) => {
      const { engine, workers } = install(t, Engine, { holdInit: true });
      const active = observe(engine.prewarm());
      await drain();
      assert.equal(workers[0].messages.length, 1, 'the first caller is waiting on init');
      const controller = new AbortController();
      const queued = observe(invoke(engine, kind, controller.signal));
      await drain();
      controller.abort();
      await drain();
      assert.equal(queued.status, 'rejected', 'the cancelled call is stranded behind the other caller');
      assert.equal(queued.error?.name, 'AbortError');
      assert.equal(active.status, 'pending', 'cancelling a queued call must not touch the active one');
      assert.equal(workers[0].terminated, false);
      assert.equal(workers[0].messages.length, 1, 'the cancelled call sent no engine work');
    });
  }

  test(`${label}: requests behind a cancelled one keep their order`, async (t) => {
    const { engine, workers } = install(t, Engine, { holdInit: true });
    const order = [];
    const first = observe(engine.prewarm().then(() => order.push('first')));
    await drain();
    const controller = new AbortController();
    const cancelled = observe(engine.bestMove(FEN, controller.signal));
    const last = observe(engine.bestMove(FEN).then((move) => order.push(`last ${move}`)));
    await drain();
    controller.abort();
    await drain();
    assert.equal(cancelled.status, 'rejected');
    assert.equal(last.status, 'pending', 'the next request still waits for the active one');
    workers[0].releaseInit();
    await Promise.all([first.done, last.done]);
    assert.equal(first.status, 'resolved');
    assert.equal(last.status, 'resolved');
    assert.deepEqual(order, ['first', 'last e2e4']);
  });
}
