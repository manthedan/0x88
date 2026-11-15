import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { RecklessEngine } from '../src/lc0/recklessEngine.ts';
import { ViridithasEngine } from '../src/lc0/viridithasEngine.ts';

// The WASI counterpart of stockfish_queued_cancel and emscripten_queued_cancel:
// a request cancelled while it waits for its turn settles with AbortError at
// once, not after another caller's search finishes.

const flush = async () => {
  for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

function observe(promise) {
  const state = { status: 'pending' };
  state.done = promise.then(
    (value) => Object.assign(state, { status: 'resolved', value }),
    (error) => Object.assign(state, { status: 'rejected', error }),
  );
  return state;
}

/** One-shot WASI worker that never finishes a run, so the first search stays in flight. */
class HoldingWorker {
  static instances = [];
  messages = [];
  terminated = false;
  constructor() {
    HoldingWorker.instances.push(this);
  }
  postMessage(message) {
    this.messages.push(message);
  }
  terminate() {
    this.terminated = true;
  }
}

const ENGINES = [
  ['Reckless', () => new RecklessEngine({ depth: 2 }, '/reckless/test.wasm', { forceOneShot: true })],
  ['Viridithas', () => new ViridithasEngine({ depth: 2 }, '/viridithas/test.wasm', { forceOneShot: true })],
];

function invoke(engine, kind, signal) {
  if (kind === 'prewarm') return engine.prewarm(signal);
  if (kind === 'newGame') return engine.newGame(signal);
  if (kind === 'bestMove') return engine.bestMove(START_FEN, signal);
  return engine.analyze(START_FEN, { signal, multipv: 2, depth: 2 });
}

for (const [label, create] of ENGINES) {
  for (const kind of ['prewarm', 'newGame', 'bestMove', 'analyze']) {
    test(`${label}: a cancelled queued ${kind} settles without waiting for another caller's search`, async (t) => {
      const previousWorker = globalThis.Worker;
      const previousLocation = globalThis.location;
      globalThis.Worker = HoldingWorker;
      globalThis.location = { href: 'http://localhost:5181/arena/', origin: 'http://localhost:5181' };
      HoldingWorker.instances = [];
      const engine = create();
      t.after(async () => {
        engine.dispose();
        await flush();
        globalThis.Worker = previousWorker;
        if (previousLocation === undefined) delete globalThis.location;
        else globalThis.location = previousLocation;
      });

      const active = observe(engine.bestMove(START_FEN));
      await flush();
      const runs = () => HoldingWorker.instances.flatMap((worker) => worker.messages).filter((message) => message?.type === 'run').length;
      assert.equal(runs(), 1, 'the first caller is waiting on its search');

      const controller = new AbortController();
      const queued = observe(invoke(engine, kind, controller.signal));
      await flush();
      controller.abort();
      await flush();

      assert.equal(queued.status, 'rejected', 'the cancelled call is stranded behind the other caller');
      assert.equal(queued.error?.name, 'AbortError');
      assert.equal(active.status, 'pending', 'cancelling a queued call must not touch the active one');
      assert.ok(
        HoldingWorker.instances.every((worker) => !worker.terminated),
        'cancelling a queued call must not terminate the worker',
      );
      assert.equal(runs(), 1, 'the cancelled call sent no engine work');
    });
  }
}
