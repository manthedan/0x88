import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';
import { PlentyChessEngine } from '../src/lc0/plentychessEngine.ts';
import { RecklessEngine } from '../src/lc0/recklessEngine.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';
import { StormphraxEngine } from '../src/lc0/stormphraxEngine.ts';
import { ViridithasEngine } from '../src/lc0/viridithasEngine.ts';

const flush = async () => {
  for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

/**
 * Answers the UCI handshake for string (Stockfish) and structured (Emscripten)
 * workers, but never finishes a search or a one-shot WASI run, so the first
 * call stays in flight while later calls queue behind it.
 */
class HoldingWorker {
  static instances = [];
  terminated = false;
  constructor() {
    HoldingWorker.instances.push(this);
  }
  postMessage(message) {
    if (message?.type === 'run') return;
    const command = typeof message === 'string' ? message : message.command;
    queueMicrotask(() => {
      if (command?.startsWith('go ')) return;
      const line = command === 'uci' ? 'uciok' : command === 'isready' ? 'readyok' : null;
      if (line) this.onmessage?.({ data: typeof message === 'string' ? line : { type: 'line', stream: 'stdout', line } });
      if (typeof message !== 'string') this.onmessage?.({ data: { type: message.type === 'init' ? 'ready' : 'commandDone', id: message.id } });
    });
  }
  terminate() {
    this.terminated = true;
  }
}

const ENGINES = [
  ['Stockfish', () => new StockfishEngine({ depth: 2 }, '/stockfish/stockfish-19-lite-single.js')],
  ['Berserk', () => new BerserkEngine({ depth: 2 })],
  ['PlentyChess', () => new PlentyChessEngine({ depth: 2 })],
  ['Stormphrax', () => new StormphraxEngine({ depth: 2 })],
  ['Reckless', () => new RecklessEngine({ depth: 2 }, '/reckless/test.wasm', { forceOneShot: true })],
  ['Viridithas', () => new ViridithasEngine({ depth: 2 }, '/viridithas/test.wasm', { forceOneShot: true })],
];

for (const [label, create] of ENGINES) {
  test(`${label} dispose rejects in-flight and queued work without spawning a replacement worker`, async (t) => {
    const previousWorker = globalThis.Worker;
    const previousLocation = globalThis.location;
    globalThis.Worker = HoldingWorker;
    globalThis.location = { href: 'http://localhost:5181/arena/', origin: 'http://localhost:5181' };
    HoldingWorker.instances = [];
    t.after(() => {
      globalThis.Worker = previousWorker;
      if (previousLocation === undefined) delete globalThis.location;
      else globalThis.location = previousLocation;
    });

    const engine = create();
    const inFlight = engine.bestMove(START_FEN);
    const queuedSearch = engine.bestMove(START_FEN);
    const queuedNewGame = engine.newGame();
    const settled = [inFlight, queuedSearch, queuedNewGame].map((promise) =>
      promise.then(
        () => 'resolved',
        (error) => error,
      ),
    );
    await flush();
    const workers = HoldingWorker.instances.length;
    assert.ok(workers >= 1, 'the first search started a worker');

    engine.dispose();
    engine.dispose();
    for (const outcome of await Promise.all(settled)) {
      assert.notEqual(outcome, 'resolved');
      assert.equal(outcome.name, 'AbortError', outcome.message);
    }
    await flush();
    assert.equal(HoldingWorker.instances.length, workers, 'queued work must not recreate (and leak) a worker after dispose');
    assert.ok(
      HoldingWorker.instances.every((worker) => worker.terminated),
      'every worker is terminated',
    );

    await assert.rejects(engine.bestMove(START_FEN), { name: 'AbortError' });
    await assert.rejects(engine.analyze(START_FEN, { depth: 1 }), { name: 'AbortError' });
    await assert.rejects(engine.prewarm(), { name: 'AbortError' });
    assert.equal(HoldingWorker.instances.length, workers, 'calls after dispose never start a worker');
    assert.deepEqual(engine.lastInfo(), []);
    engine.dispose();
  });
}
