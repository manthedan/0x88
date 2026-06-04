import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseBestMove, parseStockfishInfo, StockfishEngine, normalizeStockfishFlavor, stockfishFlavorRequiresIsolation, stockfishFlavorUrl, stockfishGoCommand } from '../src/lc0/stockfishEngine.ts';

test('parseBestMove extracts the UCI move and handles (none)', () => {
  assert.equal(parseBestMove('bestmove e2e4 ponder e7e5'), 'e2e4');
  assert.equal(parseBestMove('bestmove a7a8q'), 'a7a8q');
  assert.equal(parseBestMove('bestmove (none)'), null);
  assert.equal(parseBestMove('info depth 12 score cp 31'), null);
});

test('stockfish flavor helpers map browser Stockfish builds', () => {
  assert.equal(normalizeStockfishFlavor(null), 'lite-single');
  assert.equal(normalizeStockfishFlavor('full'), 'single');
  assert.equal(normalizeStockfishFlavor('full-threaded'), 'threaded');
  assert.equal(stockfishFlavorRequiresIsolation('single'), false);
  assert.equal(stockfishFlavorRequiresIsolation('threaded'), true);
  assert.equal(stockfishFlavorUrl('lite-single'), '/stockfish/stockfish-18-lite-single.js');
  assert.equal(stockfishFlavorUrl('single'), '/stockfish/stockfish-18-single.js');
  assert.equal(stockfishFlavorUrl('lite-threaded'), '/stockfish/stockfish-18-lite.js');
  assert.equal(stockfishFlavorUrl('threaded'), '/stockfish/stockfish-18.js');
});

test('stockfishGoCommand prefers movetime over depth and clamps depth', () => {
  assert.equal(stockfishGoCommand({ depth: 6 }), 'go depth 6');
  assert.equal(stockfishGoCommand({}), 'go depth 4');
  assert.equal(stockfishGoCommand({ depth: 0 }), 'go depth 1');
  assert.equal(stockfishGoCommand({ depth: 6, movetimeMs: 200 }), 'go movetime 200');
});

test('parseStockfishInfo extracts score and PV fields', () => {
  assert.deepEqual(parseStockfishInfo('info depth 9 multipv 2 score cp -34 nodes 1200 nps 240000 pv e7e5 g1f3'), {
    multipv: 2,
    depth: 9,
    scoreCp: -34,
    mateIn: undefined,
    nps: 240000,
    pvUci: ['e7e5', 'g1f3'],
  });
  assert.deepEqual(parseStockfishInfo('info depth 12 score mate 3 pv e2e4'), {
    multipv: 1,
    depth: 12,
    scoreCp: undefined,
    mateIn: 3,
    pvUci: ['e2e4'],
  });
  assert.equal(parseStockfishInfo('info depth 12 score cp 31'), null);
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitUntil(predicate, timeoutMs = 200) {
  const started = Date.now();
  while (!predicate()) {
    if (Date.now() - started > timeoutMs) throw new Error('timed out waiting for condition');
    await sleep(1);
  }
}

test('StockfishEngine drains a stopped search before sending the next position', async () => {
  class MockStockfishWorker {
    static instances = [];

    constructor() {
      this.events = [];
      this.onmessage = null;
      this.onerror = null;
      this.timer = null;
      MockStockfishWorker.instances.push(this);
    }

    postMessage(command) {
      this.events.push(`cmd:${command}`);
      if (command === 'uci') queueMicrotask(() => this.emit('uciok'));
      else if (command === 'isready') queueMicrotask(() => this.emit('readyok'));
      else if (String(command).startsWith('go ')) this.scheduleBestMove(50);
      else if (command === 'stop') this.scheduleBestMove(30);
    }

    scheduleBestMove(ms) {
      if (this.timer) clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        this.emit('info depth 1 multipv 1 score cp 7 pv e2e4');
        this.emit('bestmove e2e4');
      }, ms);
    }

    emit(line) {
      this.events.push(`evt:${line}`);
      this.onmessage?.({ data: line });
    }

    terminate() {
      if (this.timer) clearTimeout(this.timer);
      this.events.push('terminate');
    }
  }

  const previousWorker = globalThis.Worker;
  globalThis.Worker = MockStockfishWorker;
  try {
    const fen1 = '8/8/8/8/8/8/4P3/4K3 w - - 0 1';
    const fen2 = '8/8/8/8/4P3/8/8/4K3 b - - 0 1';
    const engine = new StockfishEngine({ depth: 8 }, '/mock-stockfish.js');
    const abortFirst = new AbortController();
    const first = engine.analyze(fen1, { multipv: 2, depth: 8, signal: abortFirst.signal });

    await waitUntil(() => MockStockfishWorker.instances[0]?.events.includes('cmd:go depth 8'));
    const worker = MockStockfishWorker.instances[0];
    abortFirst.abort();
    const second = engine.analyze(fen2, { multipv: 1, depth: 5 });

    await sleep(5);
    assert.equal(worker.events.includes(`cmd:position fen ${fen2}`), false, 'second position was sent before the first search drained');

    await Promise.all([first, second]);
    const firstBest = worker.events.indexOf('evt:bestmove e2e4');
    const secondPosition = worker.events.indexOf(`cmd:position fen ${fen2}`);
    assert.ok(firstBest >= 0, 'first search did not emit a bestmove');
    assert.ok(secondPosition > firstBest, 'second search started before Stockfish acknowledged stop with bestmove');
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
  }
});
