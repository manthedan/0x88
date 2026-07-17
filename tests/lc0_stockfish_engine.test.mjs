import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseBestMove, parseStockfishInfo, StockfishEngine, defaultStockfishUrl, normalizeStockfishFlavor, stockfishFlavorRequiresIsolation, stockfishFlavorUrl, stockfishGoCommand, stockfishWorkerUrl } from '../src/lc0/stockfishEngine.ts';

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
  assert.equal(stockfishFlavorUrl('lite-single').startsWith('/stockfish/stockfish-18-lite-single'), true);
  assert.equal(stockfishFlavorUrl('single'), '/stockfish/stockfish-18-single.js');
  assert.equal(stockfishFlavorUrl('lite-threaded'), '/stockfish/stockfish-18-lite.js');
  assert.equal(stockfishFlavorUrl('threaded'), '/stockfish/stockfish-18.js');
});

test('production asset base hosts the full single-threaded Stockfish pair', async () => {
  const previousBase = globalThis.LC0_BROWSER_ASSET_BASE_URL;
  const previousLocation = globalThis.location;
  globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
  Object.defineProperty(globalThis, 'location', {
    value: { href: 'https://0x88.app/app/analysis/', origin: 'https://0x88.app' },
    configurable: true,
  });
  try {
    const productionModule = await import(`../src/lc0/stockfishEngine.ts?r2-full=${Date.now()}`);
    assert.equal(productionModule.stockfishFlavorUrl('single'), 'https://assets.0x88.app/stockfish/stockfish-18-single.js');
    const worker = productionModule.stockfishWorkerUrl(productionModule.stockfishFlavorUrl('single'));
    assert.match(worker.url, /^blob:/);
    assert.match(worker.url, /stockfish-18-single\.wasm$/);
    URL.revokeObjectURL(worker.objectUrl);
  } finally {
    if (previousBase === undefined) delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
    else globalThis.LC0_BROWSER_ASSET_BASE_URL = previousBase;
    if (previousLocation === undefined) delete globalThis.location;
    else Object.defineProperty(globalThis, 'location', { value: previousLocation, configurable: true });
  }
});

test('Stockfish lite-single uses relaxed SIMD when validated and falls back otherwise', () => {
  const originalValidate = WebAssembly.validate;
  try {
    WebAssembly.validate = () => true;
    assert.equal(defaultStockfishUrl(), '/stockfish/stockfish-18-lite-single-relaxed.js');
    assert.equal(stockfishFlavorUrl('lite-single'), '/stockfish/stockfish-18-lite-single-relaxed.js');
    assert.equal(stockfishFlavorUrl('single'), '/stockfish/stockfish-18-single.js');
    assert.equal(stockfishFlavorUrl('lite-threaded'), '/stockfish/stockfish-18-lite.js');
    assert.equal(stockfishFlavorUrl('threaded'), '/stockfish/stockfish-18.js');

    WebAssembly.validate = () => false;
    assert.equal(defaultStockfishUrl(), '/stockfish/stockfish-18-lite-single.js');
    assert.equal(stockfishFlavorUrl('lite-single'), '/stockfish/stockfish-18-lite-single.js');
    assert.equal(stockfishFlavorUrl('single'), '/stockfish/stockfish-18-single.js');
    assert.equal(stockfishFlavorUrl('lite-threaded'), '/stockfish/stockfish-18-lite.js');
    assert.equal(stockfishFlavorUrl('threaded'), '/stockfish/stockfish-18.js');
  } finally {
    WebAssembly.validate = originalValidate;
  }
});

test('same-origin threaded Stockfish uses its native worker URL', () => {
  const previousLocation = globalThis.location;
  Object.defineProperty(globalThis, 'location', {
    value: { href: 'http://localhost:5181/app/analysis/', origin: 'http://localhost:5181' },
    configurable: true,
  });
  try {
    assert.deepEqual(stockfishWorkerUrl('/stockfish/stockfish-18-lite.js'), {
      url: '/stockfish/stockfish-18-lite.js',
    });
  } finally {
    if (previousLocation === undefined) delete globalThis.location;
    else Object.defineProperty(globalThis, 'location', { value: previousLocation, configurable: true });
  }
});

test('Stockfish pthread bootstrap loads trusted external artifacts through a local helper', () => {
  const previousLocation = globalThis.location;
  const previousBase = globalThis.LC0_BROWSER_ASSET_BASE_URL;
  Object.defineProperty(globalThis, 'location', {
    value: { href: 'https://0x88.app/chess/app/analysis/', origin: 'https://0x88.app' },
    configurable: true,
  });
  globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
  try {
    const worker = stockfishWorkerUrl('https://assets.0x88.app/stockfish/stockfish-18-lite.js');
    assert.equal(worker.objectUrl, undefined);
    assert.equal(worker.url, 'https://assets.0x88.app/stockfish/stockfish-18-lite.js');
    assert.equal(worker.bootstrapWasmUrl, 'https://assets.0x88.app/stockfish/stockfish-18-lite.wasm');
    assert.equal(new StockfishEngine({}, 'https://assets.0x88.app/stockfish/stockfish-18-lite.js').maxThreads(), 32);
  } finally {
    if (previousBase === undefined) delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
    else globalThis.LC0_BROWSER_ASSET_BASE_URL = previousBase;
    if (previousLocation === undefined) delete globalThis.location;
    else Object.defineProperty(globalThis, 'location', { value: previousLocation, configurable: true });
  }
});

test('StockfishEngine reports thread capacity from the resolved worker URL', () => {
  assert.equal(new StockfishEngine({}, '/stockfish/stockfish-18-lite-single.js').maxThreads(), 1);
  assert.equal(new StockfishEngine({}, '/stockfish/stockfish-18-lite.js').maxThreads(), 32);
  assert.equal(new StockfishEngine({}, 'https://assets.0x88.app/stockfish/stockfish-18-lite.js').maxThreads(), 1);
});

test('StockfishEngine starts threaded artifacts through the local pthread bootstrap', async () => {
  class MockStockfishWorker {
    static instances = [];

    constructor(url, options) {
      this.url = String(url);
      this.options = options;
      this.messages = [];
      this.onmessage = null;
      this.onerror = null;
      MockStockfishWorker.instances.push(this);
    }

    postMessage(message) {
      this.messages.push(message);
      if (message === 'uci') queueMicrotask(() => this.onmessage?.({ data: 'uciok' }));
      else if (message === 'isready') queueMicrotask(() => this.onmessage?.({ data: 'readyok' }));
    }

    terminate() {}
  }

  const previousWorker = globalThis.Worker;
  const previousLocation = globalThis.location;
  const previousBase = globalThis.LC0_BROWSER_ASSET_BASE_URL;
  globalThis.Worker = MockStockfishWorker;
  Object.defineProperty(globalThis, 'location', {
    value: { href: 'https://0x88.app/app/analysis/', origin: 'https://0x88.app' },
    configurable: true,
  });
  globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
  try {
    const engine = new StockfishEngine({ threads: 2 }, 'https://assets.0x88.app/stockfish/stockfish-18-lite.js');
    await engine.prewarm();
    const worker = MockStockfishWorker.instances[0];
    assert.match(worker.url, /stockfishPthreadBootstrap\.js\?no-inline#https%3A%2F%2Fassets\.0x88\.app%2Fstockfish%2Fstockfish-18-lite\.wasm$/);
    assert.equal(worker.options.name, 'stockfish-pthread-bootstrap');
    assert.deepEqual(worker.messages.slice(0, 4), [
      'uci',
      'setoption name Threads value 2',
      'isready',
    ]);
    engine.dispose();
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
    if (previousBase === undefined) delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
    else globalThis.LC0_BROWSER_ASSET_BASE_URL = previousBase;
    if (previousLocation === undefined) delete globalThis.location;
    else Object.defineProperty(globalThis, 'location', { value: previousLocation, configurable: true });
  }
});

test('cross-origin Stockfish wrapper hash initializes the UCI worker, not a pthread helper', () => {
  const previousLocation = globalThis.location;
  const previousBase = globalThis.LC0_BROWSER_ASSET_BASE_URL;
  const previousCreateObjectUrl = URL.createObjectURL;
  Object.defineProperty(globalThis, 'location', {
    value: { href: 'https://0x88.app/app/analysis/' },
    configurable: true,
  });
  globalThis.LC0_BROWSER_ASSET_BASE_URL = 'https://assets.0x88.app';
  URL.createObjectURL = () => 'blob:https://0x88.app/stockfish-wrapper';
  try {
    const worker = stockfishWorkerUrl('https://assets.0x88.app/stockfish/stockfish-18-lite-single-relaxed.js');
    assert.equal(worker.objectUrl, 'blob:https://0x88.app/stockfish-wrapper');
    assert.equal(worker.url, 'blob:https://0x88.app/stockfish-wrapper#https%3A%2F%2Fassets.0x88.app%2Fstockfish%2Fstockfish-18-lite-single-relaxed.wasm');
    assert.equal(worker.url.includes(',worker'), false);
  } finally {
    URL.createObjectURL = previousCreateObjectUrl;
    if (previousBase === undefined) delete globalThis.LC0_BROWSER_ASSET_BASE_URL;
    else globalThis.LC0_BROWSER_ASSET_BASE_URL = previousBase;
    if (previousLocation === undefined) delete globalThis.location;
    else Object.defineProperty(globalThis, 'location', { value: previousLocation, configurable: true });
  }
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
    nodes: 1200,
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

test('StockfishEngine disposal rejects an active search instead of leaving it pending', async () => {
  class HangingStockfishWorker {
    constructor() {
      this.onmessage = null;
      this.onerror = null;
    }

    postMessage(command) {
      if (command === 'uci') queueMicrotask(() => this.onmessage?.({ data: 'uciok' }));
      else if (command === 'isready') queueMicrotask(() => this.onmessage?.({ data: 'readyok' }));
      // Deliberately never answer go/stop; dispose must settle the request.
    }

    terminate() {}
  }

  const previousWorker = globalThis.Worker;
  globalThis.Worker = HangingStockfishWorker;
  try {
    const engine = new StockfishEngine({ depth: 8 }, '/mock-stockfish.js');
    const search = engine.bestMove('8/8/8/8/8/8/4P3/4K3 w - - 0 1');
    await sleep(5);
    engine.dispose();
    await assert.rejects(search, (error) => error.name === 'AbortError');
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
  }
});

test('StockfishEngine disposal rejects queued searches without restarting a worker', async () => {
  class HangingStockfishWorker {
    static instances = [];

    constructor() {
      this.onmessage = null;
      this.onerror = null;
      HangingStockfishWorker.instances.push(this);
    }

    postMessage(command) {
      if (command === 'uci') queueMicrotask(() => this.onmessage?.({ data: 'uciok' }));
      else if (command === 'isready') queueMicrotask(() => this.onmessage?.({ data: 'readyok' }));
    }

    terminate() {}
  }

  const previousWorker = globalThis.Worker;
  globalThis.Worker = HangingStockfishWorker;
  try {
    const engine = new StockfishEngine({ depth: 8 }, '/mock-stockfish.js');
    const first = engine.bestMove('8/8/8/8/8/8/4P3/4K3 w - - 0 1');
    const queued = engine.bestMove('8/8/8/8/4P3/8/8/4K3 b - - 0 1');
    await sleep(5);
    engine.dispose();
    await Promise.all([
      assert.rejects(first, (error) => error.name === 'AbortError'),
      assert.rejects(queued, (error) => error.name === 'AbortError'),
    ]);
    assert.equal(HangingStockfishWorker.instances.length, 1, 'queued search recreated a worker after disposal');
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
  }
});

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

test('StockfishEngine rejects an aborted analysis if Stockfish never returns bestmove', async () => {
  class HungStockfishWorker {
    static instances = [];

    constructor() {
      this.events = [];
      this.onmessage = null;
      this.onerror = null;
      HungStockfishWorker.instances.push(this);
    }

    postMessage(command) {
      this.events.push(`cmd:${command}`);
      if (command === 'uci') queueMicrotask(() => this.emit('uciok'));
      else if (command === 'isready') queueMicrotask(() => this.emit('readyok'));
    }

    emit(line) {
      this.events.push(`evt:${line}`);
      this.onmessage?.({ data: line });
    }

    terminate() {
      this.events.push('terminate');
    }
  }

  const previousWorker = globalThis.Worker;
  globalThis.Worker = HungStockfishWorker;
  try {
    const engine = new StockfishEngine({ depth: 8 }, '/mock-stockfish.js');
    const controller = new AbortController();
    const analysis = engine.analyze('8/8/8/8/8/8/4P3/4K3 w - - 0 1', { multipv: 1, depth: 8, signal: controller.signal });

    await waitUntil(() => HungStockfishWorker.instances[0]?.events.includes('cmd:go depth 8'));
    controller.abort();
    await assert.rejects(analysis, (error) => error.name === 'AbortError');
    assert.ok(HungStockfishWorker.instances[0].events.includes('cmd:stop'));
    assert.ok(HungStockfishWorker.instances[0].events.includes('terminate'));
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
  }
});
