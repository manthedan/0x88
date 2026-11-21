import assert from 'node:assert/strict';
import { test } from 'node:test';
import { RecklessEngine } from '../src/lc0/recklessEngine.ts';

const FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const flush = async () => {
  for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

class MockWorker {
  static instances = [];
  messages = [];
  terminated = false;
  constructor() {
    MockWorker.instances.push(this);
  }
  postMessage(message) {
    this.messages.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  emit(data) {
    this.onmessage?.({ data });
  }
  line(line) {
    this.emit({ type: 'persistent-line', stream: 'stdout', line });
  }
  commands() {
    const buffer = this.messages[0].inputBuffer;
    const control = new Int32Array(buffer, 0, 4);
    const data = new Uint8Array(buffer, 16);
    const end = Atomics.load(control, 1);
    const start = Atomics.load(control, 0);
    const bytes = Uint8Array.from({ length: end - start }, (_, i) => data[(start + i) % data.length]);
    Atomics.store(control, 0, end);
    return new TextDecoder().decode(bytes);
  }
}

function setup(t, isolated, runtimeOptions = {}) {
  const previousWorker = globalThis.Worker;
  const previousIsolation = globalThis.crossOriginIsolated;
  MockWorker.instances = [];
  globalThis.Worker = MockWorker;
  globalThis.crossOriginIsolated = isolated;
  const engine = new RecklessEngine({ depth: 3 }, '/reckless/test.wasm', runtimeOptions);
  t.after(() => {
    engine.dispose();
    globalThis.Worker = previousWorker;
    globalThis.crossOriginIsolated = previousIsolation;
  });
  return engine;
}

test('Reckless ignores late messages and errors from a replaced persistent worker', async (t) => {
  const engine = setup(t, true, { disablePersistentFallback: true });
  const first = engine.bestMove(FEN);
  await flush();
  const old = MockWorker.instances[0];
  old.emit({ type: 'persistent-error', error: 'wasm trap' });
  await assert.rejects(first, /wasm trap/);
  assert.equal(old.terminated, true);

  const next = engine.bestMove(FEN);
  await flush();
  const replacement = MockWorker.instances[1];
  assert.ok(replacement, 'the next search starts a replacement worker');
  assert.match(replacement.commands(), /go depth 3/);
  // Output the terminated worker had already queued must not leak into, or
  // tear down, the replacement's search.
  old.line('bestmove a2a3');
  old.emit({ type: 'persistent-exit', exitCode: 1 });
  old.onerror?.({ message: 'late worker error' });
  await flush();
  assert.equal(replacement.terminated, false);
  replacement.line('info depth 3 score cp 20 nodes 100 pv e2e4');
  replacement.line('bestmove e2e4');
  assert.equal(await next, 'e2e4');
});

test('Reckless signals a running WASI search and drains it before stepping to the next position', async (t) => {
  const engine = setup(t, true, { disablePersistentFallback: true });
  const controller = new AbortController();
  const cancelled = assert.rejects(engine.analyze(FEN, { signal: controller.signal, depth: 40 }), { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  assert.match(worker.commands(), /go depth 40/);
  const stop = new Int32Array(worker.messages[0].stopBuffer);
  assert.equal(stop.length, 1);
  controller.abort();
  assert.equal(Atomics.load(stop, 0), 1);
  const next = engine.bestMove(FEN, undefined, { moves: ['e2e4'] });
  await flush();
  assert.equal(worker.commands(), '');
  worker.line('bestmove d2d4');
  await cancelled;
  await flush();
  assert.equal(worker.terminated, false);
  assert.equal(Atomics.load(stop, 0), 0);
  assert.equal(worker.commands(), `position fen ${FEN} moves e2e4\ngo depth 3\n`);
  worker.line('bestmove e7e5');
  assert.equal(await next, 'e7e5');
});

test('Reckless retires a persistent worker that never acknowledges the stop flag', async (t) => {
  const engine = setup(t, true, { disablePersistentFallback: true });
  const controller = new AbortController();
  const cancelled = assert.rejects(engine.bestMove(FEN, controller.signal), { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  controller.abort();
  assert.equal(worker.terminated, false);
  await cancelled;
  assert.equal(worker.terminated, true);
});

test('Reckless ignores late one-shot replies from a replaced worker', async (t) => {
  const engine = setup(t, false);
  const controller = new AbortController();
  const first = engine.bestMove(FEN, controller.signal);
  await flush();
  const old = MockWorker.instances[0];
  controller.abort();
  await assert.rejects(first, { name: 'AbortError' });
  const next = engine.bestMove(FEN);
  await flush();
  const replacement = MockWorker.instances[1];
  const run = replacement.messages.at(-1);
  // A stale reply carrying the replacement's request id and a late error.
  old.emit({ type: 'result', id: run.id, stdout: ['bestmove a2a3'], stderr: [], exitCode: 0 });
  old.onerror?.({ message: 'late worker error' });
  await flush();
  assert.equal(replacement.terminated, false);
  replacement.emit({ type: 'result', id: run.id, stdout: ['bestmove e2e4'], stderr: [], exitCode: 0 });
  assert.equal(await next, 'e2e4');
});

test('Reckless one-shot newGame does not spawn a WASI process', async (t) => {
  const engine = setup(t, false);
  await engine.newGame();
  assert.equal(MockWorker.instances.length, 0, 'every one-shot search already starts from a fresh process');
  assert.deepEqual(engine.lastInfo(), []);
});

test('Reckless keeps the browser-API load status across an aborted search', async (t) => {
  const engine = setup(t, false, { backend: 'browser-api' });
  const controller = new AbortController();
  const search = engine.bestMove(FEN, controller.signal);
  await flush();
  MockWorker.instances[0].emit({ type: 'status', phase: 'ready', url: '/reckless/test.wasm', loadedBytes: 10, totalBytes: 10, elapsedMs: 5 });
  assert.equal(engine.runtimeStatus().browserApiLoad?.phase, 'ready');
  controller.abort();
  await assert.rejects(search, { name: 'AbortError' });
  assert.equal(MockWorker.instances[0].terminated, true);
  assert.equal(engine.runtimeStatus().browserApiLoad?.phase, 'ready', 'disposing the browser-API worker keeps its last load status');
});
