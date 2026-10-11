import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ViridithasEngine } from '../src/lc0/viridithasEngine.ts';

const FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const flush = () => new Promise((resolve) => setImmediate(resolve));
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
  line(line) {
    this.onmessage?.({ data: { type: 'persistent-line', stream: 'stdout', line } });
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
  get stop() {
    return new Int32Array(this.messages[0].stopBuffer);
  }
}
function setup(t, isolated = true, options = {}) {
  const previousWorker = globalThis.Worker;
  const previousIsolation = globalThis.crossOriginIsolated;
  MockWorker.instances = [];
  globalThis.Worker = MockWorker;
  globalThis.crossOriginIsolated = isolated;
  const engine = new ViridithasEngine({ depth: 12 }, '/viridithas/test.wasm', options);
  t.after(() => {
    engine.dispose();
    globalThis.Worker = previousWorker;
    globalThis.crossOriginIsolated = previousIsolation;
  });
  return engine;
}

test('Viridithas abort drains bestmove before replacement, retaining the worker and rejecting stale info', async (t) => {
  const engine = setup(t);
  const controller = new AbortController();
  const old = engine.analyze(FEN, { signal: controller.signal });
  const rejected = assert.rejects(old, { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  assert.match(worker.commands(), /go depth 12/);
  worker.line('uciok');
  controller.abort();
  assert.equal(Atomics.load(worker.stop, 0), 1);
  assert.equal(worker.commands(), 'stop\n');
  assert.equal(worker.terminated, false);
  const next = engine.analyze(FEN, { moves: ['e2e4'] });
  await flush();
  assert.equal(worker.commands(), '', 'replacement cannot overtake the old search');
  worker.line('info depth 11 score cp 999 nodes 999 pv a2a3');
  worker.line('bestmove a2a3');
  await rejected;
  await flush();
  assert.equal(MockWorker.instances.length, 1);
  assert.equal(Atomics.load(worker.stop, 0), 0);
  assert.match(worker.commands(), /position fen .* moves e2e4/);
  worker.line('info depth 12 score cp 20 nodes 123 pv e7e5');
  worker.line('bestmove e7e5');
  const infos = await next;
  assert.equal(infos[0].scoreCp, 20);
  assert.equal(infos[0].nodes, 123);
  assert.deepEqual(infos[0].pvUci, ['e7e5']);
  assert.equal(worker.terminated, false);
});

test('Viridithas watchdog replaces an unresponsive worker and ignores its late lines/errors', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const engine = setup(t);
  const controller = new AbortController();
  const old = engine.analyze(FEN, { signal: controller.signal });
  const rejected = assert.rejects(old, { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  controller.abort();
  t.mock.timers.tick(99);
  assert.equal(worker.terminated, false);
  t.mock.timers.tick(1);
  await rejected;
  assert.equal(worker.terminated, true);
  const next = engine.analyze(FEN);
  await flush();
  const replacement = MockWorker.instances[1];
  worker.line('bestmove a2a3');
  worker.onerror({ message: 'stale failure' });
  assert.equal(replacement.terminated, false);
  replacement.line('info depth 12 score cp 30 nodes 456 pv d2d4');
  replacement.line('bestmove d2d4');
  assert.equal((await next)[0].nodes, 456);
});

test('Viridithas dispose during cooperative drain rejects and clears the watchdog', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const engine = setup(t);
  const controller = new AbortController();
  const search = engine.analyze(FEN, { signal: controller.signal });
  const rejected = assert.rejects(search, { name: 'AbortError' });
  await flush();
  controller.abort();
  engine.dispose();
  await rejected;
  t.mock.timers.tick(100);
  assert.equal(engine.runtimeStatus().mode, 'idle');
});

test('Viridithas sync-command abort drains readyok instead of waiting for nonexistent bestmove', async (t) => {
  const engine = setup(t);
  const controller = new AbortController();
  const prewarm = engine.prewarm(controller.signal);
  const rejected = assert.rejects(prewarm, { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  controller.abort();
  worker.line('uciok');
  worker.line('readyok');
  await rejected;
  assert.equal(worker.terminated, false);
});

for (const [label, isolated, runtime] of [
  ['non-isolated', false, {}],
  ['forced one-shot', true, { forceOneShot: true }],
]) {
  test(`Viridithas ${label} retains immediate termination on abort`, async (t) => {
    const engine = setup(t, isolated, runtime);
    const controller = new AbortController();
    const search = engine.analyze(FEN, { signal: controller.signal });
    const rejected = assert.rejects(search, { name: 'AbortError' });
    await flush();
    const worker = MockWorker.instances[0];
    assert.equal(worker.messages[0].type, 'run');
    controller.abort();
    await rejected;
    assert.equal(worker.terminated, true);
  });
}

test('Viridithas a cancelled queued request does not overwrite the active drain or stop flag', async (t) => {
  const engine = setup(t);
  const activeController = new AbortController();
  const active = engine.analyze(FEN, { signal: activeController.signal });
  const rejectedActive = assert.rejects(active, { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  worker.commands();
  activeController.abort();
  const queuedController = new AbortController();
  const queued = engine.analyze(FEN, { signal: queuedController.signal });
  const rejectedQueued = assert.rejects(queued, { name: 'AbortError' });
  queuedController.abort();
  await flush();
  assert.equal(Atomics.load(worker.stop, 0), 1);
  assert.equal(worker.commands(), 'stop\n');
  worker.line('bestmove e2e4');
  await Promise.all([rejectedActive, rejectedQueued]);
  assert.equal(MockWorker.instances.length, 1);
  assert.equal(worker.terminated, false);
  assert.equal(worker.commands(), '');
});

test('Viridithas one-shot ignores a disposed worker error during its replacement request', async (t) => {
  const engine = setup(t, false);
  const controller = new AbortController();
  const old = engine.analyze(FEN, { signal: controller.signal });
  const rejected = assert.rejects(old, { name: 'AbortError' });
  await flush();
  const worker = MockWorker.instances[0];
  controller.abort();
  await rejected;
  const next = engine.analyze(FEN);
  await flush();
  const replacement = MockWorker.instances[1];
  worker.onerror({ message: 'stale one-shot error' });
  assert.equal(replacement.terminated, false);
  replacement.onmessage({
    data: { type: 'result', id: replacement.messages[0].id, exitCode: 0, stdout: ['info depth 12 score cp 10 nodes 123 pv e2e4', 'bestmove e2e4'] },
  });
  assert.equal((await next)[0].nodes, 123);
});

test('Viridithas an already-aborted request never starts a worker', async (t) => {
  const engine = setup(t);
  await assert.rejects(engine.analyze(FEN, { signal: AbortSignal.abort() }), { name: 'AbortError' });
  assert.equal(MockWorker.instances.length, 0);
});
