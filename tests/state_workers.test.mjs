import assert from 'node:assert/strict';
import test from 'node:test';
import { BT4_NET, Bt4WorkerSearcher } from '../src/lc0/bt4Engine.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { START_FEN } from '../src/chess/board.ts';

class FakeWorker {
  static instances = [];
  static autoReady = true;
  listeners = new Map();
  messages = [];
  terminated = false;
  constructor() { FakeWorker.instances.push(this); }
  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }
  removeEventListener(type, listener) { this.listeners.get(type)?.delete(listener); }
  emit(type, data) { for (const listener of this.listeners.get(type) ?? []) listener(data); }
  postMessage(message) {
    this.messages.push(message);
    if (message.type === 'init' && FakeWorker.autoReady) queueMicrotask(() => this.ready(message.id));
  }
  ready(id = this.messages.find((message) => message.type === 'init').id) {
    this.emit('message', { data: { type: 'ready', id, backend: 'fake', inputNames: [], outputNames: [] } });
  }
  terminate() { this.terminated = true; }
}

const drain = () => new Promise(setImmediate);
async function withBigNet(run) {
  const original = globalThis.Worker;
  globalThis.Worker = FakeWorker;
  const searcher = new Bt4WorkerSearcher(BT4_NET);
  try { await run(searcher); }
  finally { searcher.dispose(); globalThis.Worker = original; FakeWorker.autoReady = true; }
}
function maia(worker, onFailure) {
  // Use the real post-init constructor without fetching models or requiring GPU.
  return new Maia3BrowserEvaluator(worker, { cacheStatus: 'disabled', model: '/unused' }, 1500, 1500,
    { inputNames: [], outputNames: [], backend: 'fake' }, onFailure);
}

for (const type of ['error', 'messageerror']) {
  test(`B02: big-net ${type} invalidates loaded state, settles all requests and permits retry`, () => withBigNet(async (searcher) => {
    await searcher.init();
    const worker = FakeWorker.instances.at(-1);
    const requests = [searcher.evaluate(START_FEN), searcher.evaluate(START_FEN)];
    const rejected = requests.map((request) => assert.rejects(request, /injected fatal|could not be decoded/));
    await drain();
    worker.emit(type, { message: 'injected fatal worker crash' });
    assert.equal(searcher.loaded, false);
    assert.equal(searcher.backend, '');
    assert.equal(worker.terminated, true);
    await Promise.all(rejected);
    await searcher.init();
    const replacement = FakeWorker.instances.at(-1);
    assert.notEqual(replacement, worker);
    worker.emit('error', { message: 'late crash' });
    worker.ready();
    assert.equal(searcher.loaded, true);
    assert.equal(replacement.terminated, false);
  }));

  test(`B03: Maia runtime ${type} rejects all inference, keeps the failure and terminates once`, async () => {
    const worker = new FakeWorker();
    const failures = [];
    const evaluator = maia(worker, (error) => failures.push(error));
    const requests = [evaluator.evaluate(START_FEN), evaluator.evaluateConditions(START_FEN, [{ selfElo: 1500, oppoElo: 1500 }])];
    const rejected = requests.map((request) => assert.rejects(request, /injected fatal|could not be decoded/));
    worker.emit(type, { message: 'injected fatal worker crash' });
    await Promise.all(rejected);
    assert.equal(worker.terminated, true);
    assert.equal(failures.length, 1);
    assert.notEqual(evaluator.failure.name, 'AbortError');
    worker.emit('message', { data: { type: 'result', id: 1 } });
    worker.emit('error', { message: 'duplicate crash' });
    evaluator.terminate();
    await assert.rejects(evaluator.evaluate(START_FEN), (error) => error === failures[0]);
    await evaluator.dispose();
    assert.equal(failures.length, 1);
    assert.equal([...worker.listeners.values()].every((listeners) => !listeners.size), true);
    const replacement = maia(new FakeWorker());
    assert.equal(replacement.failure, null);
    replacement.terminate();
  });
}

test('B02: crashed initialization cannot clear a replacement init or accept an old ready event', () => withBigNet(async (searcher) => {
  FakeWorker.autoReady = false;
  const first = searcher.init();
  const rejected = assert.rejects(first, /init crash/);
  const worker = FakeWorker.instances.at(-1);
  worker.emit('error', { message: 'init crash' });
  const second = searcher.init();
  const replacement = FakeWorker.instances.at(-1);
  await rejected;
  worker.ready();
  assert.equal(searcher.loaded, false);
  assert.equal(replacement.terminated, false);
  replacement.ready();
  await second;
  await searcher.init();
  assert.equal(FakeWorker.instances.at(-1), replacement);
}));

test('B02: cancellation keeps a healthy worker; cache-size changes recreate it', () => withBigNet(async (searcher) => {
  await searcher.init();
  const worker = FakeWorker.instances.at(-1);
  const controller = new AbortController();
  const request = searcher.search(START_FEN, { visits: 1, signal: controller.signal });
  await drain();
  controller.abort();
  const search = worker.messages.find((message) => message.type === 'search');
  assert.deepEqual(worker.messages.at(-1), { type: 'cancel', target: search.id });
  worker.emit('message', { data: { type: 'result', id: search.id, result: { cancelled: true } } });
  await request;
  assert.equal(searcher.loaded, true);
  await searcher.init({ evalCacheEntries: 32 });
  assert.equal(worker.terminated, true);
  assert.notEqual(FakeWorker.instances.at(-1), worker);
}));

test('B03: a crash during graceful dispose settles it instead of waiting forever', async () => {
  const worker = new FakeWorker();
  const evaluator = maia(worker);
  const disposed = assert.rejects(evaluator.dispose(), /dispose crash/);
  worker.emit('error', { message: 'dispose crash' });
  await disposed;
  await evaluator.dispose();
});

test('B03: protocol request errors remain nonfatal', async () => {
  const worker = new FakeWorker();
  const evaluator = maia(worker);
  const rejected = assert.rejects(evaluator.evaluate(START_FEN), /bad request/);
  worker.emit('message', { data: { type: 'error', id: 1, message: 'bad request' } });
  await rejected;
  assert.equal(evaluator.failure, null);
  assert.equal(worker.terminated, false);
  const next = assert.rejects(evaluator.evaluate(START_FEN), { name: 'AbortError' });
  evaluator.terminate();
  await next;
});
