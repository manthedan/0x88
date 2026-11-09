// Research-only negative tests. Explicit run:
// node --experimental-strip-types --test tests/state_workers.repro.mjs
// Deliberately NOT *.test.mjs: Node's default discovery would include *.repro.test.mjs.
import assert from 'node:assert/strict';
import test from 'node:test';
import { BT4_NET, Bt4WorkerSearcher } from '../src/lc0/bt4Engine.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { START_FEN } from '../src/chess/board.ts';

class FakeWorker {
  static instances = [];
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
    if (message.type === 'init') queueMicrotask(() => this.emit('message', {
      data: { type: 'ready', id: message.id, backend: 'fake', inputNames: [], outputNames: [] },
    }));
  }
  terminate() { this.terminated = true; }
}

// No browser, network, models, GPU, or wall-clock timeout is needed.
test('B02: fatal big-net worker error must invalidate loaded state and allow fresh init', async () => {
  const original = globalThis.Worker;
  globalThis.Worker = FakeWorker;
  const searcher = new Bt4WorkerSearcher(BT4_NET);
  try {
    await searcher.init();
    assert.equal(searcher.loaded, true);
    const worker = FakeWorker.instances.at(-1);
    worker.emit('error', { message: 'injected fatal worker crash' });
    // Actual at a297f38: true. init() subsequently returns the cached backend
    // and new requests are posted to the same failed worker.
    assert.equal(searcher.loaded, false, 'crashed worker is still advertised as loaded');
    await searcher.init();
    assert.notEqual(FakeWorker.instances.at(-1), worker);
  } finally {
    searcher.dispose();
    globalThis.Worker = original;
  }
});

test('B03: Maia runtime worker error must reject an outstanding evaluate request', async () => {
  const worker = new FakeWorker();
  // JS can call the TS-private constructor; use the real constructor to isolate
  // the POST-init lifecycle without downloading a model. create() uses this same
  // constructor after its temporary init error listener is removed.
  const evaluator = new Maia3BrowserEvaluator(worker, { cacheStatus: 'disabled', model: '/unused' }, 1500, 1500,
    { inputNames: [], outputNames: [], backend: 'fake' });
  let settled = 'pending';
  const pending = evaluator.evaluate(START_FEN).then(() => { settled = 'resolved'; }, () => { settled = 'rejected'; });
  try {
    worker.emit('error', { message: 'injected fatal worker crash' });
    // Drain the async evaluate continuation and its observer without a timeout race.
    await new Promise(setImmediate);
    assert.equal(settled, 'rejected', 'runtime worker crash leaves evaluate pending');
  } finally {
    evaluator.terminate();
    await pending;
  }
});
