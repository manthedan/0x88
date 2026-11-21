import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';

function controlledWorkers(t) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  const workers = [];
  globalThis.Worker = class extends EventTarget {
    messages = [];
    terminated = false;
    constructor() {
      super();
      workers.push(this);
    }
    postMessage(message) {
      this.messages.push(message);
      if (message.type === 'init') queueMicrotask(() => this.reply({ id: message.id, type: 'ready', backend: 'mock-webgpu' }));
    }
    reply(data) {
      this.dispatchEvent(new MessageEvent('message', { data }));
    }
    terminate() {
      this.terminated = true;
    }
  };
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'Worker', previous);
    else delete globalThis.Worker;
  });
  return workers;
}

for (const teardown of ['page', 'pool']) {
  test(`${teardown} teardown during cache reconfiguration cannot resurrect an unowned worker`, async (t) => {
    const workers = controlledWorkers(t);
    const pool = new BigNetSessionPool(
      (config) => new Bt4WorkerSearcher(config),
      undefined,
      undefined,
      () => true,
    );
    const searcher = pool.acquire('t3');
    await searcher.init({ evalCacheEntries: 0 });
    const oldWorker = workers[0];
    const reinitializing = searcher.init({ evalCacheEntries: 2048 });
    assert.ok(oldWorker.messages.some((message) => message.type === 'dispose'));
    assert.equal(workers.length, 1, 'reconfiguration waits before creating its replacement');
    if (teardown === 'pool') pool.disposeAllNow();
    else searcher.dispose();
    oldWorker.reply({ id: -1, type: 'disposed' });
    await assert.rejects(reinitializing, /worker disposed/);
    assert.equal(oldWorker.terminated, true);
    assert.equal(workers.length, 1, 'late cleanup cannot restart the retired searcher');
    const replacement = await pool.acquireAsync('bt4', true);
    await replacement.init();
    assert.equal(workers.length, 2);
    assert.equal(workers[1].terminated, false);
    const closing = replacement.dispose();
    workers[1].reply({ id: -1, type: 'disposed' });
    await closing;
  });
}

for (const failure of ['release', 'transport']) {
  test(`a retained searcher can explicitly retry after ${failure} cleanup failure`, async (t) => {
    const workers = controlledWorkers(t);
    const searcher = new Bt4WorkerSearcher(T3_NET);
    await searcher.init({ evalCacheEntries: 0 });
    const reinitializing = searcher.init({ evalCacheEntries: 2048 });
    if (failure === 'release') workers[0].reply({ id: -1, type: 'error', error: 'injected release failure' });
    else workers[0].dispatchEvent(new Event('error'));
    await assert.rejects(reinitializing, /release failure|disposal transport failed/);
    assert.equal(workers[0].terminated, true);
    assert.equal(workers.length, 1, 'the failed attempt does not hide cleanup errors by retrying');
    assert.equal(await searcher.init({ evalCacheEntries: 2048 }), 'mock-webgpu');
    assert.equal(workers.length, 2, 'a later explicit retry creates a fresh worker');
    assert.equal(searcher.isInitializedFor({ evalCacheEntries: 2048 }), true);
    const closing = searcher.dispose();
    workers[1].reply({ id: -1, type: 'disposed' });
    await closing;
  });
}

test('Bt4WorkerSearcher.search strips extra non-cloneable options at the worker boundary', async (t) => {
  const workers = controlledWorkers(t);
  const searcher = new Bt4WorkerSearcher(T3_NET);
  await searcher.init();
  const worker = workers[0];
  const origPostMessage = worker.postMessage.bind(worker);
  const posted = [];
  worker.postMessage = (message) => {
    const cloned = structuredClone(message);
    posted.push(cloned);
    origPostMessage(message);
    if (cloned.type === 'search') {
      queueMicrotask(() =>
        worker.reply({
          id: cloned.id,
          type: 'result',
          result: { fen: cloned.input.fen, visits: 4, value: 0, children: [], pv: [] },
        }),
      );
    }
  };
  const result = await searcher.search(
    { fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' },
    {
      visits: 4,
      searchPolicy: {
        selectMove() {
          return 'e2e4';
        },
      },
    },
  );
  assert.equal(result.visits, 4);
  const searchMsg = posted.find((message) => message.type === 'search');
  assert.ok(searchMsg);
  assert.equal('searchPolicy' in searchMsg, false);
  assert.equal(searchMsg.visits, 4);
  const closing = searcher.dispose();
  worker.reply({ id: -1, type: 'disposed' });
  await closing;
});
