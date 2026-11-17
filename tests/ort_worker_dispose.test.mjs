import assert from 'node:assert/strict';
import test from 'node:test';
import { Bt4WorkerSearcher } from '../src/lc0/bt4Engine.ts';
import { disposeOrtWorker } from '../src/lc0/disposeOrtWorker.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { makeSoakSmall } from '../src/lc0/mobileSoakSmall.ts';
import { env } from '../src/nn/ortRuntime.ts';
import { releaseOrtWorkerResources } from '../src/nn/ortWorkerDispose.ts';

class WorkerStub extends EventTarget {
  messages = [];
  terminations = 0;
  postMessage(message) {
    this.messages.push(message);
  }
  terminate() {
    this.terminations++;
  }
  reply(data) {
    this.dispatchEvent(new MessageEvent('message', { data }));
  }
}

test('graceful close accepts only its ack, is idempotent and cleans listeners/timer', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const worker = new WorkerStub();
  const closed = disposeOrtWorker(worker);
  assert.equal(disposeOrtWorker(worker), closed);
  assert.deepEqual(worker.messages, [{ type: 'cancel' }, { type: 'dispose', id: -1 }]);
  worker.reply({ type: 'ready', id: -1 });
  worker.reply({ type: 'disposed', id: 123 });
  assert.equal(worker.terminations, 0);
  worker.reply({ type: 'disposed', id: -1 });
  await closed;
  t.mock.timers.tick(1500);
  worker.reply({ type: 'disposed', id: -1 });
  assert.equal(worker.terminations, 1);
});

for (const failure of ['timeout', 'error-reply', 'transport-error', 'post-throws']) {
  test(`graceful close falls back to termination: ${failure}`, async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const worker = new WorkerStub();
    if (failure === 'post-throws')
      worker.postMessage = () => {
        throw Error('closed');
      };
    const closed = disposeOrtWorker(worker);
    if (failure === 'timeout') {
      t.mock.timers.tick(1499);
      assert.equal(worker.terminations, 0);
      t.mock.timers.tick(1);
    } else if (failure === 'error-reply') worker.reply({ type: 'error', id: -1 });
    else if (failure === 'transport-error') worker.dispatchEvent(new Event('error'));
    if (failure === 'error-reply' || failure === 'transport-error') await assert.rejects(closed, /release failed|transport failed/);
    else await closed;
    assert.equal(worker.terminations, 1);
  });
}

test('termination failure remains visible to an awaitable lab owner', async () => {
  const worker = new WorkerStub();
  worker.terminate = () => {
    throw Error('cannot terminate');
  };
  const closed = disposeOrtWorker(worker);
  worker.reply({ type: 'disposed', id: -1 });
  await assert.rejects(closed, /cannot terminate/);
});

for (const failRelease of [false, true]) {
  test(`dedicated worker releases session/bindings before destroying and dropping its device (${failRelease})`, async (t) => {
    const descriptor = Object.getOwnPropertyDescriptor(env.webgpu, 'device');
    const calls = [];
    const device = { destroy: () => calls.push('destroy') };
    Object.defineProperty(env.webgpu, 'device', { configurable: true, value: device });
    t.after(() => {
      if (descriptor) Object.defineProperty(env.webgpu, 'device', descriptor);
      else delete env.webgpu.device;
    });
    const gate = Promise.withResolvers();
    const release = releaseOrtWorkerResources(async () => {
      calls.push('release');
      await gate.promise;
      if (failRelease) throw Error('release failed');
    });
    await Promise.resolve();
    assert.deepEqual(calls, ['release']);
    gate.resolve();
    if (failRelease) await assert.rejects(release, /release failed/);
    else await release;
    assert.deepEqual(calls, ['release', 'destroy']);
    assert.equal(env.webgpu.device, undefined);
  });
}

function installWorker(t) {
  const workers = [];
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  class ControlledWorker extends WorkerStub {
    constructor() {
      super();
      workers.push(this);
    }
    postMessage(message) {
      super.postMessage(message);
      if (message.type === 'init')
        queueMicrotask(() => this.reply({ type: 'ready', id: message.id, backend: 'webgpu', ortIoMode: 'gpu-capture', ortRuntime: { ioMode: 'gpu-capture' } }));
    }
  }
  Object.defineProperty(globalThis, 'Worker', { configurable: true, value: ControlledWorker });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor);
    else delete globalThis.Worker;
  });
  return workers;
}

test('big-net dispose detaches immediately and a late old ack cannot invalidate its replacement', async (t) => {
  const workers = installWorker(t);
  const engine = new Bt4WorkerSearcher();
  await engine.init();
  const old = workers[0];
  const searching = engine.search('unused', {});
  await Promise.resolve();
  const rejected = assert.rejects(searching, /disposed/);
  const closed = engine.dispose();
  assert.equal(engine.loaded, false);
  assert.equal(old.terminations, 0);
  await rejected;
  await engine.init();
  old.reply({ type: 'disposed', id: -1 });
  await closed;
  assert.equal(engine.loaded, true);
  assert.equal(workers[1].terminations, 0);
  const end = engine.dispose();
  workers[1].reply({ type: 'disposed', id: -1 });
  await end;
});

test('Small uses Analysis device-selected worker IO; abort rejects immediately while release waits for ack', async (t) => {
  const workers = installWorker(t);
  const signal = new AbortController();
  const engine = makeSoakSmall(signal.signal);
  await engine.load();
  const worker = workers[0];
  const init = worker.messages[0];
  assert.equal(init.deviceModelSelection, true);
  assert.equal(init.evalCacheEntries, 2048);
  assert.equal(init.ortIoMode, undefined, 'same default as product Analysis');
  assert.equal(engine.runtime().previousSoakIoMode, 'cpu');
  assert.equal(engine.runtime().ioMode, 'gpu-capture');
  const searching = engine.search('unused');
  signal.abort();
  await assert.rejects(searching, { name: 'AbortError' });
  assert.equal(worker.terminations, 0);
  const closed = engine.dispose();
  worker.reply({ type: 'disposed', id: -1 });
  await closed;
  assert.equal(worker.terminations, 1);
});

test('normal Maia disposal preserves admitted inference results while rejecting new work', async () => {
  const worker = new WorkerStub();
  const engine = new Maia3BrowserEvaluator(worker, { url: 'model' }, 1500, 1500, { inputNames: [], outputNames: [], backend: 'test' });
  const evaluating = engine.evaluate('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const closed = engine.dispose();
  await assert.rejects(engine.evaluate('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'), { name: 'AbortError' });
  worker.reply({ type: 'result', id: 1, logitsMove: new Float32Array(4352).buffer, logitsValue: new Float32Array(3).buffer });
  worker.reply({ type: 'disposed', id: -1 });
  assert.equal((await evaluating).legalPriors.length, 20);
  await closed;
  assert.equal(worker.terminations, 1);
});

test('Maia abort keeps logical cancellation immediate and bounds a missing release ack', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const worker = new WorkerStub();
  const engine = new Maia3BrowserEvaluator(worker, { url: 'model' }, 1500, 1500, { inputNames: [], outputNames: [], backend: 'test' });
  engine.terminate();
  let closed = false;
  const pending = engine.dispose().then(() => {
    closed = true;
  });
  await Promise.resolve();
  assert.equal(closed, false);
  t.mock.timers.tick(1500);
  await pending;
  assert.equal(worker.terminations, 1);
});
