import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { START_FEN } from '../src/chess/board.ts';
import { BT4_NET, Bt4WorkerSearcher } from '../src/lc0/bt4Engine.ts';
import { createCriticalEngines } from '../src/lc0/criticalMomentsEngines.ts';
import { MAIA3_POLICY_SIZE, Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { createPlayEngines } from '../src/lc0/play/playEngines.ts';
import { WorkerDeviceLostError } from '../src/lc0/workerDeviceRecovery.ts';
import { createOrtDeviceLossMonitor, watchOrtDeviceLoss } from '../src/nn/ortDeviceLoss.ts';
import { env } from '../src/nn/ortRuntime.ts';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8');

// Test actual handler bodies, without mounting whole pages or importing GPU/lab assets.
function messageHandler(path, functionName, bindings) {
  const source = ts.createSourceFile(path, read(path), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  let owner;
  const find = (node) => {
    if (ts.isFunctionDeclaration(node) && node.name?.text === functionName) owner = node;
    ts.forEachChild(node, find);
  };
  find(source);
  assert.ok(owner, `missing function ${functionName}`);
  let callback;
  const visit = (node) => {
    if (
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      node.expression.name.text === 'addEventListener' &&
      node.arguments[0]?.text === 'message'
    )
      callback ??= node.arguments[1];
    ts.forEachChild(node, visit);
  };
  visit(owner);
  assert.ok(callback, `missing message handler in ${functionName}`);
  const code = ts.transpile(`return (${callback.getText(source)});`, { target: ts.ScriptTarget.ESNext });
  return new Function(...Object.keys(bindings), code)(...Object.values(bindings));
}

for (const [path, name, mapName, extras] of [
  ['../src/lc0/policyOnlyBrowser.ts', 'ensureSearchWorkerInstance', 'workerPending', {}],
  ['../src/lc0/policyOnlyBrowser.ts', 'postLabWorkerRequest', 'labWorkerPending', {}],
  ['../src/lc0/batchingStrengthBrowser.ts', 'createWorker', 'pending', {}],
]) {
  test(`${name} ignores unsolicited device loss, even with a forged active request ID`, () => {
    const pending = new Map();
    let resolved;
    pending.set(7, {
      resolve: (value) => {
        resolved = value;
      },
      reject: assert.fail,
      progress: assert.fail,
    });
    const handle = messageHandler(path, name, { ...extras, [mapName]: pending });
    for (const notification of [
      { type: 'deviceLost', loss: {} },
      { type: 'deviceLost', id: 7, loss: {} },
    ]) {
      handle({ data: notification });
      assert.equal(resolved, undefined);
      assert.equal(pending.size, 1);
    }
    handle({ data: { type: 'ready', id: 7, backend: 'expected', modelCache: 'test' } });
    assert.equal(resolved.backend, 'expected');
    assert.equal(pending.size, 0);
  });
}

class FakeWorker extends EventTarget {
  posted = [];
  postMessage(message) {
    this.posted.push(message);
  }
  terminated = false;
  terminate() {
    this.terminated = true;
  }
  emit(data) {
    this.dispatchEvent(new MessageEvent('message', { data }));
  }
}

function fakeWorkers(t) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  const workers = [];
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: class extends FakeWorker {
      constructor() {
        super();
        workers.push(this);
      }
      postMessage(message) {
        super.postMessage(message);
        if (message.type === 'dispose') queueMicrotask(() => this.emit({ type: 'disposed', id: message.id }));
      }
    },
  });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor);
    else delete globalThis.Worker;
  });
  return workers;
}

const loss = { type: 'deviceLost', loss: { reason: 'unknown', message: 'reset' } };
const ready = (worker, backend) => worker.emit({ type: 'ready', id: worker.posted[0].id, backend });
const result = { move: 'e2e4', visits: 10, value: 0, children: [], pv: ['e2e4'] };

for (const cancel of [false, true]) {
  test(`big-net search invalidates and terminates its worker; ${cancel ? 'Stop prevents retry' : 'second loss retries on WASM'}`, async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const workers = fakeWorkers(t);
    const searcher = new Bt4WorkerSearcher(BT4_NET, () => undefined);
    const warnings = [];
    searcher.onDeviceLossWarning = (warning) => warnings.push(warning);
    const init = searcher.init({ noDownload: true });
    ready(workers[0], 'auto->webgpu');
    await init;
    const controller = new AbortController();
    const search = searcher.search(START_FEN, { signal: controller.signal, visits: (backend) => (backend === 'wasm' ? 10 : 100) });
    await tick();
    const firstId = workers[0].posted.at(-1).id;
    workers[0].emit({ ...loss, id: firstId });
    workers[0].emit({ type: 'error', id: firstId, error: 'GPUBuffer.mapAsync failed' });
    assert.equal(searcher.loaded, false);
    assert.equal(workers[0].terminated, true);
    assert.match(warnings[0], /GPU context was reset/);
    if (cancel) {
      controller.abort();
      assert.equal((await search).cancelled, true);
      assert.equal(workers.length, 1);
      return;
    }
    await tick();
    assert.equal(workers[1].posted[0].ep, 'auto');
    assert.equal(workers[1].posted[0].noModelDownload, true);
    ready(workers[1], 'auto->webgpu');
    await tick();
    workers[0].emit(loss);
    assert.equal(warnings.length, 1);
    const readback = { type: 'error', id: workers[1].posted.at(-1).id, error: 'GPUBuffer.mapAsync failed' };
    let failedBeforeLoss = false;
    void search.catch(() => {
      failedBeforeLoss = true;
    });
    workers[1].emit(readback);
    await tick();
    assert.equal(failedBeforeLoss, false, 'the readback failure leaves time for the subsequent device loss');
    workers[1].emit(loss);
    await tick();
    assert.equal(workers[2].posted[0].ep, 'wasm');
    assert.equal(workers[2].posted[0].noModelDownload, true);
    ready(workers[2], 'wasm');
    await tick();
    const request = workers[2].posted.at(-1);
    assert.equal(request.visits, 10);
    workers[2].emit({ id: request.id, type: 'searchResult', result });
    assert.equal((await search).move, 'e2e4');
    assert.match(warnings[1], /Switching to CPU/);
    assert.equal(searcher.deviceLost.message, 'reset');
    await searcher.dispose();
  });
}

for (const stage of ['prefetch', 'session init']) {
  test(`Arena Stop settles a warm search while its replacement is stalled in ${stage}`, async (t) => {
    const workers = fakeWorkers(t);
    let prefetchSignal;
    let loads = 0;
    const searcher = new Bt4WorkerSearcher(BT4_NET, (_config, { signal }) => {
      if (++loads !== 2 || stage !== 'prefetch') return;
      prefetchSignal = signal;
      return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }));
    });
    t.after(() => searcher.dispose());
    const init = searcher.init();
    ready(workers[0], 'auto->webgpu');
    await init;
    const controller = new AbortController();
    const search = searcher.search(START_FEN, { signal: controller.signal, visits: 10 });
    await tick();
    assert.equal(workers[0].posted.at(-1).type, 'search');
    workers[0].emit(loss);
    await tick();
    if (stage === 'prefetch') assert.equal(prefetchSignal.aborted, false);
    else assert.equal(workers[1].posted[0].type, 'init');
    controller.abort();
    assert.equal(await Promise.race([search.then((result) => result.cancelled), tick().then(() => 'stuck')]), true);
    if (stage === 'prefetch') {
      assert.equal(prefetchSignal.aborted, true);
      assert.equal(workers.length, 1);
    } else {
      assert.equal(workers[1].terminated, true);
    }
  });
}

test('a big-net inference error without device loss is reported without starting another worker', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const workers = fakeWorkers(t);
  const searcher = new Bt4WorkerSearcher(BT4_NET, () => undefined);
  const search = searcher.search(START_FEN, { visits: 10 });
  const failed = assert.rejects(search, /GPUBuffer.mapAsync failed/);
  ready(workers[0], 'auto->webgpu');
  await tick();
  workers[0].emit({ type: 'error', id: workers[0].posted.at(-1).id, error: 'GPUBuffer.mapAsync failed' });
  t.mock.timers.tick(1000);
  await failed;
  assert.equal(searcher.loaded, true);
  assert.equal(workers.length, 1);
  await searcher.dispose();
});

test('Play’s LQO policy-book evaluation is retried on a fresh worker', async (t) => {
  const workers = fakeWorkers(t);
  const searcher = new Bt4WorkerSearcher(BT4_NET, () => undefined);
  const evaluation = searcher.evaluate(START_FEN, { signal: new AbortController().signal });
  ready(workers[0], 'auto->webgpu');
  await tick();
  workers[0].emit(loss);
  await tick();
  assert.equal(workers[0].terminated, true);
  ready(workers[1], 'auto->webgpu');
  await tick();
  const retry = workers[1].posted.at(-1);
  assert.equal(retry.type, 'evaluate');
  assert.equal(retry.input, START_FEN);
  const policy = { legalPriors: [{ uci: 'e2e4', prior: 1 }] };
  workers[1].emit({ id: retry.id, type: 'evaluation', result: policy });
  assert.deepEqual(await evaluation, policy);
  await searcher.dispose();
});

test('Play retries the same Maia3 move after a lost worker and pins WASM after the second loss', async (t) => {
  const workers = fakeWorkers(t);
  const evaluators = [];
  const engines = createPlayEngines({
    loadCentipawn: assert.fail,
    createMaia3: async (options) => {
      const evaluator = await Maia3BrowserEvaluator.create(options, { load: async () => ({ model: 'fake' }), clear: assert.fail });
      evaluators.push(evaluator);
      return evaluator;
    },
  });
  const warnings = [];
  const ui = { current: () => () => true, mounted: () => true, note() {}, progress() {}, hideProgress() {}, deviceWarning: (text) => warnings.push(text) };
  const move = engines.maia3Move(1500, ui, ['fake'], new AbortController().signal, START_FEN, { style: 'argmax' });
  await tick();
  ready(workers[0], 'auto->webgpu');
  await tick();
  workers[0].emit(loss);
  assert.ok(evaluators[0].failure instanceof WorkerDeviceLostError);
  assert.equal(workers[0].terminated, true);
  await tick();
  assert.equal(workers[1].posted[0].ep, undefined);
  // The replacement can also lose a device before its ready reply.
  workers[1].emit({ ...loss, id: 0 });
  await tick();
  assert.equal(workers[1].terminated, true);
  assert.equal(workers[2].posted[0].ep, 'wasm');
  ready(workers[2], 'wasm');
  await tick();
  const logitsMove = new Float32Array(MAIA3_POLICY_SIZE).buffer;
  const logitsValue = new Float32Array(3).buffer;
  workers[2].emit({ type: 'result', id: workers[2].posted.at(-1).id, logitsMove, logitsValue });
  assert.match((await move).move, /^[a-h][1-8][a-h][1-8]$/);
  assert.match(warnings[1], /Switching to CPU/);
  await engines.retireUnusedEngines();
});

test('Critical Moments replaces a lost Lc0 worker during a running comparison', async (t) => {
  const workers = fakeWorkers(t);
  const warnings = [];
  const broker = { register() {}, unregister() {}, acquire: async () => ({ release() {} }) };
  const report = createCriticalEngines(broker, undefined, new AbortController().signal, (text) => warnings.push(text));
  t.after(report.dispose);
  const lane = report.lanes.find((lane) => lane.id === 'lc0');
  const evaluation = lane.evaluate({ fen: START_FEN, history: [START_FEN] }, undefined, () => {});
  await tick();
  ready(workers[0], 'auto->webgpu');
  await tick();
  workers[0].emit(loss);
  await tick();
  assert.equal(workers[0].terminated, true);
  assert.match(warnings[0], /GPU context was reset/);
  ready(workers[1], 'auto->webgpu');
  await tick();
  workers[1].emit({ id: workers[1].posted.at(-1).id, type: 'searchResult', result: { ...result, children: [{ move: 'e2e4', visits: 10, value: 0 }] } });
  assert.equal((await evaluation).provenance.runtime, 'auto->webgpu');
});

test('device monitors reuse one promise subscription, remove retired listeners, and switch devices', async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(env.webgpu, 'device');
  t.after(() => {
    if (descriptor) Object.defineProperty(env.webgpu, 'device', descriptor);
    else delete env.webgpu.device;
  });
  const fakeDevice = () => {
    const lost = Promise.withResolvers();
    const then = lost.promise.then.bind(lost.promise);
    const subscription = t.mock.method(lost.promise, 'then', (...args) => then(...args));
    return { device: { lost: lost.promise }, lose: lost.resolve, count: () => subscription.mock.callCount() };
  };
  const first = fakeDevice();
  const second = fakeDevice();
  Object.defineProperty(env.webgpu, 'device', { value: first.device, configurable: true, writable: true });
  const received = [];
  const monitor = createOrtDeviceLossMonitor((loss) => received.push(loss));
  for (let i = 0; i < 100; i++) monitor.refresh();
  const stale = [];
  const stop = watchOrtDeviceLoss((loss) => stale.push(loss));
  stop();
  assert.equal(first.count(), 1);
  env.webgpu.device = second.device;
  monitor.refresh();
  first.lose({ reason: 'destroyed', message: 'retired' });
  await tick();
  assert.deepEqual(received, []);
  assert.deepEqual(stale, []);
  for (let i = 0; i < 100; i++) monitor.refresh();
  second.lose({ reason: 'unknown', message: 'current' });
  await tick();
  assert.deepEqual(received, [{ reason: 'unknown', message: 'current' }]);
  assert.equal(second.count(), 1);
  monitor.refresh();
  await tick();
  assert.equal(received.length, 1);
  monitor.stop();
  const late = [];
  const unsubscribe = watchOrtDeviceLoss((loss) => late.push(loss));
  unsubscribe();
  await tick();
  assert.deepEqual(late, []);
  const cleanup = watchOrtDeviceLoss(() => {
    throw new Error('diagnostic owner failed');
  });
  await tick(); // A late subscriber must not create an uncaught exception in the worker.
  cleanup();
});

test('both worker producers emit deviceLost without a request id and use one owner monitor', () => {
  for (const path of ['../src/lc0/searchWorkerCore.ts', '../src/lc0/maia3Worker.ts']) {
    const source = read(path);
    assert.match(source, /const deviceLossMonitor = createOrtDeviceLossMonitor\(\(loss\) => post\(\{ type: 'deviceLost', loss \}\)\)/);
    assert.equal((source.match(/createOrtDeviceLossMonitor\(/g) ?? []).length, 1);
    assert.doesNotMatch(source, /type: 'deviceLost', id:/);
  }
});
