import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createOrtDeviceLossMonitor, watchOrtDeviceLoss } from '../src/nn/ortDeviceLoss.ts';
import { env } from '../src/nn/ortRuntime.ts';
import { Bt4WorkerSearcher, BT4_NET } from '../src/lc0/bt4Engine.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';

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
  ['../src/lc0/analysisBrowser.ts', 'initWorker', 'workerPending', { searchWorker: 1, worker: 1, showModelProgress() {}, analysisWorkerProgressMessages: 0 }],
  ['../src/lc0/criticalMomentsEngines.ts', 'createCriticalEngines', 'pending', {}],
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
  terminate() {}
  emit(data) {
    this.dispatchEvent(new MessageEvent('message', { data }));
  }
}

test('Bt4 init keeps telemetry separate from ready and stores the loss', async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  let worker;
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: class extends FakeWorker {
      constructor() {
        super();
        worker = this;
      }
    },
  });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor);
    else delete globalThis.Worker;
  });
  const searcher = new Bt4WorkerSearcher(BT4_NET);
  const init = searcher.init();
  const id = worker.posted[0].id;
  for (const payload of [
    { type: 'deviceLost', loss: { reason: 'unknown', message: 'reset' } },
    { type: 'deviceLost', id, loss: { reason: 'unknown', message: 'reset' } },
  ])
    worker.emit(payload);
  await tick();
  assert.equal(searcher.loaded, false);
  assert.equal(searcher.deviceLost.message, 'reset');
  worker.emit({ type: 'ready', id, backend: 'expected', servedModelUrl: 'test' });
  assert.equal(await init, 'expected');
  searcher.dispose();
});

test('Maia init and inference handlers never resolve a request from device telemetry', async () => {
  const worker = new FakeWorker();
  const init = Maia3BrowserEvaluator.postInit(worker, 'test');
  let initialized = false;
  void init.then(() => {
    initialized = true;
  });
  worker.emit({ type: 'deviceLost', loss: {} });
  worker.emit({ type: 'deviceLost', id: 0, loss: {} });
  await tick();
  assert.equal(initialized, false);
  worker.emit({ type: 'ready', id: 0, backend: 'expected', inputNames: [], outputNames: [] });
  assert.equal((await init).backend, 'expected');
  let result;
  const owner = {
    deviceLost: null,
    pending: new Map([
      [
        1,
        {
          resolve: (value) => {
            result = value;
          },
        },
      ],
    ]),
  };
  Maia3BrowserEvaluator.prototype.onMessage.call(owner, { data: { type: 'deviceLost', id: 1, loss: { reason: 'unknown', message: 'reset' } } });
  assert.equal(result, undefined);
  assert.equal(owner.pending.size, 1);
  assert.equal(owner.deviceLost.message, 'reset');
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
