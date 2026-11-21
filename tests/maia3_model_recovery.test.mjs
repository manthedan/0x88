import assert from 'node:assert/strict';
import test from 'node:test';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';

const URL = 'http://localhost/models/maia3/unit.onnx';
const ARTIFACT = 'http://localhost/artifacts/maia-verified';
function modelLoad(fresh = false) {
  return {
    model: new TextEncoder().encode('abc').buffer,
    url: ARTIFACT,
    logicalUrl: URL,
    mode: 'cache',
    cacheStatus: fresh ? 'miss' : 'hit',
    verification: fresh ? 'hashed' : 'metadata',
    sha256Valid: true,
    expectedSha256: 'a'.repeat(64),
    elapsedMs: 0,
    telemetry: { source: 'cache-storage', totalMs: 0 },
  };
}
function setup(t, error = 'INVALID_PROTOBUF', freshFailure = false) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  const workers = [],
    loads = [],
    evictions = [];
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: class extends EventTarget {
      constructor() {
        super();
        this.index = workers.length;
        workers.push(this);
      }
      terminated = false;
      postMessage(message) {
        if (message.type !== 'init') return;
        queueMicrotask(() =>
          this.dispatchEvent(
            new MessageEvent('message', {
              data:
                this.index === 0 || freshFailure
                  ? { type: 'error', id: message.id, message: error }
                  : { type: 'ready', id: message.id, backend: 'test', inputNames: [], outputNames: [] },
            }),
          ),
        );
      }
      terminate() {
        this.terminated = true;
      }
    },
  });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'Worker', previous);
    else delete globalThis.Worker;
  });
  const deps = {
    load: async (url, options) => {
      loads.push({ url, options });
      return modelLoad(options.forceReload);
    },
    clear: async (...args) => {
      evictions.push(args);
      return true;
    },
  };
  return { deps, workers, loads, evictions };
}

test('Maia boots its worker while verified model bytes are still pending', async (t) => {
  const { deps, workers } = setup(t);
  const download = Promise.withResolvers();
  deps.load = () => download.promise;
  const init = t.mock.method(Worker.prototype, 'postMessage', function (message) {
    if (message.type === 'init') this.dispatchEvent(new MessageEvent('message', { data: { type: 'ready', id: message.id } }));
    if (message.type === 'dispose') this.dispatchEvent(new MessageEvent('message', { data: { type: 'disposed', id: message.id } }));
  });
  const creating = Maia3BrowserEvaluator.create({ modelUrl: URL }, deps);
  try {
    assert.equal(workers.length, 1);
    assert.equal(init.mock.callCount(), 0, 'the worker must not init a session with unverified bytes');
  } finally {
    download.resolve(modelLoad());
    const evaluator = await creating;
    await evaluator.dispose();
  }
});

test('a Maia worker boot failure cancels its pending model load', async (t) => {
  const { deps, workers } = setup(t);
  let loadSignal;
  deps.load = (_url, { signal }) => {
    loadSignal = signal;
    return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }));
  };
  const posted = t.mock.method(Worker.prototype, 'postMessage', function (message) {
    if (message.type === 'dispose') this.dispatchEvent(new MessageEvent('message', { data: { type: 'disposed', id: message.id } }));
  });
  const creating = Maia3BrowserEvaluator.create({ modelUrl: URL }, deps);
  assert.equal(workers.length, 1);
  workers[0].dispatchEvent(Object.assign(new Event('error'), { message: 'worker bootstrap failed' }));
  await assert.rejects(creating, /worker bootstrap failed/);
  assert.equal(loadSignal.aborted, true);
  assert.deepEqual(posted.mock.calls.map(({ arguments: [message] }) => message.type), ['cancel', 'dispose']);
  assert.equal(workers[0].terminated, true);
});

test('Maia abort rejects init promptly but publishes its bounded retirement promise to the owner', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const fake = setup(t);
  t.mock.method(Worker.prototype, 'postMessage', () => {}); // No ready or release ack.
  const abort = new AbortController();
  let retirement;
  const creating = Maia3BrowserEvaluator.create(
    {
      modelUrl: URL,
      signal: abort.signal,
      onInitCleanup: (cleanup) => {
        retirement = cleanup;
      },
    },
    fake.deps,
  );
  for (let i = 0; i < 10 && !fake.workers.length; i++) await Promise.resolve();
  assert.equal(fake.workers.length, 1);
  const rejected = assert.rejects(creating, { name: 'AbortError' });
  abort.abort();
  await rejected;
  assert.ok(retirement instanceof Promise);
  assert.equal(fake.workers[0].terminated, false);
  t.mock.timers.tick(1499);
  assert.equal(fake.workers[0].terminated, false);
  t.mock.timers.tick(1);
  await retirement;
  assert.equal(fake.workers[0].terminated, true);
});

test('Maia publishes its worker close synchronously at abort, before the owner snapshots retirements', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const fake = setup(t);
  t.mock.method(Worker.prototype, 'postMessage', () => {}); // Init never acks.
  const abort = new AbortController();
  const published = [];
  const creating = Maia3BrowserEvaluator.create({ modelUrl: URL, signal: abort.signal, onInitCleanup: (cleanup) => published.push(cleanup) }, fake.deps);
  for (let i = 0; i < 10 && !fake.workers.length; i++) await Promise.resolve();
  const rejected = assert.rejects(creating, { name: 'AbortError' });
  abort.abort();
  assert.equal(published.length, 1, 'Play switching to Lc0 Small snapshots the barrier right after abort()');
  await rejected;
  assert.equal(published.length, 1, 'the close is published once');
  t.mock.timers.tick(1500);
  await published[0];
  assert.equal(fake.workers[0].terminated, true);
});

test('Maia repairs the exact metadata-trusted cache entry once, in a fresh worker', async (t) => {
  const fake = setup(t);
  const evaluator = await Maia3BrowserEvaluator.create({ modelUrl: URL }, fake.deps);
  t.after(() => evaluator.terminate());
  assert.equal(fake.workers.length, 2);
  assert.equal(fake.workers[0].terminated, true);
  assert.equal(fake.workers[1].terminated, false);
  assert.deepEqual(fake.evictions, [[ARTIFACT, 'maia3-browser-models-v1']]);
  assert.deepEqual(
    fake.loads.map((l) => l.url),
    [URL, URL],
  );
  assert.equal(fake.loads[1].options.forceReload, true);
  assert.equal(fake.loads[1].options.allowUnverified, false);
  assert.equal(evaluator.modelLoad.verification, 'hashed');
  assert.equal(evaluator.modelLoad.revalidated, true);
});

test('Maia can repair metadata-trusted bytes reacquired by the worker after an initially hashed load', async (t) => {
  const fake = setup(t);
  const load = fake.deps.load;
  fake.deps.load = async (...args) => ({ ...(await load(...args)), cacheStatus: 'miss', verification: 'hashed' });
  t.mock.method(Worker.prototype, 'postMessage', function (message) {
    if (message.type !== 'init') return;
    const { model: _model, ...info } = modelLoad();
    queueMicrotask(() =>
      this.dispatchEvent(
        new MessageEvent('message', {
          data:
            this.index === 0
              ? { type: 'error', id: message.id, message: 'allocation failed (fallback: INVALID_PROTOBUF)', modelLoad: info, modelParseError: true }
              : { type: 'ready', id: message.id, backend: 'test', inputNames: [], outputNames: [] },
        }),
      ),
    );
  });
  const evaluator = await Maia3BrowserEvaluator.create({ modelUrl: URL }, fake.deps);
  t.after(() => evaluator.terminate());
  assert.equal(fake.loads.length, 2);
  assert.equal(fake.loads[1].options.forceReload, true);
  assert.deepEqual(fake.evictions, [[ARTIFACT, 'maia3-browser-models-v1']]);
});

test('Maia propagates a second parse failure without another worker/fetch', async (t) => {
  const fake = setup(t, 'INVALID_PROTOBUF', true);
  await assert.rejects(Maia3BrowserEvaluator.create({ modelUrl: URL }, fake.deps), /INVALID_PROTOBUF/);
  assert.equal(fake.workers.length, 2);
  assert.ok(fake.workers.every((w) => w.terminated));
  assert.equal(fake.loads.length, 2);
  assert.equal(fake.evictions.length, 1);
});

test('Maia does not evict a good cache entry on device failure', async (t) => {
  const fake = setup(t, 'WebGPU device lost');
  await assert.rejects(Maia3BrowserEvaluator.create({ modelUrl: URL }, fake.deps), /device lost/);
  assert.equal(fake.loads.length, 1);
  assert.equal(fake.evictions.length, 0);
  assert.equal(fake.workers[0].terminated, true);
});

test('Maia does not hand an unverified recovery body to another worker', async (t) => {
  const fake = setup(t);
  const load = fake.deps.load;
  fake.deps.load = async (...args) => {
    const result = await load(...args);
    return args[1].forceReload ? { ...result, verification: 'unchecked' } : result;
  };
  await assert.rejects(Maia3BrowserEvaluator.create({ modelUrl: URL }, fake.deps), /freshly SHA-256-verified/);
  assert.equal(fake.workers.length, 1);
});
