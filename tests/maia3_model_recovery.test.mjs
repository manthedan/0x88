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
