import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  EngineResourceBroker,
  cpuThreadBudget,
  loadCalibratedThreads,
  loadPerformanceDial,
  saveCalibratedThreads,
  savePerformanceDial,
} from '../src/lc0/resourceBroker.ts';
import { ENGINE_RESOURCE_PROFILES, engineResourceProfile } from '../src/lc0/engineCatalog.ts';

const isolated = (hardwareConcurrency, calibratedThreads) => ({
  hardwareConcurrency,
  crossOriginIsolated: true,
  ...(calibratedThreads !== undefined ? { calibratedThreads } : {}),
});

test('cpuThreadBudget is 1 without cross-origin isolation regardless of dial', () => {
  const env = { hardwareConcurrency: 16, crossOriginIsolated: false };
  assert.equal(cpuThreadBudget(env, 'eco'), 1);
  assert.equal(cpuThreadBudget(env, 'balanced'), 1);
  assert.equal(cpuThreadBudget(env, 'max'), 1);
});

test('cpuThreadBudget scales with the dial and never drops below 1', () => {
  assert.equal(cpuThreadBudget(isolated(10), 'eco'), 5);
  assert.equal(cpuThreadBudget(isolated(10), 'balanced'), 8);
  assert.equal(cpuThreadBudget(isolated(10), 'max'), 9);
  assert.equal(cpuThreadBudget(isolated(2), 'eco'), 1);
  assert.equal(cpuThreadBudget(isolated(2), 'balanced'), 1);
  assert.equal(cpuThreadBudget(isolated(1), 'max'), 1);
});

test('cpuThreadBudget prefers calibrated threads over hardwareConcurrency', () => {
  assert.equal(cpuThreadBudget(isolated(16, 8), 'balanced'), 6);
  assert.equal(cpuThreadBudget(isolated(16, 8), 'max'), 7);
});

test('exclusive policy grants the full budget and clamps to the engine profile', async () => {
  const broker = new EngineResourceBroker({ policy: 'exclusive', environment: () => isolated(10) });
  broker.register('sf', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('reckless', { resourceClass: 'cpu', maxThreads: 1 });

  const sf = await broker.acquire({ engineId: 'sf' });
  assert.equal(sf.threads, 8);
  sf.release();

  const reckless = await broker.acquire({ engineId: 'reckless' });
  assert.equal(reckless.threads, 1);
  reckless.release();
});

test('exclusive policy queues a second CPU lease until the first releases', async () => {
  const broker = new EngineResourceBroker({ policy: 'exclusive', environment: () => isolated(10) });
  broker.register('a', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('b', { resourceClass: 'cpu', maxThreads: 32 });

  const first = await broker.acquire({ engineId: 'a' });
  let secondGranted = false;
  const secondPromise = broker.acquire({ engineId: 'b' }).then((lease) => {
    secondGranted = true;
    return lease;
  });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(secondGranted, false);
  assert.deepEqual(broker.snapshot().queuedEngineIds, ['b']);

  first.release();
  const second = await secondPromise;
  assert.equal(secondGranted, true);
  assert.equal(second.threads, 8);
  second.release();
  assert.equal(broker.snapshot().activeLeases.length, 0);
});

test('queued exclusive requests are FIFO and abortable', async () => {
  const broker = new EngineResourceBroker({ policy: 'exclusive', environment: () => isolated(8) });
  for (const id of ['a', 'b', 'c']) broker.register(id, { resourceClass: 'cpu', maxThreads: 32 });

  const first = await broker.acquire({ engineId: 'a' });
  const abortController = new AbortController();
  const aborted = broker.acquire({ engineId: 'b', signal: abortController.signal });
  const third = broker.acquire({ engineId: 'c' });

  abortController.abort();
  await assert.rejects(aborted, (error) => error.name === 'AbortError');
  assert.deepEqual(broker.snapshot().queuedEngineIds, ['c']);

  first.release();
  const lease = await third;
  assert.equal(lease.engineId, 'c');
  lease.release();
});

test('acquire rejects immediately when the signal is already aborted', async () => {
  const broker = new EngineResourceBroker({ environment: () => isolated(8) });
  broker.register('a', { resourceClass: 'cpu', maxThreads: 32 });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(broker.acquire({ engineId: 'a', signal: controller.signal }), (error) => error.name === 'AbortError');
});

test('gpu leases never queue and never consume the cpu budget', async () => {
  const broker = new EngineResourceBroker({ policy: 'exclusive', environment: () => isolated(10) });
  broker.register('sf', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('lc0', { resourceClass: 'gpu', maxThreads: 1 });

  const sf = await broker.acquire({ engineId: 'sf' });
  const lc0 = await broker.acquire({ engineId: 'lc0' });
  assert.equal(lc0.threads, 1);
  assert.equal(sf.threads, 8);
  lc0.release();
  sf.release();
});

test('shared policy splits the budget across registered CPU participants', async () => {
  const broker = new EngineResourceBroker({ policy: 'shared', environment: () => isolated(10) });
  broker.register('sf-lite', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('sf-full', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('lc0', { resourceClass: 'gpu', maxThreads: 1 });

  // budget 8 over two equal-weight CPU participants -> 4 each, never blocking.
  const a = await broker.acquire({ engineId: 'sf-lite' });
  const b = await broker.acquire({ engineId: 'sf-full' });
  assert.equal(a.threads, 4);
  assert.equal(b.threads, 4);
  a.release();
  b.release();
});

test('shared policy respects weights, profile clamps, and the 1-thread floor', async () => {
  const broker = new EngineResourceBroker({ policy: 'shared', environment: () => isolated(10) });
  broker.register('big', { resourceClass: 'cpu', maxThreads: 32, weight: 3 });
  broker.register('small', { resourceClass: 'cpu', maxThreads: 32, weight: 1 });
  broker.register('single', { resourceClass: 'cpu', maxThreads: 1, weight: 4 });

  // budget 8, total weight 8 -> big floor(8*3/8)=3, small 1, single clamps to 1.
  assert.equal((await broker.acquire({ engineId: 'big' })).threads, 3);
  assert.equal((await broker.acquire({ engineId: 'small' })).threads, 1);
  assert.equal((await broker.acquire({ engineId: 'single' })).threads, 1);
});

test('release is idempotent and drains at most one exclusive waiter', async () => {
  const broker = new EngineResourceBroker({ policy: 'exclusive', environment: () => isolated(8) });
  broker.register('a', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('b', { resourceClass: 'cpu', maxThreads: 32 });
  broker.register('c', { resourceClass: 'cpu', maxThreads: 32 });

  const first = await broker.acquire({ engineId: 'a' });
  const second = broker.acquire({ engineId: 'b' });
  const third = broker.acquire({ engineId: 'c' });
  first.release();
  first.release();
  const b = await second;
  assert.deepEqual(broker.snapshot().queuedEngineIds, ['c']);
  b.release();
  (await third).release();
});

test('acquire throws for unregistered engines', async () => {
  const broker = new EngineResourceBroker({ environment: () => isolated(4) });
  await assert.rejects(broker.acquire({ engineId: 'ghost' }), /not registered/);
});

test('catalog resource profiles cover every family with sane values', () => {
  for (const [family, profile] of Object.entries(ENGINE_RESOURCE_PROFILES)) {
    assert.ok(profile.resourceClass === 'cpu' || profile.resourceClass === 'gpu', family);
    assert.ok(profile.maxThreads >= 1, family);
    assert.deepEqual(engineResourceProfile(family), profile);
  }
  assert.equal(ENGINE_RESOURCE_PROFILES.sf.resourceClass, 'cpu');
  assert.ok(ENGINE_RESOURCE_PROFILES.sf.maxThreads > 1, 'threaded SF wasm is available today');
  assert.equal(ENGINE_RESOURCE_PROFILES.lc0.resourceClass, 'gpu');
});

test('dial and calibration persistence round-trips through a storage shim', () => {
  const data = new Map();
  const storage = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v) };
  assert.equal(loadPerformanceDial(storage), 'balanced');
  savePerformanceDial(storage, 'eco');
  assert.equal(loadPerformanceDial(storage), 'eco');
  assert.equal(loadCalibratedThreads(storage), undefined);
  saveCalibratedThreads(storage, 6.7);
  assert.equal(loadCalibratedThreads(storage), 6);
  assert.equal(loadPerformanceDial(undefined), 'balanced');
  assert.equal(loadCalibratedThreads(undefined), undefined);
});
