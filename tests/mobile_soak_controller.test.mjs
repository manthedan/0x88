import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MobileSoakDocument, mountMobileSoak } from '../src/lc0/mobileSoakBrowser.ts';
import { SOAK_STORAGE_KEY } from '../src/lc0/mobileSoak.ts';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => Promise.withResolvers();
async function until(predicate) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await tick();
  }
  assert.fail('Controller did not reach the expected checkpoint');
}
function sample(id, fen) {
  return { engine: id, fen, ms: 10, count: 8, perSecond: 800, unit: 'visits', move: 'e2e4' };
}
function engine(id, overrides = {}) {
  return {
    id,
    async load() {},
    async search(fen) {
      return sample(id, fen);
    },
    runtime: () => ({ backend: 'test' }),
    deviceLost: () => null,
    async dispose() {},
    ...overrides,
  };
}
function browser(t, search = '?steps=sf-lite,t3') {
  const storage = new Map();
  const writes = [];
  const scope = new MobileSoakDocument();
  const fakeWindow = Object.assign(new EventTarget(), {
    localStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem(key, value) {
        storage.set(key, value);
        writes.push(value);
      },
    },
  });
  const fakeDocument = Object.assign(new EventTarget(), { visibilityState: 'visible' });
  const fakeLocation = { search };
  const originals = new Map();
  for (const [key, value] of Object.entries({ window: fakeWindow, document: fakeDocument, location: fakeLocation })) {
    originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  }
  const controllers = [];
  t.after(() => {
    for (const controller of controllers) controller.destroy();
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const services = {
    environment: async () => ({ source: 'current document' }),
    memory: async () => ({ at: '2026-11-16T00:00:00.000Z', jsHeap: null, userAgentBytes: null, wasmHeapBytes: null, error: null }),
    makeEngine: async (id) => engine(id),
    documentScope: scope,
  };
  return {
    scope,
    writes,
    location: fakeLocation,
    savedText: () => storage.get(SOAK_STORAGE_KEY),
    saved: () => JSON.parse(storage.get(SOAK_STORAGE_KEY)),
    mount(overrides = {}) {
      const changes = [];
      const controller = mountMobileSoak((report) => changes.push(report), { ...services, ...overrides });
      controllers.push(controller);
      return { ...controller, changes };
    },
    lifecycle(type, persisted = false) {
      fakeWindow.dispatchEvent(Object.assign(new Event(type), { persisted }));
    },
    hide() {
      fakeDocument.visibilityState = 'hidden';
      fakeDocument.dispatchEvent(new Event('visibilitychange'));
    },
  };
}

test('retired initial environment/auto continuation and stale API cannot overwrite a reopened run', async (t) => {
  const f = browser(t, '?auto=1&steps=sf-lite');
  const environment = deferred();
  let retiredAllocations = 0;
  const retired = f.mount({
    environment: () => environment.promise,
    makeEngine: async () => {
      retiredAllocations++;
      return engine('sf-lite');
    },
  });
  retired.destroy();
  const retiredChanges = retired.changes.length;
  f.location.search = '?steps=t3';
  const load = deferred();
  const current = f.mount({ makeEngine: async (id) => engine(id, { load: () => load.promise }) });
  await current.ready;
  const run = current.api.start();
  await until(() => f.saved().inProgress?.id === 't3');
  const saved = f.savedText();
  const writes = f.writes.length;
  environment.resolve({ source: 'retired' });
  await retired.ready;
  await tick();
  await retired.api.start(['sf-lite']);
  await retired.api.continueAfterCrash();
  retired.api.reset();
  assert.equal(retiredAllocations, 0, 'pending auto mode must not allocate after destroy');
  assert.equal(retired.changes.length, retiredChanges);
  assert.equal(f.writes.length, writes, 'retired work must not even attempt a storage write');
  assert.equal(f.savedText(), saved, 'new controller marker is untouched');
  assert.equal(window.__mobileSoak, current.api);
  load.resolve();
  await run;
  assert.equal(current.api.results.steps[0].status, 'ok');
});

test('retired per-run environment refresh cannot publish after navigation', async (t) => {
  const f = browser(t);
  const refresh = deferred();
  let probes = 0;
  const retired = f.mount({ environment: () => (++probes === 1 ? Promise.resolve({ source: 'initial' }) : refresh.promise) });
  await retired.ready;
  const pending = retired.api.start();
  await until(() => probes === 2);
  retired.destroy();
  const current = f.mount();
  await current.ready;
  await current.api.start([]);
  const saved = f.savedText();
  const writes = f.writes.length;
  refresh.resolve({ source: 'retired refresh' });
  await pending;
  assert.equal(f.savedText(), saved);
  assert.equal(f.writes.length, writes);
  assert.equal(current.api.state, 'done');
});

test('retired step cleanup cannot overwrite recovery; remount waits for the old teardown', async (t) => {
  const f = browser(t);
  const search = deferred();
  const dispose = deferred();
  const nextLoad = deferred();
  let searches = 0;
  let disposals = 0;
  let nextAllocations = 0;
  const makeEngine = async (id) =>
    id === 'sf-lite'
      ? engine(id, {
          search(fen) {
            searches++;
            return search.promise.then(() => sample(id, fen));
          },
          dispose() {
            disposals++;
            return dispose.promise;
          },
        })
      : (nextAllocations++, engine(id, { load: () => nextLoad.promise }));
  const retired = f.mount({ makeEngine });
  await retired.ready;
  const pending = retired.api.start();
  await until(() => searches === 1);
  f.lifecycle('pagehide');
  f.hide();
  assert.equal(f.saved().inProgress.id, 'sf-lite');
  assert.deepEqual(
    f
      .saved()
      .events.slice(-2)
      .map((event) => event.type),
    ['pagehide', 'visibilitychange'],
  );
  retired.destroy();
  const changes = retired.changes.length;
  const current = f.mount({ makeEngine });
  await current.ready;
  assert.equal(current.api.state, 'crashed-awaiting-choice');
  assert.equal(current.api.results.steps[0].status, 'crashed');
  const continued = current.api.continueAfterCrash();
  await tick();
  const saved = f.savedText();
  const writes = f.writes.length;
  search.resolve();
  await until(() => disposals === 1);
  assert.equal(nextAllocations, 0, 'new controller may not race pending cleanup');
  assert.equal(searches, 1, 'retired controller must not search the second position');
  assert.equal(f.savedText(), saved);
  assert.equal(f.writes.length, writes);
  assert.equal(retired.changes.length, changes);
  dispose.resolve();
  await pending;
  await until(() => nextAllocations === 1);
  assert.equal(f.saved().inProgress.id, 't3');
  assert.equal(f.saved().steps[0].status, 'crashed');
  nextLoad.resolve();
  await continued;
  assert.deepEqual(
    current.api.results.steps.map((step) => step.status),
    ['crashed', 'ok'],
  );
});

test('mounted pagehide snapshot still preserves the breadcrumb through cleanup and BFCache restore', async (t) => {
  const f = browser(t, '?steps=sf-lite');
  const search = deferred();
  let searches = 0;
  const current = f.mount({
    makeEngine: async (id) =>
      engine(id, {
        search(fen) {
          searches++;
          return search.promise.then(() => sample(id, fen));
        },
      }),
  });
  await current.ready;
  const run = current.api.start();
  await until(() => searches === 1);
  f.lifecycle('pagehide', true);
  const changes = current.changes.length;
  search.resolve();
  await run;
  assert.equal(current.api.state, 'done');
  assert.equal(current.changes.length, changes, 'departing UI remains frozen');
  assert.equal(f.saved().inProgress.id, 'sf-lite', 'step cleanup must not clear departure marker');
  f.hide();
  assert.equal(f.saved().inProgress.id, 'sf-lite');
  f.lifecycle('pageshow', true);
  assert.equal(f.saved().state, 'done');
  assert.equal(f.saved().inProgress, null);
  assert.equal(f.saved().events.at(-1).type, 'pageshow');
});

test('unsuccessful teardown blocks Start, Continue and Reset across remounts until document reload', async (t) => {
  const f = browser(t);
  const allocations = [];
  const makeEngine = async (id) => {
    allocations.push(id);
    return engine(id, {
      async dispose() {
        if (id === 'sf-lite') throw new Error('release failed');
      },
    });
  };
  const failed = f.mount({ makeEngine });
  await failed.ready;
  await failed.api.start();
  assert.equal(failed.api.state, 'done');
  assert.equal(failed.api.unsafeDisposal, true);
  assert.equal(failed.api.results.steps[0].status, 'failed');
  assert.match(failed.api.results.steps[0].error, /Disposal: Error: release failed/);
  assert.equal(failed.api.results.steps[1].status, 'skipped');
  assert.match(failed.api.results.steps[1].error, /Reload the document/);
  assert.ok(failed.api.results.events.some((event) => event.type === 'unsafe-disposal'));
  assert.equal(failed.changes.at(-1).state, 'done', 'UI is republished with the latched API flag');
  const report = failed.api.results;
  const saved = f.savedText();
  await failed.api.start(['t3']);
  await failed.api.continueAfterCrash();
  failed.api.reset();
  assert.equal(failed.api.results, report, 'Reset must not erase the evidence or unlock ownership');
  assert.equal(f.savedText(), saved);
  assert.deepEqual(allocations, ['sf-lite']);
  failed.destroy();
  const reopened = f.mount({ makeEngine });
  await reopened.ready;
  assert.equal(reopened.api.unsafeDisposal, true, 'route navigation is not a document reload');
  const restored = reopened.api.results;
  const reopenedSaved = f.savedText();
  await reopened.api.start(['t3']);
  reopened.api.reset();
  await reopened.api.continueAfterCrash();
  assert.equal(reopened.api.results, restored);
  assert.equal(f.savedText(), reopenedSaved);
  assert.deepEqual(allocations, ['sf-lite']);
  reopened.destroy();
  const reloaded = f.mount({ makeEngine, documentScope: new MobileSoakDocument() });
  await reloaded.ready;
  assert.equal(reloaded.api.unsafeDisposal, false, 'only a fresh document clears the latch');
  await reloaded.api.start(['t3']);
  assert.deepEqual(allocations, ['sf-lite', 't3']);
  assert.equal(reloaded.api.results.steps[0].status, 'ok');
});

test('late retired disposal failure notifies the current UI and blocks an already queued remount', async (t) => {
  const f = browser(t);
  const search = deferred();
  let searching = false;
  const allocations = [];
  const makeEngine = async (id) => {
    allocations.push(id);
    return engine(id, {
      search(fen) {
        searching = true;
        return search.promise.then(() => sample(id, fen));
      },
      async dispose() {
        throw new Error('late release failure');
      },
    });
  };
  const retired = f.mount({ makeEngine });
  await retired.ready;
  const run = retired.api.start();
  await until(() => searching);
  retired.destroy();
  const current = f.mount({ makeEngine });
  await current.ready;
  assert.equal(current.api.state, 'crashed-awaiting-choice');
  const continued = current.api.continueAfterCrash();
  await tick();
  const updates = current.changes.length;
  search.resolve();
  await Promise.all([run, continued]);
  assert.deepEqual(allocations, ['sf-lite']);
  assert.equal(current.api.unsafeDisposal, true);
  assert.equal(current.api.state, 'done');
  assert.ok(current.changes.length > updates, 'late ownership failure must refresh the mounted UI');
  assert.deepEqual(
    current.api.results.steps.map((step) => step.status),
    ['crashed', 'skipped'],
  );
  assert.equal(f.saved().steps[0].status, 'crashed', 'retired failed result cannot replace the recovered crash');
  assert.equal(f.saved().steps[1].status, 'skipped');
});
