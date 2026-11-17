import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import ts from 'typescript';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { beginSoakStep, emptySoakReport, parseSoakQuery, SOAK_STORAGE_KEY } from '../src/lc0/mobileSoak.ts';
import { MobileSoakDocument, mountMobileSoak } from '../src/lc0/mobileSoakBrowser.ts';

// Execute the real adapter without importing unrelated TVM-only TypeScript parameter properties.
const engineSource = readFileSync(new URL('../src/lc0/mobileSoakEngines.ts', import.meta.url), 'utf8');
const makeSoakEngine = new Function(
  'Maia3BrowserEvaluator',
  'BIG_NETS',
  'isDefaultBrowserUciFamily',
  'ortDequantFoldEnabled',
  'requestedOrtGraphOptimizationLevel',
  'MAIA3_ORT_DEFAULTS',
  ts.transpile(engineSource.slice(engineSource.indexOf('export function makeSoakEngine')).replace('export function', 'function'), {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
  }) + '; return makeSoakEngine;',
)(
  Maia3BrowserEvaluator,
  {},
  () => false,
  () => true,
  () => 'all',
  { dequantFold: true },
);

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

for (const mode of ['manual', 'auto', 'continue']) {
  for (const accepted of [false, true]) {
    test(`${mode} diagnostics await explicit phone consent before engines or ballast (accepted=${accepted})`, async (t) => {
      const search = `?steps=sf-lite,t3&labBallastMb=1&t3Model=/models/lc0/diagnostic.onnx${mode === 'auto' ? '&auto=1' : ''}`;
      const f = browser(t, search);
      if (mode === 'continue') {
        const at = '2026-11-16T00:00:00.000Z';
        const interrupted = beginSoakStep(emptySoakReport(parseSoakQuery(search), at), 'sf-lite', at);
        window.localStorage.setItem(SOAK_STORAGE_KEY, JSON.stringify(interrupted));
      }
      const permission = deferred();
      const requests = [];
      const loaded = [];
      const controller = f.mount({
        requestConsent: async (ids, signal, overrides) => {
          assert.equal(overrides.t3, '/models/lc0/diagnostic.onnx', 'effective model overrides reach the consent decision');
          requests.push([...ids]);
          assert.equal(signal.aborted, false);
          return permission.promise;
        },
        makeEngine: async (id) => {
          loaded.push(id);
          return engine(id);
        },
      });
      await controller.ready;
      const running = mode === 'auto' ? null : mode === 'continue' ? controller.api.continueAfterCrash() : controller.api.start();
      await until(() => requests.length === 1);
      assert.deepEqual(requests[0], mode === 'continue' ? ['t3'] : ['sf-lite', 't3']);
      assert.deepEqual(loaded, []);
      assert.equal(
        controller.api.results.events.some((event) => event.type === 'ballast-start'),
        false,
      );
      permission.resolve(accepted);
      if (running) await running;
      else await until(() => controller.api.state === 'done');
      assert.deepEqual(loaded, accepted ? requests[0] : []);
      assert.equal(controller.api.state, 'done');
      if (!accepted) assert.ok(controller.api.results.steps.some((step) => step.status === 'skipped' && /consent declined/.test(step.error)));
    });
  }
}

test('leaving during the lab consent dialog cannot start engines after acceptance', async (t) => {
  const f = browser(t, '?steps=t3');
  const permission = deferred();
  let requested = false;
  let loaded = 0;
  const controller = f.mount({
    requestConsent: async () => {
      requested = true;
      return permission.promise;
    },
    makeEngine: async (id) => {
      loaded++;
      return engine(id);
    },
  });
  await controller.ready;
  const running = controller.api.start();
  await until(() => requested);
  controller.destroy();
  permission.resolve(true);
  await running;
  assert.equal(loaded, 0);
});

for (const failCleanup of [false, true]) {
  test(`aborted Maia init keeps remount behind owned cleanup and propagates teardown failure (${failCleanup})`, async (t) => {
    const f = browser(t, '?steps=maia3,t3');
    const cleanup = deferred();
    let initializing = false;
    let newLoads = 0;
    t.mock.method(
      Maia3BrowserEvaluator,
      'create',
      ({ signal, onInitCleanup }) =>
        new Promise((_, reject) => {
          initializing = true;
          signal.addEventListener(
            'abort',
            () => {
              onInitCleanup(cleanup.promise);
              reject(new DOMException('Maia init cancelled', 'AbortError'));
            },
            { once: true },
          );
        }),
    );
    const makeEngine = async (id, signal) =>
      id === 'maia3'
        ? makeSoakEngine(id, signal)
        : engine(id, {
            load: async () => {
              newLoads++;
            },
          });
    const old = f.mount({ makeEngine });
    await old.ready;
    const oldRun = old.api.start();
    await until(() => initializing);
    old.destroy();
    const current = f.mount({ makeEngine });
    await current.ready;
    assert.equal(current.api.state, 'crashed-awaiting-choice');
    const resumed = current.api.continueAfterCrash();
    await tick();
    assert.equal(newLoads, 0, 'logical init cancellation must not release the document queue');
    if (failCleanup) cleanup.reject(new Error('retired Maia release failed'));
    else cleanup.resolve();
    await Promise.all([oldRun, resumed]);
    assert.equal(current.api.unsafeDisposal, failCleanup);
    assert.equal(newLoads, failCleanup ? 0 : 1);
  });
}

test('lab gap follows awaited teardown, holds ballast, and custom Start preserves params', async (t) => {
  const f = browser(t, '?steps=t3,maia3&labGapSeconds=20&labBallastMb=1');
  const release = deferred();
  const gap = deferred();
  const calls = [];
  const c = f.mount({
    makeEngine: async (id) =>
      engine(id, {
        load: async () => {
          calls.push(`load:${id}`);
          assert.equal(c.api.results.labBallastAllocatedBytes, 1048576);
        },
        dispose: async () => {
          calls.push(`dispose:${id}`);
          if (id === 't3') await release.promise;
        },
      }),
    wait: async (ms) => {
      calls.push(`gap:${ms}`);
      await gap.promise;
    },
  });
  await c.ready;
  const run = c.api.start(['t3', 'maia3']);
  await until(() => calls.includes('dispose:t3'));
  assert.deepEqual(calls, ['load:t3', 'dispose:t3']);
  release.resolve();
  await until(() => calls.includes('gap:20000'));
  assert.equal(c.api.results.steps.length, 1);
  gap.resolve();
  await run;
  assert.deepEqual(calls, ['load:t3', 'dispose:t3', 'gap:20000', 'load:maia3', 'dispose:maia3']);
  assert.equal(c.api.results.labBallastMb, 1);
  assert.equal(c.api.results.events.filter((e) => e.type === 'gap-start').length, 1);
  assert.equal(c.api.state, 'done');
});

test('destroy aborts a lab gap without admitting the next engine', async (t) => {
  const f = browser(t, '?steps=t3,maia3&labGapSeconds=20');
  const calls = [];
  const c = f.mount({
    makeEngine: async (id) => {
      calls.push(id);
      return engine(id);
    },
  });
  await c.ready;
  const run = c.api.start();
  await until(() => c.api.results.events.some((e) => e.type === 'gap-start'));
  c.destroy();
  await run;
  assert.deepEqual(calls, ['t3']);
});

test('labSearchSeconds runs one timed production-adapter search after one load in the measured step', async (t) => {
  const f = browser(t, '?steps=lqo&labSearchSeconds=90');
  let loads = 0,
    searches = 0,
    disposals = 0;
  const current = f.mount({
    makeEngine: async (id) =>
      engine(id, {
        async load() {
          loads++;
        },
        async search(fen, options) {
          searches++;
          assert.equal(options.durationMs, 90000);
          options.progress(8);
          return sample(id, fen);
        },
        async dispose() {
          disposals++;
        },
      }),
  });
  await current.ready;
  await current.api.start();
  assert.deepEqual([loads, searches, disposals], [1, 1, 1]);
  assert.equal(current.api.results.steps[0].searches.length, 1);
  assert.equal(current.api.results.steps[0].status, 'ok');
});

test('active API, storage and published rows stay running through load, search and disposal', async (t) => {
  const f = browser(t, '?steps=sf-lite');
  const load = deferred(),
    search = deferred(),
    dispose = deferred();
  let loading = false,
    searching = false,
    disposing = false;
  const current = f.mount({
    makeEngine: async (id) =>
      engine(id, {
        load() {
          loading = true;
          return load.promise;
        },
        search(fen) {
          searching = true;
          return search.promise.then(() => sample(id, fen));
        },
        dispose() {
          disposing = true;
          return dispose.promise;
        },
      }),
  });
  await current.ready;
  const run = current.api.start();
  const assertRunning = () => {
    assert.equal(current.api.state, 'running');
    assert.equal(current.api.results.inProgress.id, 'sf-lite');
    assert.equal(current.api.results.steps[0].status, 'running');
    assert.equal(f.saved().steps[0].status, 'running');
    assert.equal(current.changes.at(-1).steps[0].status, 'running');
  };
  await until(() => loading);
  assertRunning();
  load.resolve();
  await until(() => searching);
  assertRunning();
  search.resolve();
  await until(() => disposing);
  assertRunning();
  dispose.resolve();
  await run;
  assert.equal(current.api.results.inProgress, null);
  assert.equal(current.api.results.steps[0].status, 'ok');
  assert.equal(f.saved().steps[0].status, 'ok');
});

test('failed teardown is still running until its active marker clears, then becomes failed', async (t) => {
  const f = browser(t, '?steps=sf-lite');
  const disposal = deferred(),
    afterDispose = deferred();
  let memories = 0,
    disposing = false;
  const memory = { at: 'test', jsHeap: null, userAgentBytes: null, wasmHeapBytes: null, error: null };
  const current = f.mount({
    memory: async () => (++memories === 4 ? afterDispose.promise : memory),
    makeEngine: async (id) =>
      engine(id, {
        dispose() {
          disposing = true;
          return disposal.promise;
        },
      }),
  });
  await current.ready;
  const run = current.api.start();
  await until(() => disposing);
  disposal.reject(new Error('release failed'));
  await until(() => memories === 4);
  assert.equal(current.api.results.steps[0].status, 'running');
  assert.equal(current.api.results.inProgress.id, 'sf-lite');
  assert.match(current.api.results.steps[0].error, /release failed/);
  assert.equal(current.api.unsafeDisposal, true);
  afterDispose.resolve(memory);
  await run;
  assert.equal(current.api.results.steps[0].status, 'failed');
  assert.equal(current.api.results.inProgress, null);
});

test('API step selection and crash continuation preserve the saved run model overrides', async (t) => {
  const f = browser(t, '?steps=lqo,t3&lqoModel=/models/lc0/lqo_v2.f16.onnx&t3Model=/models/lc0/unit-t3.f16.onnx');
  const search = deferred();
  const calls = [];
  let searching = false;
  const makeEngine = async (id, _signal, modelOverride) => {
    calls.push([id, modelOverride]);
    return engine(id, {
      runtime: () => ({ requestedModelUrl: modelOverride }),
      search(fen) {
        searching = true;
        return search.promise.then(() => sample(id, fen));
      },
    });
  };
  const old = f.mount({ makeEngine });
  await old.ready;
  const pending = old.api.start(['lqo', 't3']);
  await until(() => searching);
  assert.equal(f.saved().steps[0].runtime.lqo.requestedModelUrl, '/models/lc0/lqo_v2.f16.onnx');
  old.destroy();
  f.location.search = '?steps=t3';
  const current = f.mount({ makeEngine });
  await current.ready;
  assert.equal(current.api.state, 'crashed-awaiting-choice');
  const continued = current.api.continueAfterCrash();
  search.resolve();
  await Promise.all([pending, continued]);
  assert.deepEqual(calls, [
    ['lqo', '/models/lc0/lqo_v2.f16.onnx'],
    ['t3', '/models/lc0/unit-t3.f16.onnx'],
  ]);
  assert.deepEqual(
    current.api.results.steps.map((s) => s.status),
    ['crashed', 'ok'],
  );
  assert.equal(current.api.results.modelOverrides.lqo, '/models/lc0/lqo_v2.f16.onnx');
});

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

test('navigation retires a pooled big net synchronously before production reacquires the key', async (t) => {
  const f = browser(t, '?steps=bt4');
  const pendingSearch = deferred();
  let searching = false;
  const pool = new BigNetSessionPool((config) => ({
    key: config.key,
    disposals: 0,
    dispose() {
      this.disposals++;
    },
  }));
  let acquired;
  const retired = f.mount({
    makeEngine: async (id) => {
      acquired = pool.acquire('bt4');
      const owned = acquired;
      const retire = () => {
        pool.detach('bt4', owned);
      };
      return engine(id, {
        search(fen) {
          searching = true;
          return pendingSearch.promise.then(() => sample(id, fen));
        },
        retire,
        async dispose() {
          retire();
          owned.dispose();
        },
      });
    },
  });
  await retired.ready;
  const run = retired.api.start();
  await until(() => searching);
  assert.equal(pool.peek('bt4'), acquired);
  retired.destroy();
  assert.equal(pool.has('bt4'), false, 'record must be detached before destroy returns, not in async finally');
  const destination = pool.acquire('bt4'); // Play/Arena/Analysis do not wait on the soak queue.
  assert.notEqual(destination, acquired, 'destination must not inherit the retiring searcher');
  pendingSearch.resolve();
  await run;
  assert.equal(acquired.disposals, 1);
  assert.equal(destination.disposals, 0);
  assert.equal(pool.peek('bt4'), destination, 'late cleanup must not remove the replacement');
  pool.disposeAllNow();
});

test('deferred CPU setup records preparation, init+search and memory phase without fabricated load or throughput', async (t) => {
  const f = browser(t, '?steps=viridithas');
  const current = f.mount({
    makeEngine: async (id) =>
      engine(id, {
        runtime: () => ({ backend: 'one-shot', loadDeferred: true, firstInitSearchMs: 2500 }),
        async search(fen) {
          return { ...sample(id, fen), ms: 2500, initializationIncluded: true, perSecond: null, engineReportedNps: 1000 };
        },
      }),
  });
  await current.ready;
  await current.api.start();
  const result = current.api.results.steps[0];
  assert.equal(result.status, 'ok');
  assert.equal(result.loadDeferred, true);
  assert.equal(result.loadMs, null);
  assert.equal(typeof result.prepareMs, 'number');
  assert.equal(result.firstInitSearchMs, 2500);
  assert.equal(result.firstSearchMs, 2500);
  assert.equal(result.searches[0].perSecond, null);
  assert.deepEqual(
    result.memory.map((memory) => memory.phase),
    ['before-load', 'after-prepare', 'after-search', 'after-dispose'],
  );
  assert.equal(f.saved().steps[0].loadMs, null);
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
