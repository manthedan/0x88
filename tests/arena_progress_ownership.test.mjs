import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { BT4_NET, Bt4WorkerSearcher, LQO_NET } from '../src/lc0/bt4Engine.ts';
import { LC0_MODEL_DOWNLOAD_NOT_APPROVED } from '../src/lc0/modelCache.ts';

// Execute the production closures, not a reimplementation; UI sinks are spies.
const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
const slice = (from, to) => stripTypeScriptTypes(source.slice(source.indexOf(from), source.indexOf(to, source.indexOf(from))));
for (const operation of ['warmup', 'move']) {
  for (const cancelled of [true, false]) {
    test(`Arena ${operation} progress stays with its ${cancelled ? 'cancelled' : 'live'} operation`, async (t) => {
      const workers = [];
      const events = [];
      const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
      globalThis.Worker = class extends EventTarget {
        messages = [];
        constructor() {
          super();
          workers.push(this);
        }
        postMessage(message) {
          this.messages.push(message);
          if (message.type === 'search') queueMicrotask(() => this.reply({ id: message.id, result: { visits: 1, move: 'e2e4' } }));
          if (message.type === 'resetSearch') queueMicrotask(() => this.reply({ id: message.id }));
        }
        reply(data) {
          this.dispatchEvent(new MessageEvent('message', { data }));
        }
        terminate() {}
      };
      const searcher = new Bt4WorkerSearcher(LQO_NET);
      t.after(() => {
        searcher.dispose();
        if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
        else delete globalThis.Worker;
      });
      const name = operation === 'warmup' ? 'lc0BigNetWarmup' : 'lc0BigNetMove';
      const start = source.indexOf(`  const ${name} =`);
      const end = source.indexOf(operation === 'warmup' ? '  const centipawnWarmup =' : '  const centipawnMove =', start);
      assert.ok(start >= 0 && end > start);
      const factory = new Function(
        'deps',
        `const { bigNetFor, warmupPositions, showDownloadProgress, hideDownloadProgress, arenaCacheEntries, engines, arenaBudgetMode, arenaMovetimeMs, setArenaVisitProgress, recordBt4SearchOutput, clearArenaSearchProgress, renderBigNetWarning } = deps;\n${stripTypeScriptTypes(source.slice(start, end))}\nreturn ${name};`,
      )({
        bigNetFor: () => ({ config: LQO_NET, searcher }),
        warmupPositions: [parseFen(START_FEN)],
        showDownloadProgress: (...args) => events.push(['show', ...args]),
        hideDownloadProgress: () => events.push(['hide']),
        arenaCacheEntries: () => 2048,
        engines: new Map(),
        arenaBudgetMode: () => 'visits',
        arenaMovetimeMs: () => 1000,
        setArenaVisitProgress: () => {},
        recordBt4SearchOutput: () => {},
        renderBigNetWarning: () => {},
        clearArenaSearchProgress: () => {},
      });
      const controller = new AbortController();
      const pending =
        operation === 'warmup'
          ? factory('lqo')(controller.signal)
          : factory('seat-1', { variant: 'lqo', strength: 1 }, 'lqo')([parseFen(START_FEN)], controller.signal);
      await new Promise(setImmediate);
      const worker = workers[0];
      assert.ok(worker, 'operation is awaiting shared model initialization');
      if (cancelled) {
        controller.abort();
        assert.deepEqual(events.at(-1), ['hide'], "Stop releases this operation's progress display at once");
      }
      // After Stop, a replacement progress surface now belongs to another op.
      events.length = 0;
      worker.reply({ type: 'downloadProgress', loadedBytes: 20, totalBytes: 100 });
      const init = worker.messages.find((message) => message.type === 'init');
      worker.reply({ id: init.id, backend: 'mock-webgpu' });
      await pending;
      assert.deepEqual(
        events,
        cancelled ? [] : [['show', 'Lc0 Queen Odds', 20, 100, 'Downloading'], ['hide']],
        'old download callbacks/finally must not modify the replacement surface',
      );
    });
  }
}

// A cold prefetch only watches its searcher's dispose(), and search() waits for init():
// without retirement, Stop would wait out a stalled download and the worker would init after it.
for (const operation of ['warmup', 'move']) {
  test(`Arena Stop during a stalled cold prefetch retires the net and settles the ${operation}`, async (t) => {
    const workers = [];
    const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
    globalThis.Worker = class extends EventTarget {
      constructor() {
        super();
        workers.push(this);
      }
      postMessage() {}
      terminate() {}
    };
    t.after(() => {
      if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
      else delete globalThis.Worker;
    });
    let prefetchSignal;
    const searcher = new Bt4WorkerSearcher(LQO_NET, (_config, options) => {
      prefetchSignal = options.signal;
      // A download that never finishes; only an abort ends it.
      return new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true }));
    });
    let pooled = searcher;
    const deps = {
      BIG_NETS: { bt4: LQO_NET },
      isMemoryConstrainedDevice: () => false,
      arenaRunDownloads: new Set(),
      arenaCacheEntries: () => 2048,
      peekBigNetSearcher: () => pooled,
      disposeBigNetSearcherNow: () => {
        pooled?.dispose();
        pooled = null;
      },
      acquireBigNetSearcherAsync: async () => searcher,
      warmupPositions: [parseFen(START_FEN)],
      showDownloadProgress: () => {},
      hideDownloadProgress: () => {},
      engines: new Map(),
      arenaBudgetMode: () => 'visits',
      arenaMovetimeMs: () => 1000,
      setArenaVisitProgress: () => {},
      recordBt4SearchOutput: () => {},
      clearArenaSearchProgress: () => {},
      renderBigNetWarning: () => {},
    };
    const name = operation === 'warmup' ? 'lc0BigNetWarmup' : 'lc0BigNetMove';
    const closure = slice(`  const ${name} =`, operation === 'warmup' ? '  const centipawnWarmup =' : '  const centipawnMove =');
    const factory = new Function(
      'deps',
      `const { ${Object.keys(deps)} } = deps;\n${slice('async function bigNetFor(', 'let runtimeIsolation')}\n${closure}\nreturn ${name};`,
    )(deps);
    const controller = new AbortController();
    const pending =
      operation === 'warmup'
        ? factory('bt4')(controller.signal)
        : factory('seat-1', { variant: 'bt4', strength: 1 }, 'bt4')([parseFen(START_FEN)], controller.signal);
    for (let i = 0; i < 50 && !prefetchSignal; i++) await new Promise(setImmediate);
    assert.ok(prefetchSignal, 'the cold load is prefetching');
    controller.abort();
    const settled = await Promise.race([pending.then(() => 'settled'), new Promise((resolve) => setTimeout(resolve, 200, 'stuck'))]);
    assert.equal(settled, 'settled', 'Stop does not wait for the stalled prefetch');
    assert.equal(prefetchSignal.aborted, true, 'the prefetch is cancelled');
    assert.equal(pooled, null, 'the cold searcher is retired from the pool');
    assert.equal(workers.length, 0, 'no session worker initializes after Stop');
  });
}

for (const approved of [false, true]) {
  test(`Arena does not inherit a warm worker's prior download approval during recovery (approved=${approved})`, async (t) => {
    const prevWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
    const inits = [];
    const searches = [];
    const prompts = [];
    const arenaRunDownloads = new Set(['bt4']);
    globalThis.Worker = class extends EventTarget {
      postMessage(message) {
        const emit = (data) => this.dispatchEvent(new MessageEvent('message', { data: { id: message.id, ...data } }));
        if (message.type === 'init') {
          inits.push(message);
          this.ep = message.ep;
          if (message.ep === 'wasm' && message.noModelDownload) {
            queueMicrotask(() => emit({ type: 'error', error: LC0_MODEL_DOWNLOAD_NOT_APPROVED }));
          } else {
            if (!message.noModelDownload) assert.equal(arenaRunDownloads.has('bt4'), true, 'network loading requires approval');
            queueMicrotask(() => emit({ type: 'ready', backend: message.ep === 'wasm' ? 'wasm' : 'auto->webgpu' }));
          }
        } else if (message.type === 'search') {
          searches.push(message.input);
          if (this.ep !== 'wasm') queueMicrotask(() => emit({ type: 'deviceLost', loss: { reason: 'unknown', message: 'reset' } }));
          else queueMicrotask(() => emit({ type: 'searchResult', result: { move: 'e2e4' } }));
        } else if (message.type === 'dispose') queueMicrotask(() => emit({ type: 'disposed' }));
      }
      terminate() {}
    };
    const searcher = new Bt4WorkerSearcher(BT4_NET, () => undefined);
    t.after(async () => {
      await searcher.dispose();
      if (prevWorker) Object.defineProperty(globalThis, 'Worker', prevWorker);
      else delete globalThis.Worker;
    });
    await searcher.init({ evalCacheEntries: 2048 });
    arenaRunDownloads.clear();
    const deps = {
      BIG_NETS: { bt4: BT4_NET },
      LC0_MODEL_DOWNLOAD_NOT_APPROVED,
      arenaRunDownloads,
      isMemoryConstrainedDevice: () => true,
      arenaCacheEntries: () => 2048,
      peekBigNetSearcher: () => searcher,
      preparePhoneBigNet: () => undefined,
      disposeArenaCpuEngines() {},
      acquireBigNetSearcherAsync: async () => searcher,
      inspectLargeNetDownload: async () => ({ cached: true }),
      requestLargeNetDownloadConsent: async (config, _signal, _warn, _notice, inspect) => {
        prompts.push(await inspect(config));
        return approved ? 'download' : false;
      },
      el: () => ({}),
      showWorkerDeviceWarning() {},
      engines: new Map(),
      arenaBudgetMode: () => 'visits',
      renderBigNetWarning() {},
      recordBt4SearchOutput() {},
      hideDownloadProgress() {},
      clearArenaSearchProgress() {},
    };
    const factory = new Function(
      'deps',
      `const { ${Object.keys(deps)} } = deps;
      ${slice('async function bigNetFor(', 'let runtimeIsolation')}
      ${slice('  const lc0BigNetMove =', '  const centipawnMove =')}
      return lc0BigNetMove;`,
    )(deps);
    const positions = [parseFen(START_FEN)];
    const move = factory('seat-1', { variant: 'bt4', strength: 10 }, 'bt4')(positions, new AbortController().signal);
    if (approved) assert.equal(await move, 'e2e4');
    else await assert.rejects(move, /no longer cached and its download was not approved/);
    assert.deepEqual(prompts, [{ cached: false }]);
    assert.deepEqual(
      inits.map(({ ep, noModelDownload }) => [ep, noModelDownload]),
      [['auto', undefined], ['auto', true], ['wasm', true], ...(approved ? [['wasm', undefined]] : [])],
    );
    assert.deepEqual(
      searches,
      Array.from({ length: approved ? 3 : 2 }, () => ({ positions })),
    );
  });
}
