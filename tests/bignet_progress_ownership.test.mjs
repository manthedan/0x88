import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';
import { productionEnginePort, testAnalysisEngines } from './lib/analysis_engines.mjs';

function extract(file, first, last) {
  const source = readFileSync(new URL(file, import.meta.url), 'utf8');
  const start = source.indexOf(first);
  const end = source.indexOf(last, start);
  assert.ok(start >= 0 && end > start, 'production function boundaries');
  return stripTypeScriptTypes(source.slice(start, end));
}
const arena = extract('../src/lc0/arenaBrowser.ts', '  const lc0BigNetWarmup =', '  const centipawnWarmup =');

for (const workflow of ['Arena', 'Analysis']) {
  for (const effect of ['download progress', 'cleanup']) {
    for (const cancelled of [true, false]) {
      test(`${workflow} ${effect} belongs to its ${cancelled ? 'cancelled' : 'live'} request`, async (t) => {
        const workers = [];
        const commands = [];
        const before = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
        globalThis.Worker = class extends EventTarget {
          constructor() {
            super();
            workers.push(this);
          }
          terminate() {}
          reply(data) {
            this.dispatchEvent(new MessageEvent('message', { data }));
          }
          postMessage(message) {
            commands.push(message);
            if (message.type === 'search') queueMicrotask(() => this.reply({ id: message.id, result: { visits: 1 } }));
            if (message.type === 'resetSearch') queueMicrotask(() => this.reply({ id: message.id }));
          }
        };
        t.after(() => (before ? Object.defineProperty(globalThis, 'Worker', before) : delete globalThis.Worker));
        const searcher = new Bt4WorkerSearcher(T3_NET);
        t.after(() => searcher.dispose());
        const effects = [];
        const show = (...args) => effects.push(['show', ...args]);
        const hide = () => effects.push(['hide']);
        const bigNetFor = () => ({ config: T3_NET, searcher });
        const controller = new AbortController();
        const pending =
          workflow === 'Arena'
            ? new Function(
                'bigNetFor',
                'warmupPositions',
                'showDownloadProgress',
                'hideDownloadProgress',
                'arenaCacheEntries',
                `${arena}\nreturn lc0BigNetWarmup('t3');`,
              )(
                bigNetFor,
                [parseFen(START_FEN)],
                show,
                hide,
                () => 2048,
              )(controller.signal)
            : testAnalysisEngines({
                // The controller's port: its model progress item is the shared slot.
                port: productionEnginePort({
                  mountAbort: new AbortController(),
                  showLoadingProgressItem: (id, item) => id === 'model' && show(item),
                  hideLoadingProgressItem: (id) => id === 'model' && hide(),
                  renderRecklessRuntimeInfo: () => {},
                }),
                bigNets: { acquire: async () => searcher, peek: () => searcher, release() {}, disposeNow() {} },
              }).initBigNetsForRun(['t3'], controller, () => controller.signal.aborted);
        await new Promise(setImmediate);
        const init = commands.find((message) => message.type === 'init');
        assert.ok(init, 'the real pooled searcher is waiting for model initialization');
        if (cancelled) controller.abort();
        effects.length = 0; // The shared surface may now belong to a new request/mount.
        if (effect === 'download progress') {
          workers[0].reply({ type: 'downloadProgress', loadedBytes: 20, totalBytes: 100 });
          const observed = [...effects];
          workers[0].reply({ id: init.id, backend: 'mock-webgpu' });
          await pending;
          assert.deepEqual(
            observed.map(([kind]) => kind),
            cancelled ? [] : ['show'],
            'old download callback published into the current surface',
          );
        } else {
          workers[0].reply({ id: init.id, backend: 'mock-webgpu' });
          await pending;
          assert.deepEqual(
            effects.map(([kind]) => kind),
            cancelled ? [] : ['hide'],
            'old finally cleared the current surface',
          );
        }
      });
    }
  }
}
