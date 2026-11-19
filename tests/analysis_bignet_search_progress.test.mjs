import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';
import { productionEnginePort, testAnalysisEngines } from './lib/analysis_engines.mjs';

// Companion to bignet_progress_ownership.test.mjs, for the search step instead
// of init: a cancelled Analysis big-net search settles after
// cancelAnalysisOperation has released the shared model progress slot, so its
// finally must not clear whatever the next run or mount has shown there.
// Found by the round 4 model-hot-spot run (P6, F1). Runs the engine host's
// bigNetLines through the controller's own port.

const tick = () => new Promise(setImmediate);

for (const ending of ['live', 'stopped', 'unmounted']) {
  test(`Analysis big-net search cleanup belongs to its ${ending} run`, async (t) => {
    const workers = [];
    const before = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
    globalThis.Worker = class extends EventTarget {
      messages = [];
      constructor() {
        super();
        workers.push(this);
      }
      terminate() {}
      reply(data) {
        this.dispatchEvent(new MessageEvent('message', { data }));
      }
      postMessage(message) {
        this.messages.push(message);
        if (message.type === 'cancel') queueMicrotask(() => this.reply({ id: message.target, type: 'searchResult', result: { cancelled: true } }));
      }
    };
    t.after(() => (before ? Object.defineProperty(globalThis, 'Worker', before) : delete globalThis.Worker));

    const session = { evalCacheEntries: 0 };
    const searcher = new Bt4WorkerSearcher(T3_NET);
    t.after(() => searcher.dispose());
    const loading = searcher.init(session);
    const init = workers[0].messages.find((message) => message.type === 'init');
    workers[0].reply({ id: init.id, type: 'ready', backend: 'mock-webgpu' });
    await loading;

    const effects = [];
    const mountAbort = new AbortController();
    const engines = testAnalysisEngines({
      port: productionEnginePort({
        mountAbort,
        showLoadingProgressItem: (id, item) => effects.push(['show', id, item]),
        hideLoadingProgressItem: (id) => effects.push(['hide', id]),
        renderRecklessRuntimeInfo: () => {},
      }),
      bigNets: { acquire: async () => searcher, peek: () => searcher, release() {}, disposeNow() {} },
    });

    const controller = new AbortController();
    const pending = engines.bigNetLines('t3', START_FEN, 8, [parseFen(START_FEN)], 1, searcher.servedModelUrl, controller.signal, () => {});
    await tick();
    const search = workers[0].messages.find((message) => message.type === 'search');
    assert.ok(search, 'the real searcher posted the search');

    if (ending === 'stopped') controller.abort();
    if (ending === 'unmounted') mountAbort.abort();
    effects.length = 0; // From here the shared slot may belong to the next run or mount.
    if (ending !== 'stopped') workers[0].reply({ id: search.id, type: 'searchResult', result: { visits: 8, pv: [] } });
    await pending;

    assert.deepEqual(
      effects.map(([kind, id]) => `${kind}:${id}`),
      ending === 'live' ? ['hide:model'] : [],
      'a cancelled search cleared the model progress slot it no longer owns',
    );
  });
}
