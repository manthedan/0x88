import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { BT4_NET, Bt4WorkerSearcher, bigNetCpuWarning, checkBigNetAsset, renderBigNetCpuWarning } from '../src/lc0/bt4Engine.ts';

// Run the real request function with a deferred worker evaluation: no DOM,
// browser, network or model assets. UI writes record what the next mount sees.
const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
const requestSource = source.slice(source.indexOf('async function ctxRequestEngineMove('), source.indexOf('async function ctxVerifySelectedEngineReady('));
const preloadSource = source.slice(source.indexOf('async function ctxPreloadLqoAfterBookMove('), source.indexOf('async function ctxRequestEngineMove('));

test('a phone plays the known LQO book reply before its background reclaim wait resolves', async () => {
  const reclaim = Promise.withResolvers();
  let starts = 0;
  const acquire = Promise.withResolvers();
  const ctx = { opponent: { active: { level: 0 } }, humanColor: 'w', moves: ['e2e4'] };
  const deps = {
    engines: {
      retireUnusedEngines: async () => {},
      acquireBigNet: () => {
        acquire.resolve();
        return reclaim.promise;
      },
    },
    ctxSelectedEngine: () => ({ family: 'lc0', variant: 'lqo' }),
    ctxStrengthFor: () => 1,
    isMemoryConstrainedDevice: () => true,
    peekBigNetSearcher: () => null,
    ctxSetEngineNote: () => {},
    ctxShowDownloadProgress: () => {},
    ctxHideDownloadProgress: () => {},
    ctxIsDisposed: () => false,
    moveToUci: (move) => move,
    lqoBlackBookMove: () => 'e7e5',
  };
  const request = new Function(...Object.keys(deps), `${stripTypeScriptTypes(preloadSource + requestSource)}; return ctxRequestEngineMove;`)(
    ...Object.values(deps),
  );
  const moved = request(ctx, new AbortController().signal);
  await acquire.promise;
  assert.equal(await Promise.race([moved, new Promise(setImmediate)]), 'e7e5');
  const searcher = {
    loaded: false,
    config: { name: 'LQO' },
    init: async () => {
      starts++;
    },
  };
  reclaim.resolve(searcher);
  await new Promise(setImmediate);
  assert.equal(starts, 1);
});
test('Play shows the CPU warning before posting the engine’s first search', async (t) => {
  const config = { ...BT4_NET, modelUrl: '/test/play-first-search.onnx' };
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 200 }));
  await checkBigNetAsset(config);
  const warning = { textContent: '', hidden: true };
  const searchPosted = Promise.withResolvers();
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  globalThis.Worker = class extends EventTarget {
    postMessage(message) {
      const reply = (data) => this.dispatchEvent(new MessageEvent('message', { data: { id: message.id, ...data } }));
      if (message.type === 'init') queueMicrotask(() => reply({ type: 'ready', backend: 'auto->wasm' }));
      if (message.type === 'search') {
        searchPosted.resolve({ ...warning });
        queueMicrotask(() => reply({ type: 'searchResult', result: { move: 'e2e4' } }));
      }
      if (message.type === 'dispose') reply({ type: 'disposed' });
    }
    terminate() {}
  };
  const searcher = new Bt4WorkerSearcher(config);
  t.after(async () => {
    await searcher.dispose();
    if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
    else delete globalThis.Worker;
  });
  const deps = {
    engines: { retireUnusedEngines: async () => {}, acquireBigNet: async () => searcher },
    ctxSelectedEngine: () => ({ family: 'lc0', variant: 'bt4' }),
    ctxStrengthFor: () => 8,
    ctxIsDisposed: () => false,
    ctxSetEngineNote: () => {},
    ctxShowDownloadProgress: () => {},
    ctxHideDownloadProgress: () => {},
    ctxRender: () => renderBigNetCpuWarning(warning, [bigNetCpuWarning(config, searcher.backend)]),
    PLAY_DRAW_SCORE: 0,
    PLAY_SEARCH_CONTEMPT_LIMIT: 0,
  };
  const request = new Function(...Object.keys(deps), `${stripTypeScriptTypes(requestSource)}; return ctxRequestEngineMove;`)(...Object.values(deps));
  const ctx = { opponent: { active: { level: 0 } }, humanColor: 'b', positions: [], moves: [] };
  deps.ctxRender();
  assert.equal(warning.hidden, true, 'no CPU warning was predicted before worker init');
  assert.equal(await request(ctx, new AbortController().signal), 'e2e4');
  const duringSearch = await searchPosted.promise;
  assert.equal(duringSearch.hidden, false);
  assert.match(duringSearch.textContent, /CPU fallback/);
});

for (const lifecycle of ['current', 'cancelled', 'disposed']) {
  for (const rejected of [false, true]) {
    test(`LQO policy progress belongs to its live turn (${lifecycle}, rejected=${rejected})`, async () => {
      const evaluation = Promise.withResolvers();
      const entered = Promise.withResolvers();
      const writes = [];
      const searcher = {
        loaded: false,
        config: { name: 'LQO', approxMb: 100 },
        evaluate: () => {
          entered.resolve();
          return evaluation.promise;
        },
      };
      const deps = {
        engines: { retireUnusedEngines: async () => {}, acquireBigNet: async () => searcher },
        ctxSelectedEngine: () => ({ family: 'lc0', variant: 'lqo' }),
        ctxStrengthFor: () => 1,
        ctxSetEngineNote: (...args) => writes.push(['note', ...args]),
        ctxShowDownloadProgress: (...args) => writes.push(['show', ...args]),
        ctxHideDownloadProgress: () => writes.push(['hide']),
        ctxIsDisposed: (ctx) => ctx.disposed,
        moveToUci: (move) => move,
        lqoWhiteFirstPolicyBookMove: () => 'e2e4',
      };
      const request = new Function(...Object.keys(deps), `${stripTypeScriptTypes(requestSource)}; return ctxRequestEngineMove;`)(...Object.values(deps));
      const abort = new AbortController();
      const ctx = { opponent: { active: { level: 0 } }, humanColor: 'b', moves: [], positions: [], disposed: false };
      const running = request(ctx, abort.signal);
      await entered.promise;
      searcher.onDownloadProgress(10, 100);
      assert.ok(
        writes.some(([kind]) => kind === 'show'),
        'current load still reports progress',
      );
      if (lifecycle !== 'current') abort.abort();
      ctx.disposed = lifecycle === 'disposed';
      writes.length = 0; // After cancellation, the replacement owns the shared DOM.
      searcher.onDownloadProgress(20, 100);
      if (rejected) evaluation.reject(new Error('retired worker disposed'));
      else evaluation.resolve({ legalPriors: [] });
      if (rejected) await assert.rejects(running, /retired worker/);
      else assert.equal(await running, lifecycle === 'current' ? 'e2e4' : null);
      if (lifecycle === 'current') {
        assert.deepEqual(
          writes.map(([kind]) => kind),
          rejected ? ['show', 'hide'] : ['show', 'hide', 'note'],
        );
      } else {
        assert.deepEqual(writes, [], "late progress and finally must not show/hide the new game's progress");
      }
    });
  }
}
