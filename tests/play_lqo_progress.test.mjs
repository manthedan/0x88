import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import {
  BIG_NETS,
  BT4_NET,
  Bt4WorkerSearcher,
  bigNetBackendUsesWasm,
  bigNetCpuWarning,
  checkBigNetAsset,
  renderBigNetCpuWarning,
} from '../src/lc0/bt4Engine.ts';
import { BIG_NET_LEVELS } from '../src/lc0/play/playSearchSettings.ts';

// Run the real request function with a deferred worker evaluation: no DOM,
// browser, network or model assets. UI writes record what the next mount sees.
const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
const requestSource = source.slice(source.indexOf('async function ctxRequestEngineMove('), source.indexOf('async function ctxVerifySelectedEngineReady('));
const strengthSource = source.slice(source.indexOf('function ctxStrengthFor('), source.indexOf('function ctxRenderLevelOptions('));
const preloadSource = source.slice(source.indexOf('async function ctxPreloadLqoAfterBookMove('), source.indexOf('async function ctxRequestEngineMove('));

const readinessSource = source.slice(source.indexOf('async function ctxVerifySelectedEngineReady('), source.indexOf('const THINKING_SHOW_DELAY_MS'));

for (const variant of ['lqo', 'bt4', 't3']) {
  test(`Play readiness: ${variant === 'lqo' ? 'unlock White after the book, not the net' : `report the ${variant} download`}`, async (t) => {
    const download = Promise.withResolvers();
    const pending = Promise.withResolvers();
    const searcher = {
      loaded: false,
      config: BIG_NETS[variant],
      search: () => {
        pending.resolve();
        return download.promise;
      },
      resetTree() {},
    };
    const option = { family: 'lc0', variant, id: variant };
    const deps = {
      engines: { retireUnusedEngines: async () => {}, acquireBigNet: t.mock.fn(async () => searcher) },
      ctxSelectedEngine: () => option,
      loadLqoOpeningBook: t.mock.fn(async () => ({})),
      LQO_BLACK_MISSING_QUEEN_FEN: 'odds-fen',
      ctxSetEngineNote() {},
      ctxShowDownloadProgress: t.mock.fn(),
      ctxHideDownloadProgress: t.mock.fn(),
      ctxRender() {},
      ctxSetEngineFailure: (_ctx, error) => assert.fail(error),
      lqoSearchOptions: () => ({}),
      PLAY_DRAW_SCORE: 0,
      PLAY_SEARCH_CONTEMPT_LIMIT: 0,
    };
    const verify = new Function(...Object.keys(deps), `${stripTypeScriptTypes(readinessSource)}; return ctxVerifySelectedEngineReady;`)(...Object.values(deps));
    const ctx = {
      board: { turn: 'w' }, humanColor: 'w', startFen: 'odds-fen', moves: [], positions: [],
      opponent: { active: {} }, abort: new AbortController(), enginePreparationGeneration: 0, dom: { buttonEl: () => ({}) },
    };
    const preparing = verify(ctx);
    if (variant === 'lqo') {
      await new Promise(setImmediate);
      assert.equal(deps.loadLqoOpeningBook.mock.callCount(), 1);
      assert.equal(deps.engines.acquireBigNet.mock.callCount(), 0);
    } else {
      await pending.promise;
      assert.equal(searcher.config.streamDownloadProgress, true);
      searcher.onDownloadProgress(10, 100);
      assert.deepEqual(deps.ctxShowDownloadProgress.mock.calls.at(-1).arguments, [ctx, `Lc0 ${searcher.config.name}`, 10, 100]);
      assert.equal(ctx.engineReady, false);
      assert.equal(deps.loadLqoOpeningBook.mock.callCount(), 0);
      download.resolve();
    }
    await preparing;
    assert.equal(ctx.engineReady, true);
    assert.equal(deps.ctxHideDownloadProgress.mock.callCount(), 1);
  });
}

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
    loadLqoOpeningBook: async () => ({ lqoBlackBookMove: () => 'e7e5' }),
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
test('Play sends and displays the loaded WASM budget on the first search despite a successful probe', async (t) => {
  const config = { ...BT4_NET, modelUrl: '/test/play-first-search.onnx' };
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 200 }));
  await checkBigNetAsset(config);
  const warning = { textContent: '', hidden: true };
  let captionText;
  let adapterSupported = false;
  const searchPosted = Promise.withResolvers();
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  globalThis.Worker = class extends EventTarget {
    postMessage(message) {
      const reply = (data) => this.dispatchEvent(new MessageEvent('message', { data: { id: message.id, ...data } }));
      if (message.type === 'init') queueMicrotask(() => reply({ type: 'ready', backend: 'auto->wasm' }));
      if (message.type === 'search') {
        searchPosted.resolve({ ...warning, caption: captionText, visits: message.visits });
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
  const option = { family: 'lc0', variant: 'bt4' };
  const deps = {
    engines: { retireUnusedEngines: async () => {}, acquireBigNet: async () => searcher },
    ctxSelectedEngine: () => option,
    ctxDraftEngine: () => option,
    ctxSelectedLevel: (ctx) => ctx.opponent.active.level,
    peekBigNetSearcher: (key) => (key === config.key ? searcher : null),
    bt4SupportedSync: () => adapterSupported,
    bigNetBackendUsesWasm,
    BIG_NET_LEVELS,
    BIG_NETS,
    ctxIsDisposed: () => false,
    ctxSetEngineNote: () => {},
    ctxShowDownloadProgress: () => {},
    ctxHideDownloadProgress: () => {},
    ctxRender: (ctx) => {
      renderBigNetCpuWarning(warning, [bigNetCpuWarning(config, searcher.backend)]);
      captionText = caption(ctx);
    },
    PLAY_DRAW_SCORE: 0,
    PLAY_SEARCH_CONTEMPT_LIMIT: 0,
  };
  const { request, caption } = new Function(
    ...Object.keys(deps),
    `${stripTypeScriptTypes(strengthSource + requestSource)}; return { request: ctxRequestEngineMove, caption: ctxStrengthCaption };`,
  )(...Object.values(deps));
  const ctx = { opponent: { active: { level: 4 } }, humanColor: 'b', positions: [], moves: [] };
  assert.match(caption(ctx), /^≈ 8 visits per move/);
  adapterSupported = true;
  deps.ctxRender(ctx);
  assert.match(captionText, /^≈ 800 visits per move/);
  assert.equal(warning.hidden, true, 'no CPU warning was predicted before worker init');
  assert.equal(await request(ctx, new AbortController().signal), 'e2e4');
  const duringSearch = await searchPosted.promise;
  assert.equal(duringSearch.visits, 8);
  assert.match(duringSearch.caption, /^≈ 8 visits per move/);
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
        loadLqoOpeningBook: async () => ({ lqoWhiteFirstPolicyBookMove: () => 'e2e4' }),
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
