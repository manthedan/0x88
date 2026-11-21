import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';
import { LC0_MODEL_DOWNLOAD_NOT_APPROVED } from '../src/lc0/modelCache.ts';
import { committedPlayOpponent, transitionPlayOpponent } from '../src/lc0/playOpponentState.ts';
import { testAnalysisEngines } from './lib/analysis_engines.mjs';

const source = Object.fromEntries(
  ['analysis', 'arena', 'play'].map((name) => [name, readFileSync(new URL(`../src/lc0/${name}Browser.ts`, import.meta.url), 'utf8')]),
);
function extract(text, first, last, tail = '') {
  const start = text.indexOf(first);
  const end = text.indexOf(last, start);
  assert.ok(start >= 0 && end > start, `production boundaries: ${first}`);
  return stripTypeScriptTypes(text.slice(start, end) + tail);
}

/** Play reaches its DOM and listeners through ctx.dom / ctx.scope; point them at this harness's fakes. */
function playScope(ctx, deps) {
  ctx.dom = { el: deps.el, selectEl: deps.selectEl, buttonEl: deps.buttonEl };
  ctx.scope = { listen: (target, type, fn) => deps.trackListener(ctx, target, type, fn), clearTimeout: (id) => deps.clearTimeout(id) };
  return ctx;
}
const playPickerAbort = extract(source.play, 'function ctxCancelPickerConsent(', 'function ctxCancelEngineTurn(');
const tick = () => new Promise(setImmediate);
async function until(predicate) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await tick();
  }
  assert.fail('expected an async checkpoint');
}

for (const replacement of [
  'cpu-profile',
  'manual',
  'add',
  'remove',
  'lines',
  'runtime',
  'unmount',
  'picker-cpu-profile',
  'picker-add',
  'picker-remove',
  'picker-lines',
  'picker-unmount',
]) {
  test(`Analysis ${replacement} supersedes a pending profile/picker probe without another prompt or stale commit`, async () => {
    const picker = replacement.startsWith('picker-');
    const action = replacement.replace(/^picker-/, '');
    const probe = Promise.withResolvers();
    const mountAbort = new AbortController();
    const prompts = [];
    const messages = [];
    const nodes = new Map();
    const listeners = new Map();
    const el = (id) => {
      if (!nodes.has(id)) nodes.set(id, { addEventListener: (type, handler) => listeners.set(`${id}:${type}`, handler) });
      return nodes.get(id);
    };
    const initial = [
      { family: picker ? 'lc0' : 'sf', variant: picker ? 'small' : 'lite', strength: 8 },
      { family: 'sf', variant: 'full', strength: 12 },
    ];
    const deps = {
      engineRows: initial,
      BIG_NETS,
      mountAbort,
      normalizeDeployEngineRow: (row) => row,
      clampStrengthForRow: (row) => row.strength,
      isLc0BigNetVariant: (key) => key === 'bt4' || key === 't3',
      phoneBigNetBlocked: () => false,
      phoneLargeNetsBlocked: () => false,
      bigNetSelectable: () => true,
      requestAnalysisBigNetConsent: (config, signal) => {
        prompts.push({ key: config.key, signal });
        return probe.promise;
      },
      engineSetupRuntime: () => 'onnx',
      params: new URLSearchParams(),
      selectedLc0Runtime: () => 'onnx',
      cancelAllAnalysisAssetPrefetches() {},
      inputEl: el,
      selectEl: el,
      el,
      storageSet() {},
      LAST_ENGINE_PROFILE_STORAGE_KEY: 'last-profile',
      renderEngineList() {},
      renderEngineProfiles() {},
      profileSummary: () => '',
      currentEngineProfile: () => ({}),
      engineProfileMessage: (message) => messages.push(message),
      lineCache: new Map(),
      completeAnalysisKeys: new Set(),
      deferAnalysisSettingsRefresh() {},
      prefetchAnalysisAssetsForAddedRow() {},
      persistManualEngineRows() {},
      cancelAnalysisAssetPrefetch() {},
      tree: { current: { fen: 'position' } },
      nextEngineFamily: () => 'sf',
      defaultVariant: () => 'full',
      defaultStrength: () => 12,
    };
    const code =
      extract(source.analysis, 'async function profileRowsForUse(', 'function loadEngineProfiles(') +
      extract(source.analysis, 'let profileLoadAbort:', 'function saveCurrentEngineProfile(') +
      extract(source.analysis, "  el('engineList').addEventListener('change'", "  el('engineProfileSelect').addEventListener(") +
      extract(source.analysis, "  el('engineList').addEventListener('click'", '  criticalPanel = mountCriticalMomentsPanel(');
    const api = new Function(
      'deps',
      `let { ${Object.keys(deps)} } = deps; ${code}; return {
      apply: applyEngineProfile, rows: () => engineRows,
      manual() { abortEngineSelectionConsent(); engineRows = [{family:'sf',variant:'full',strength:12}]; }
    };`,
    )(deps);
    const old = picker
      ? listeners.get('engineList:change')({ target: { value: 'bt4', dataset: { i: '0' }, classList: { contains: (name) => name === 'row-var' } } })
      : api.apply({ name: 'old', rows: ['bt4', 't3'].map((variant) => ({ family: 'lc0', variant, strength: 10 })), multiPv: 1 });
    await until(() => prompts.length === 1);
    if (action === 'cpu-profile') await api.apply({ name: 'new', rows: [{ family: 'sf', variant: 'full', strength: 12 }], multiPv: 2 });
    else if (action === 'manual') api.manual();
    else if (action === 'add') listeners.get('addEngine:click')();
    else if (action === 'remove') listeners.get('engineList:click')({ target: { closest: () => ({ dataset: { i: '0' } }) } });
    else if (action === 'lines') {
      el('multiPvInput').value = '7';
      listeners.get('multiPvInput:change')();
    } else if (action === 'runtime') listeners.get('lc0RuntimeSelect:change')();
    else mountAbort.abort();
    assert.equal(prompts[0].signal.aborted, true);
    probe.resolve(true);
    await old;
    assert.deepEqual(
      prompts.map((entry) => entry.key),
      ['bt4'],
      'the old second row must never prompt',
    );
    assert.deepEqual(api.rows(), ['unmount', 'add', 'lines', 'runtime'].includes(action) ? initial : [{ family: 'sf', variant: 'full', strength: 12 }]);
    if (action === 'add') assert.equal(api.rows().length, 3);
    if (action === 'remove') assert.equal(api.rows().length, 1);
    if (action === 'lines') assert.equal(el('multiPvInput').value, '7');
    assert.equal(
      messages.some((message) => message.includes('“old”')),
      false,
    );
  });
}

for (const phase of [
  'checkpoint-writes',
  'compatibility',
  'assets',
  'gpu',
  'model',
  'download',
  'download-cancel',
  'download-superseded',
  'save',
  'save-with-approval',
  'save-superseded',
]) {
  test(`Arena consent revocation cancels startup while awaiting ${phase}`, async () => {
    const gate = Promise.withResolvers();
    let paused = phase === 'checkpoint-writes';
    let revoked = false;
    let admitted = 0;
    let prompts = 0;
    const presented = [];
    const nodes = new Map();
    const el = (id) => {
      if (!nodes.has(id)) nodes.set(id, { textContent: '', value: '' });
      return nodes.get(id);
    };
    const wait = (at, result) => {
      if (phase !== at && !(['save-with-approval', 'save-superseded'].includes(phase) && at === 'save')) return Promise.resolve(result);
      paused = true;
      return gate.promise.then(() => result);
    };
    const rows = [
      { family: 'lc0', variant: 'bt4', strength: 8 },
      { family: 'sf', variant: 'lite', strength: 8 },
    ];
    const deps = {
      running: false,
      startPending: false,
      abort: null,
      mountAbort: new AbortController(),
      abortArenaPickerConsent() {},
      setArenaStartDisabled() {},
      syncArenaPhase() {},
      checkpointWrites: phase === 'checkpoint-writes' ? gate.promise : Promise.resolve(),
      isStaleMount: () => false,
      checkpointUnreadable: false,
      checkpointCompatibility: () => wait('compatibility', 'compatible'),
      normalizeArenaControls: () => ({}),
      ARENA_CONTROL_IDS: [],
      el,
      syncSeatRowsFromDom() {},
      arenaBigNetPairingBlocked: () => false,
      arenaTournamentMode: () => 'match',
      seatRows: rows,
      phoneSafeEngineVariant: (family, variant) => (revoked && family === 'lc0' && variant === 'bt4' ? 'small' : variant),
      phoneLargeNetsBlocked: () => revoked,
      ARENA_BIG_NET_KEYS: ['bt4', 't3'],
      arenaDownloadApprovals: new Set(phase === 'save-with-approval' ? ['bt4'] : []),
      arenaRunDownloads: new Set(),
      requestLargeNetDownloadConsent: (config, signal) => {
        prompts++;
        assert.equal(config, BIG_NETS.bt4);
        assert.equal(signal.aborted, false);
        assert.notEqual(phase, 'save-with-approval', 'picker acceptance must not cause a second startup dialog');
        if (phase === 'download-cancel' || phase === 'download-superseded') {
          paused = true;
          return gate.promise.then(() => false);
        }
        return wait('download', 'download');
      },
      disposeBigNetSearcherNow() {},
      renderSeatSelectors() {},
      persistArenaSeatRows() {},
      buildEngines() {},
      populateSeats() {},
      ensureSelectedRecklessAssetsAvailable: () => wait('assets', true),
      seatEngineId: (index) => String(index),
      engines: new Map([
        ['0', { name: 'BT4' }],
        ['1', { name: 'Stockfish' }],
      ]),
      activeSeatRows: () => rows,
      isLc0BigNetVariant: (key) => key === 'bt4' || key === 't3',
      probeBt4Support: () => wait('gpu', true),
      isMemoryConstrainedDevice: () => true,
      BIG_NETS,
      checkBigNetAsset: () => wait('model', 'present'),
      bigNetSelectable: () => true,
      whileArenaMounted: (fn) => fn,
      arenaGamesPerOpening: () => 1,
      TournamentStandings: class {},
      buildSchedule: () => [{ whiteId: '0', blackId: '1', opening: {} }],
      tournamentPairings: () => [],
      selectedOpenings: () => [],
      selectedSeatsNeedLc0Evaluator: () => false,
      createArenaCheckpoint: () => ({ runId: 'test', revision: 1 }),
      captureCheckpointConfig: () => ({}),
      saveArenaCheckpoint: () => wait('save', undefined),
      checkpointToken: null,
      checkpointProblem: '',
      ArenaCheckpointConflictError: class extends Error {},
      checkpoint: null,
      hideEngineWarmupStatus() {},
      renderCheckpointStatus() {},
      presentSavedCheckpoint: (saved) => presented.push(saved),
      arenaRemainingIndices: () => {
        admitted++;
        throw new Error('cancelled startup entered a live match');
      },
    };
    const code =
      extract(source.arena, 'function clearStartPending()', 'function createIsolatedCpuEngine(') +
      extract(source.arena, 'async function startMatchUnchecked(', 'function exportPgn()') +
      extract(source.arena, 'function revokeArenaPhoneLargeNets()', 'function wireEvents()');
    const api = new Function(
      'deps',
      `let { ${Object.keys(deps)} } = deps; ${code}; return {
      start: startMatch, revoke: revokeArenaPhoneLargeNets,
      state: () => ({running,startPending,abort,checkpoint}), token: () => checkpointToken,
      replace() {
        abort = new AbortController(); startPending = true;
        checkpoint = {runId:'replacement'}; checkpointToken = {runId:'replacement',revision:99};
        return abort;
      }
    };`,
    )(deps);
    const starting = api.start();
    await until(() => paused);
    const controller = api.state().abort;
    assert.ok(controller, 'startup owns a cancellation controller before the first await');
    let replacement = null;
    if (phase.endsWith('-superseded')) {
      controller.abort();
      replacement = api.replace();
    } else if (phase === 'download-cancel') {
      assert.equal(prompts, 1);
      assert.equal(controller.signal.aborted, false);
    } else {
      if (phase === 'save') assert.equal(prompts, 1, 'restored/URL startup asks before checkpoint creation or warmup');
      if (phase === 'save-with-approval') assert.equal(prompts, 0);
      revoked = true;
      api.revoke();
      assert.equal(controller.signal.aborted, true);
    }
    gate.resolve();
    await starting;
    if (replacement) {
      assert.deepEqual(api.state(), { running: false, startPending: true, abort: replacement, checkpoint: { runId: 'replacement' } });
      assert.deepEqual(api.token(), { runId: 'replacement', revision: 99 });
    } else {
      // A save that lands before the cancellation is observed already replaced the
      // stored slot: the page must name that checkpoint by both token and value.
      const saved = phase === 'save' || phase === 'save-with-approval' ? { runId: 'test', revision: 1 } : null;
      assert.deepEqual(api.state(), { running: false, startPending: false, abort: null, checkpoint: saved });
      assert.deepEqual(api.token(), saved);
      // Results, status and PGN export follow the stored run, not the one it replaced.
      assert.deepEqual(presented, saved ? [saved] : []);
    }
    assert.equal(admitted, 0);
    assert.doesNotMatch(el('message').textContent, /failed/);
  });
}

test('Arena Start carries its consent to the load: a cache hit approves no download', async () => {
  // Startup from the Small load through the checkpoint save.
  const from = source.arena.indexOf('  if (selectedSeatsNeedLc0Evaluator() && !lc0Cache) {');
  const to = source.arena.indexOf('  if (startupCancelled()) {\n    if (checkpoint === runCheckpoint', from);
  assert.ok(from >= 0 && to > from, 'production boundaries: Start setup');
  const start = stripTypeScriptTypes(`async function start() {\n${source.arena.slice(from, to)}}`);
  const bigNetFor = extract(source.arena, 'async function bigNetFor(', 'let runtimeIsolation');
  const prompts = [];
  const saves = [];
  let answer = 'cached';
  let smallLoaded = false;
  let loaded = false;
  let configured = true;
  const matchAbort = new AbortController();
  const deps = {
    arenaRunDownloads: new Set(),
    arenaDownloadApprovals: new Set(),
    checkedBigNetVariants: new Set(['bt4']),
    BIG_NETS,
    startupCancelled: () => false,
    isMemoryConstrainedDevice: () => true,
    matchAbort,
    abort: matchAbort,
    mountSignal: new AbortController().signal,
    isStaleMount: () => false,
    selectedSeatsNeedLc0Evaluator: () => true,
    loadSmall: async () => {
      smallLoaded = true;
    },
    requestLargeNetDownloadConsent: async (config) => {
      // Small can evict a cached large net, so the answer must reflect the cache after it loads.
      assert.equal(smallLoaded, true, 'consent is asked after Lc0 Small loads');
      prompts.push(config.key);
      return answer;
    },
    clearStartPending() {},
    resume: null,
    claimArenaCheckpoint: () => assert.fail('not a resume'),
    createArenaCheckpoint: () => ({ runId: 'new', revision: 1 }),
    captureCheckpointConfig: () => ({}),
    participants: [],
    compatibility: 'compatible',
    schedule: [{ whiteId: '0', blackId: '1' }],
    byPid: new Map([
      ['0', { engine: { id: 'sf-a' } }],
      ['1', { engine: { id: 'sf-b' } }],
    ]),
    rowForEngineId: () => ({ family: 'sf', variant: 'lite' }),
    isLc0BigNetVariant: (key) => key === 'bt4' || key === 't3',
    saveArenaCheckpoint: async (saved) => {
      saves.push(saved.runId);
    },
    checkpointToken: { runId: 'old', revision: 3 },
    checkpoint: { runId: 'old' },
    checkpointProblem: '',
    ArenaCheckpointConflictError: class extends Error {},
    peekBigNetSearcher: () => ({ loaded, isInitializedFor: () => loaded && configured }),
    preparePhoneBigNet: () => null,
    arenaCacheEntries: () => 0,
    disposeArenaCpuEngines() {},
    el: () => ({}),
    acquireBigNetSearcherAsync: async () => {
      loaded = true;
      return { loaded, isInitializedFor: () => true, init: async () => 'mock' };
    },
  };
  const api = new Function(
    'deps',
    `let { ${Object.keys(deps)} } = deps;
    let lc0Cache = null;
    async function loadLc0Evaluator() { await deps.loadSmall(); lc0Cache = {}; }
    ${bigNetFor}
    ${start}
    return { start: async () => { lc0Cache = null; await start(); }, bigNetFor, checkpoint: () => [checkpoint, checkpointToken] };`,
  )(deps);
  const signal = new AbortController().signal;

  // Declining keeps the previous tournament: nothing is saved over it.
  answer = false;
  await api.start();
  assert.deepEqual(saves, []);
  assert.deepEqual(api.checkpoint(), [{ runId: 'old' }, { runId: 'old', revision: 3 }]);

  // Cached at Start: the load must ask again, since storage may have evicted it.
  prompts.length = 0;
  answer = 'cached';
  await api.start();
  assert.deepEqual(saves, ['new']);
  assert.deepEqual(prompts, ['bt4']);
  assert.equal(deps.arenaRunDownloads.has('bt4'), false);
  answer = false;
  await assert.rejects(api.bigNetFor('bt4', signal), /no longer cached/);
  assert.equal(loaded, false, 'a declined download never reaches the loader');
  answer = 'cached';
  await api.bigNetFor('bt4', signal);
  assert.equal(loaded, true);
  assert.equal(prompts.length, 3);
  await api.bigNetFor('bt4', signal);
  assert.equal(prompts.length, 3, 'a reusable warm session needs no check');
  // A warm session configured differently is retired, so its replacement may download.
  configured = false;
  answer = 'download';
  await api.bigNetFor('bt4', signal);
  assert.equal(prompts.length, 4);
  loaded = false;
  await api.bigNetFor('bt4', signal);
  assert.equal(prompts.length, 4, 'an approved download is not asked again within the run');

  // A picker download approval covers the run's loads.
  prompts.length = 0;
  deps.arenaDownloadApprovals.add('bt4');
  await api.start();
  loaded = false;
  await api.bigNetFor('bt4', signal);
  assert.deepEqual(prompts, []);
});

for (const answer of [false, 'download']) {
  test(`Arena cache-hit consent loads cache-only and asks before downloading an evicted net (accepted=${!!answer})`, async () => {
    const inits = [];
    const prompts = [];
    const searcher = {
      loaded: false,
      isInitializedFor: () => false,
      async init(options) {
        inits.push(options);
        // The entry was evicted during the reclaim wait (or its body is unreadable).
        throw new Error(`${LC0_MODEL_DOWNLOAD_NOT_APPROVED}: bt4.onnx`);
      },
    };
    const deps = {
      BIG_NETS,
      LC0_MODEL_DOWNLOAD_NOT_APPROVED,
      isMemoryConstrainedDevice: () => true,
      arenaRunDownloads: new Set(),
      arenaCacheEntries: () => 0,
      peekBigNetSearcher: () => null,
      inspectLargeNetDownload: async () => ({ url: 'bt4.onnx', cached: true, downloadBytes: 5 }),
      requestLargeNetDownloadConsent: async (config, _signal, _desktopWarning, _notice, inspect) => {
        prompts.push(inspect ? await inspect(config) : 'start');
        return inspect ? answer : 'cached';
      },
      preparePhoneBigNet: () => undefined,
      disposeArenaCpuEngines() {},
      disposeBigNetSearcherNow() {},
      el: () => ({}),
      acquireBigNetSearcherAsync: async () => searcher,
    };
    const bigNetFor = new Function(
      'deps',
      `const { ${Object.keys(deps)} } = deps;\n${extract(source.arena, 'async function bigNetFor(', 'let runtimeIsolation')}\nreturn bigNetFor;`,
    )(deps);
    const loading = bigNetFor('bt4', new AbortController().signal);
    if (answer) assert.equal((await loading).searcher, searcher);
    else await assert.rejects(loading, /no longer cached and its download was not approved/);
    assert.deepEqual(inits, [{ evalCacheEntries: 0, noDownload: true }], 'the cache-hit approval reaches the loader');
    assert.deepEqual(prompts, ['start', { url: 'bt4.onnx', cached: false, downloadBytes: 5 }], 'the second prompt is a download prompt');
    assert.equal(deps.arenaRunDownloads.has('bt4'), !!answer, 'only an accepted prompt approves the download');
  });
}

for (const outcome of ['declined', 'accepted', 'download-approved', 'resume']) {
  test(`Arena resolves the first game's cache-only big-net load before replacing the checkpoint (${outcome})`, async () => {
    const from = source.arena.indexOf('  if (selectedSeatsNeedLc0Evaluator() && !lc0Cache) {');
    const to = source.arena.indexOf('  if (startupCancelled()) {\n    if (checkpoint === runCheckpoint', from);
    assert.ok(from >= 0 && to > from, 'production boundaries: Start setup');
    const start = stripTypeScriptTypes(`async function start() {\n${source.arena.slice(from, to)}}`);
    const order = [];
    const node = { textContent: '' };
    const matchAbort = new AbortController();
    const resume = outcome === 'resume' ? { runId: 'old' } : null;
    const deps = {
      arenaRunDownloads: new Set(),
      arenaDownloadApprovals: new Set(),
      checkedBigNetVariants: new Set(['bt4']),
      BIG_NETS,
      startupCancelled: () => false,
      isMemoryConstrainedDevice: () => true,
      matchAbort,
      abort: matchAbort,
      mountSignal: new AbortController().signal,
      isStaleMount: () => false,
      selectedSeatsNeedLc0Evaluator: () => false,
      lc0Cache: {},
      requestLargeNetDownloadConsent: async () => (outcome === 'download-approved' ? 'download' : 'cached'),
      clearStartPending: () => order.push('clear'),
      resume,
      claimArenaCheckpoint: (saved) => saved,
      createArenaCheckpoint: () => ({ runId: 'new', revision: 1 }),
      captureCheckpointConfig: () => ({}),
      participants: [],
      compatibility: 'compatible',
      schedule: [{ whiteId: '0', blackId: '1' }],
      byPid: new Map([
        ['0', { engine: { id: 'lc0-bt4' } }],
        ['1', { engine: { id: 'sf-full' } }],
      ]),
      rowForEngineId: (id) => (id === 'lc0-bt4' ? { family: 'lc0', variant: 'bt4' } : { family: 'sf', variant: 'full' }),
      isLc0BigNetVariant: (key) => key === 'bt4' || key === 't3',
      bigNetFor: async (variant, signal) => {
        order.push(`load ${variant}`);
        assert.equal(signal, matchAbort.signal);
        assert.equal(node.hidden, false, 'setup Cancel stays available while the load is pending');
        // The cache-only load found the net evicted and the re-asked download was declined.
        if (outcome === 'declined') throw new Error('Lc0 BT4 is no longer cached and its download was not approved.');
      },
      saveArenaCheckpoint: async (saved) => {
        order.push(`save ${saved.runId}`);
      },
      checkpointToken: { runId: 'old', revision: 3 },
      checkpoint: { runId: 'old' },
      checkpointProblem: '',
      ArenaCheckpointConflictError: class extends Error {},
      el: () => node,
    };
    const api = new Function(
      'deps',
      `let { ${Object.keys(deps)} } = deps;
      ${start}
      return { start, checkpoint: () => [checkpoint, checkpointToken] };`,
    )(deps);
    await api.start();
    const expected = {
      declined: ['load bt4', 'clear'],
      accepted: ['load bt4', 'save new'],
      'download-approved': ['save new'],
      resume: ['save old'],
    }[outcome];
    assert.deepEqual(order, expected);
    if (expected.includes('load bt4')) assert.equal(node.hidden, true, 'setup Cancel hides once the load settles');
    if (outcome === 'declined') {
      assert.deepEqual(api.checkpoint(), [{ runId: 'old' }, { runId: 'old', revision: 3 }], 'the previous tournament stays saved');
      assert.match(node.textContent, /Engine preparation failed: .*no longer cached/);
    }
  });
}

function analysisHostWith(searcher) {
  return testAnalysisEngines({ bigNets: { acquire: async () => searcher, peek: () => searcher, release() {}, disposeNow() {} } });
}

function evictedSearcher(inits) {
  let initialized = false;
  return {
    isInitializedFor: () => initialized,
    async init(options) {
      inits.push(options);
      if (options.noDownload) throw new Error(`${LC0_MODEL_DOWNLOAD_NOT_APPROVED}: net.onnx`);
      initialized = true;
    },
  };
}

for (const answer of [false, 'aborted', 'download']) {
  test(`Analysis loads a cache-hit approval cache-only and asks before downloading an evicted net (consent=${answer})`, async () => {
    const inits = [];
    const prompts = [];
    const engines = analysisHostWith(evictedSearcher(inits));
    engines.recordBigNetConsent('t3', 'cached');
    const signal = new AbortController().signal;
    const prompt = {
      signal,
      current: () => true,
      ask: async (config, promptSignal) => {
        prompts.push(config.key);
        assert.equal(promptSignal, signal);
        return answer;
      },
    };
    const errors = await engines.initBigNetsForRun(['t3'], new AbortController(), () => false, prompt);
    assert.deepEqual(prompts, ['t3'], 'the evicted net is a download prompt, asked once');
    if (answer === 'download') {
      assert.deepEqual(inits, [{ evalCacheEntries: 0, noDownload: true }, { evalCacheEntries: 0 }]);
      assert.equal(errors.size, 0);
      assert.equal((await engines.bigNetFor('t3')).cacheOnly, false, 'an accepted download covers later loads');
    } else {
      assert.deepEqual(inits, [{ evalCacheEntries: 0, noDownload: true }], 'a download without approval never reaches the loader');
      assert.match(errors.get('t3').message, /no longer cached and its download was not approved/);
      assert.equal((await engines.bigNetFor('t3')).cacheOnly, true);
      const again = await engines.initBigNetsForRun(['t3'], new AbortController(), () => false, prompt);
      assert.deepEqual(prompts, answer === false ? ['t3'] : ['t3', 't3'], 'only an explicit decline suppresses the next prompt');
      assert.match(again.get('t3').message, /not approved/);
    }
  });
}

test('a superseded Analysis run never asks to download a cache-only net', async () => {
  const prompts = [];
  const inits = [];
  const engines = analysisHostWith({
    isInitializedFor: () => false,
    async init(options) {
      inits.push(options);
      throw new Error(`${LC0_MODEL_DOWNLOAD_NOT_APPROVED}: bt4.onnx`);
    },
  });
  engines.recordBigNetConsent('bt4', 'cached');
  const ask = async (config) => {
    prompts.push(config.key);
    return 'download';
  };
  const errors = await engines.initBigNetsForRun(['bt4'], new AbortController(), () => false, {
    signal: new AbortController().signal,
    current: () => false,
    ask,
  });
  assert.deepEqual(prompts, []);
  assert.deepEqual(inits, [{ evalCacheEntries: 0, noDownload: true }]);
  assert.equal(errors.size, 0, 'the next run reports on the current rows');
});

test('Analysis records only cache-hit row approvals as cache-only', async () => {
  const inits = [];
  const prompts = [];
  const engines = analysisHostWith(evictedSearcher(inits));
  let answer;
  const ask = new Function(
    'requestLargeNetDownloadConsent',
    'engines',
    `${extract(source.analysis, 'async function requestAnalysisBigNetConsent(', 'async function refreshBigNetAvailability(')}\nreturn requestAnalysisBigNetConsent;`,
  )(async () => answer, engines);
  const cacheOnly = async () => (await engines.bigNetFor('bt4')).cacheOnly;
  const prompt = {
    signal: new AbortController().signal,
    current: () => true,
    ask: async (config) => {
      prompts.push(config.key);
      return false;
    },
  };
  const signal = new AbortController().signal;
  answer = 'cached';
  assert.equal(await ask(BIG_NETS.bt4, signal), true);
  assert.equal(await cacheOnly(), true);
  answer = false;
  assert.equal(await ask(BIG_NETS.bt4, signal), false);
  assert.equal(await cacheOnly(), true, 'a decline leaves the row unchanged');
  await engines.initBigNetsForRun(['bt4'], new AbortController(), () => false, prompt);
  await engines.initBigNetsForRun(['bt4'], new AbortController(), () => false, prompt);
  assert.deepEqual(prompts, ['bt4']);
  answer = 'download';
  assert.equal(await ask(BIG_NETS.bt4, signal), true);
  assert.equal(await cacheOnly(), false, 'a download approval lifts the cache-only load');
  answer = 'cached';
  await ask(BIG_NETS.bt4, signal);
  await engines.initBigNetsForRun(['bt4'], new AbortController(), () => false, prompt);
  assert.deepEqual(prompts, ['bt4', 'bt4'], 'choosing the net again retries a declined download');
});

test('Arena Stop landing as the pool hands over the searcher skips the cache-only init', async () => {
  const controller = new AbortController();
  let inits = 0;
  const searcher = {
    loaded: false,
    isInitializedFor: () => false,
    init: async () => {
      inits++;
    },
  };
  const deps = {
    BIG_NETS,
    LC0_MODEL_DOWNLOAD_NOT_APPROVED,
    isMemoryConstrainedDevice: () => true,
    arenaRunDownloads: new Set(),
    arenaCacheEntries: () => 0,
    peekBigNetSearcher: () => null,
    inspectLargeNetDownload: async () => assert.fail('no prompt'),
    requestLargeNetDownloadConsent: async () => 'cached',
    preparePhoneBigNet: () => undefined,
    disposeArenaCpuEngines() {},
    disposeBigNetSearcherNow: () => assert.fail('nothing to retire: no listener was registered'),
    el: () => ({}),
    acquireBigNetSearcherAsync: async () => {
      controller.abort(); // Stop between the pool's resolution and bigNetFor resuming
      return searcher;
    },
  };
  const bigNetFor = new Function(
    'deps',
    `const { ${Object.keys(deps)} } = deps;\n${extract(source.arena, 'async function bigNetFor(', 'let runtimeIsolation')}\nreturn bigNetFor;`,
  )(deps);
  await assert.rejects(bigNetFor('bt4', controller.signal), { name: 'AbortError' });
  assert.equal(inits, 0, 'a stopped load must not initialize');
});

test('a saved Arena run with no games shows its own opening, not the replaced run', () => {
  const calls = [];
  const opening = { name: 'new opening' };
  const charts = { hidden: false };
  const deps = {
    showCheckpointResults() {},
    arenaRemainingIndices: () => [0],
    arenaMatchState: 'final',
    el: (id) => (id === 'chartsPanel' ? charts : {}),
    inputEl: () => ({}),
    finishedTrails: [],
    // A finished live tournament: its last game is on the board, not in review.
    liveTrail: { label: 'old game' },
    reviewing: false,
    exitReview: () => calls.push('exitReview'),
    resetGameCharts: () => calls.push('resetGameCharts'),
    resultsSchedule: [{ opening }],
    setOpeningPreview: (shown) => calls.push(shown),
  };
  const present = new Function(
    'deps',
    `let { ${Object.keys(deps)} } = deps; ${extract(source.arena, 'function presentSavedCheckpoint(', 'function showCheckpointResults(')}; return presentSavedCheckpoint;`,
  )(deps);
  present({ completed: [] });
  // setOpeningPreview resets the board, last move and side labels.
  assert.deepEqual(calls, ['exitReview', 'resetGameCharts', opening]);
  assert.equal(charts.hidden, true, "the replaced run's charts are not shown beside the new scoreboard");
});

const restoreGuard =
  extract(source.play, 'function ctxShowEngineRetry(', 'function ctxSetEngineFailure(') +
  extract(source.play, 'async function ctxCheckRestoredDownloadConsent(', 'function ctxPrepareOrStartEngine(');
const readinessEntry = extract(source.play, 'async function ctxVerifySelectedEngineReady(', '  const generation =', 'await retirement;\nadmit();\n}');
const moveEntry = extract(source.play, 'async function ctxEngineTurn(', '  ctxSetEngineThinking(ctx, true);', 'await admit();\n}');
for (const key of Object.keys(BIG_NETS)) {
  for (const mode of ['readiness', 'engine-first']) {
    for (const accepted of [false, true]) {
      test(`restored ${key} gates ${mode} admission on cache-aware approval (accepted=${accepted})`, async () => {
        const permission = Promise.withResolvers();
        let prompts = 0;
        let admissions = 0;
        const retry = { dataset: {}, hidden: true };
        const ctx = {
          abort: new AbortController(),
          consentAbort: new AbortController(),
          disposed: false,
          restoredDownloadEngineId: key,
          restoredDownloadConsent: null,
          opponent: { active: { engineId: key } },
          board: { turn: mode === 'readiness' ? 'w' : 'b' },
          humanColor: 'w',
          moves: ['saved-move'],
          gameOver: null,
        };
        const deps = {
          BIG_NETS,
          ctxSelectedEngine: () => ({ id: key, family: 'lc0', variant: key }),
          requestLargeNetConsent: (config, signal, desktopWarning) => {
            prompts++;
            assert.equal(config, BIG_NETS[key]);
            assert.equal(signal.aborted, false);
            assert.equal(desktopWarning, false);
            return permission.promise;
          },
          ctxSetEngineNote() {},
          releaseUnusedBigNetSearchers() {},
          engines: {
            retireUnusedEngines: async () => {
              assert.equal(ctx.restoredDownloadEngineId, null, 'retirement starts only after approval');
            },
          },
          buttonEl: () => retry,
          ctxRender() {},
          ctxSetEngineFailure: (_ctx, error) => assert.fail(String(error)),
          admit: () => admissions++,
        };
        playScope(ctx, deps);
        const entry = mode === 'readiness' ? readinessEntry : moveEntry;
        const api = new Function(
          'deps',
          `const { ${Object.keys(deps)} } = deps; ${restoreGuard}; ${entry}; return ${mode === 'readiness' ? 'ctxVerifySelectedEngineReady' : 'ctxEngineTurn'};`,
        )(deps);
        const pending = api(ctx);
        assert.equal(prompts, 1);
        assert.equal(admissions, 0);
        await api(ctx);
        assert.equal(prompts, 1, 'a second readiness/move request does not duplicate the prompt or admission');
        permission.resolve(accepted);
        await pending;
        assert.equal(admissions, accepted ? 1 : 0);
        assert.equal(ctx.restoredDownloadEngineId, accepted ? null : key);
        assert.equal(retry.hidden, accepted);
        assert.equal(ctx.opponent.active.engineId, key, 'Cancel keeps the restored opponent and saved position');
      });
    }
  }
}

test('engine-first Play retires the previous large net and awaits its cleanup before acquiring the next', async () => {
  const retirement = Promise.withResolvers();
  const events = [];
  const pool = new BigNetSessionPool(
    (config) => {
      events.push(`create:${config.key}`);
      return {
        dispose() {
          events.push(`dispose:${config.key}`);
          return retirement.promise;
        },
      };
    },
    undefined,
    undefined,
    () => true,
  );
  pool.acquire('bt4');
  const deps = {
    ctxSelectedEngine: () => ({ id: 'lc0-t3', family: 'lc0', variant: 't3' }),
    releaseUnusedBigNetSearchers: (keys, ttl) => pool.releaseUnused(keys, ttl),
    admit: () => pool.acquireAsync('t3'),
  };
  const entry = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${moveEntry}; return ctxEngineTurn;`)(deps);
  const pending = entry({
    abort: new AbortController(),
    board: { turn: 'b' },
    humanColor: 'w',
    disposed: false,
    gameOver: null,
    restoredDownloadConsent: null,
  });
  await tick();
  assert.deepEqual(events, ['create:bt4', 'dispose:bt4']);
  assert.equal(pool.has('bt4'), false);
  assert.equal(pool.has('t3'), false, 'replacement construction is still behind graceful retirement');
  retirement.resolve();
  await pending;
  assert.deepEqual(events, ['create:bt4', 'dispose:bt4', 'create:t3']);
  pool.disposeAllNow();
});

for (const action of ['dismiss', 'confirm']) {
  test(`Queen Odds draft approval does not transfer to a live restored BT4 gate (${action})`, async () => {
    const active = { engineId: 'lc0-bt4', color: 'white', level: 3 };
    const draft = { ...active, engineId: 'leela-queen-odds' };
    const ctx = {
      opponent: committedPlayOpponent(active),
      lastEngineId: active.engineId,
      restoredDownloadEngineId: active.engineId,
      restoredDownloadConsent: null,
      abort: new AbortController(),
      consentAbort: new AbortController(),
      disposed: false,
      engineThinking: false,
      gameOver: null,
    };
    const node = { value: draft.engineId };
    const prompts = [];
    let selectedHandler;
    const deps = {
      ctx,
      BIG_NETS,
      committedPlayOpponent,
      transitionPlayOpponent,
      trackListener: (_ctx, _node, _event, handler) => {
        selectedHandler = handler;
      },
      selectEl: () => node,
      ctxDraftEngine: () => ({ id: draft.engineId, family: 'lc0', variant: 'lqo' }),
      ctxSelectedEngine: () =>
        ctx.opponent.active.engineId === active.engineId
          ? { id: active.engineId, family: 'lc0', variant: 'bt4' }
          : { id: draft.engineId, family: 'lc0', variant: 'lqo' },
      requestLargeNetConsent: async (config) => {
        prompts.push(config.key);
        return config.key === 'lqo';
      },
      ctxEngineOptionState: () => ({ disabled: false }),
      ctxRefreshEngineOptions() {},
      ctxRenderLevelOptions() {},
      ctxGameIsFresh: () => false,
      ctxCurrentPlayPreferences: () => draft,
      ctxRestoreOpponentControls() {},
      ctxPersistPlayPreferences() {},
      ctxRender() {},
      ctxNewGame() {},
      ctxSetEngineNote() {},
      buttonEl: () => ({ dataset: {} }),
      ctxSetEngineFailure: (_ctx, error) => assert.fail(String(error)),
      engines: { opponentCommitted() {} },
    };
    playScope(ctx, deps);
    const code =
      playPickerAbort +
      extract(source.play, 'function ctxSetOpponent(', 'function ctxRestoreOpponentControls(') +
      extract(source.play, 'function ctxMaybeQueueRestart(', 'function ctxTuneOpponent(') +
      extract(
        source.play,
        "  ctx.scope.listen(ctx.dom.selectEl('engineSelect'), 'change', async () => {",
        "  ctx.scope.listen(ctx.dom.selectEl('levelSelect')",
      ) +
      restoreGuard;
    const api = new Function(
      'deps',
      `const { ${Object.keys(deps)} } = deps; ${code}; return {dismiss:ctxDismissRestart,confirm:ctxConfirmRestart,check:ctxCheckRestoredDownloadConsent};`,
    )(deps);
    await selectedHandler();
    assert.equal(ctx.opponent.active.engineId, active.engineId);
    assert.equal(ctx.opponent.pending.engineId, draft.engineId);
    assert.equal(ctx.restoredDownloadEngineId, active.engineId, 'approving a draft leaves the live model gate intact');
    api[action](ctx);
    if (action === 'dismiss') {
      assert.equal(ctx.opponent.active.engineId, active.engineId);
      assert.equal(await api.check(ctx), false, 'Retry still asks for BT4 download approval');
      assert.deepEqual(prompts, ['lqo', 'bt4']);
    } else {
      assert.equal(ctx.opponent.active.engineId, draft.engineId);
      assert.equal(ctx.restoredDownloadEngineId, null, 'committing the replacement retires the old model gate');
      assert.equal(await api.check(ctx), true);
      assert.deepEqual(prompts, ['lqo']);
    }
  });
}

for (const [accepted, stale] of [
  [false, false],
  [false, true],
  [true, false],
  [true, true],
]) {
  test(`declined/unavailable picker selection resumes only its current reverted game (accepted=${accepted}, stale=${stale})`, async () => {
    const ctx = {
      pendingPreferredEngineId: 'leela-queen-odds',
      pendingPreferredEngineTimer: 42,
      lastEngineId: 'maia3',
      abort: new AbortController(),
      consentAbort: new AbortController(),
      disposed: false,
      engineThinking: false,
      preparingEngine: false,
      gameOver: null,
    };
    const node = { value: 'lc0-bt4' };
    let handler;
    let preparations = 0;
    let refreshes = 0;
    let note = 'Checking your saved engine assets…';
    const cleared = [];
    const deps = {
      ctx,
      BIG_NETS,
      trackListener: (_ctx, _node, _event, listener) => {
        handler = listener;
      },
      selectEl: () => node,
      clearTimeout: (timer) => cleared.push(timer),
      ctxDraftEngine: () => ({ id: 'lc0-bt4', family: 'lc0', variant: 'bt4' }),
      requestLargeNetConsent: async () => {
        if (stale) ctx.abort.abort();
        return accepted;
      },
      ctxEngineOptionState: () => ({ disabled: true }),
      ctxRefreshEngineOptions: () => refreshes++,
      ctxSetEngineNote: (_ctx, text) => {
        note = text;
      },
      ctxPrepareOrStartEngine: () => preparations++,
    };
    playScope(ctx, deps);
    const code =
      playPickerAbort +
      extract(
        source.play,
        "  ctx.scope.listen(ctx.dom.selectEl('engineSelect'), 'change', async () => {",
        "  ctx.scope.listen(ctx.dom.selectEl('levelSelect')",
      );
    new Function('deps', `const { ${Object.keys(deps)} } = deps; ${code};`)(deps);
    await handler();
    assert.equal(node.value, 'maia3', 'Cancel keeps the prior selector value');
    assert.equal(ctx.pendingPreferredEngineId, null, 'late saved asset callbacks cannot change opponents');
    assert.deepEqual(cleared, [42]);
    assert.equal(preparations, stale ? 0 : 1);
    assert.equal(refreshes, accepted && !stale ? 1 : 0);
    if (!stale) assert.equal(note, '', 'checking text cannot strand the unprepared fallback');
  });
}

test('committing an asset-probed saved opponent preserves that destination model download gate', () => {
  const code = extract(source.play, 'function ctxSetOpponent(', 'function ctxRestoreOpponentControls(');
  const setOpponent = new Function('engines', `${code}; return ctxSetOpponent;`)({ opponentCommitted() {} });
  const ctx = { opponent: committedPlayOpponent({ engineId: 'maia3' }), restoredDownloadEngineId: 'leela-queen-odds' };
  setOpponent(ctx, committedPlayOpponent({ engineId: 'leela-queen-odds' }));
  assert.equal(ctx.restoredDownloadEngineId, 'leela-queen-odds');
});

test('cancelled Play restore cannot admit an engine or overwrite a replacement consent owner', async () => {
  const permission = Promise.withResolvers();
  let writes = 0;
  const ctx = {
    abort: new AbortController(),
    consentAbort: new AbortController(),
    disposed: false,
    restoredDownloadEngineId: 'lqo',
    restoredDownloadConsent: null,
    opponent: { active: { engineId: 'lqo' } },
  };
  const deps = {
    BIG_NETS,
    ctxSelectedEngine: () => ({ id: 'lqo', variant: 'lqo' }),
    requestLargeNetConsent: () => permission.promise,
    ctxSetEngineNote: () => writes++,
    buttonEl: () => {
      writes++;
      return {};
    },
    ctxRender: () => writes++,
    ctxSetEngineFailure: () => writes++,
  };
  playScope(ctx, deps);
  const guard = new Function('deps', `const { ${Object.keys(deps)} } = deps; ${restoreGuard}; return ctxCheckRestoredDownloadConsent;`)(deps);
  const pending = guard(ctx);
  ctx.abort.abort();
  const replacement = Promise.resolve(true);
  ctx.restoredDownloadConsent = replacement;
  permission.resolve(true);
  assert.equal(await pending, false);
  assert.equal(writes, 0);
  assert.equal(ctx.restoredDownloadConsent, replacement);
});
