import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';
import { committedPlayOpponent, transitionPlayOpponent } from '../src/lc0/playOpponentState.ts';

const source = Object.fromEntries(
  ['analysis', 'arena', 'play'].map((name) => [name, readFileSync(new URL(`../src/lc0/${name}Browser.ts`, import.meta.url), 'utf8')]),
);
function extract(text, first, last, tail = '') {
  const start = text.indexOf(first);
  const end = text.indexOf(last, start);
  assert.ok(start >= 0 && end > start, `production boundaries: ${first}`);
  return stripTypeScriptTypes(text.slice(start, end) + tail);
}
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
      bigNetSelectableSync: () => true,
      requestLargeNetConsent: (config, signal) => {
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
      requestLargeNetConsent: (config, signal) => {
        prompts++;
        assert.equal(config, BIG_NETS.bt4);
        assert.equal(signal.aborted, false);
        assert.notEqual(phase, 'save-with-approval', 'picker acceptance must not cause a second startup dialog');
        if (phase === 'download-cancel' || phase === 'download-superseded') {
          paused = true;
          return gate.promise.then(() => false);
        }
        return wait('download', true);
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
      whileArenaMounted: (fn) => fn,
      arenaGamesPerOpening: () => 1,
      TournamentStandings: class {},
      buildSchedule: () => [{ whiteId: '0', blackId: '1', opening: {} }],
      tournamentPairings: () => [],
      selectedOpenings: () => [],
      createArenaCheckpoint: () => ({ runId: 'test', revision: 1 }),
      captureCheckpointConfig: () => ({}),
      saveArenaCheckpoint: () => wait('save', undefined),
      checkpointToken: null,
      checkpointProblem: '',
      ArenaCheckpointConflictError: class extends Error {},
      checkpoint: null,
      hideEngineWarmupStatus() {},
      renderCheckpointStatus() {},
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
    } else assert.deepEqual(api.state(), { running: false, startPending: false, abort: null, checkpoint: null });
    assert.equal(admitted, 0);
    assert.doesNotMatch(el('message').textContent, /failed/);
  });
}

const restoreGuard = extract(source.play, 'async function ctxCheckRestoredDownloadConsent(', '/** Imported setup must not make');
const readinessEntry = extract(source.play, 'async function ctxVerifySelectedEngineReady(', '  releaseUnusedBigNetSearchers(', 'admit();\n}');
const moveEntry = extract(source.play, 'async function ctxEngineTurn(', '  ctxSetEngineThinking(ctx, true);', 'await admit();\n}');
for (const key of Object.keys(BIG_NETS)) {
  for (const mode of ['readiness', 'engine-first']) {
    for (const accepted of [false, true]) {
      test(`restored ${key} gates ${mode} admission on cache-aware approval (accepted=${accepted})`, async () => {
        const permission = Promise.withResolvers();
        let prompts = 0;
        let admissions = 0;
        const retry = { hidden: true };
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
          buttonEl: () => retry,
          ctxRender() {},
          ctxSetEngineFailure: (_ctx, error) => assert.fail(String(error)),
          admit: () => admissions++,
        };
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
      buttonEl: () => ({}),
      ctxSetEngineFailure: (_ctx, error) => assert.fail(String(error)),
    };
    const code =
      extract(source.play, 'function ctxSetOpponent(', 'function ctxRestoreOpponentControls(') +
      extract(source.play, 'function ctxMaybeQueueRestart(', 'function ctxTuneOpponent(') +
      extract(source.play, "  trackListener(ctx, selectEl('engineSelect'), 'change', async () => {", "  trackListener(ctx, selectEl('levelSelect')") +
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
      ctxSetEngineNote: (text) => {
        note = text;
      },
      ctxPrepareOrStartEngine: () => preparations++,
    };
    const code = extract(source.play, "  trackListener(ctx, selectEl('engineSelect'), 'change', async () => {", "  trackListener(ctx, selectEl('levelSelect')");
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
  const setOpponent = new Function(`${code}; return ctxSetOpponent;`)();
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
