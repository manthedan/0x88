import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';

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

for (const replacement of ['cpu-profile', 'manual', 'unmount']) {
  test(`Analysis ${replacement} supersedes a pending large-profile cache probe without another prompt or stale commit`, async () => {
    const probe = Promise.withResolvers();
    const mountAbort = new AbortController();
    const prompts = [];
    const messages = [];
    const nodes = new Map();
    const el = (id) => {
      if (!nodes.has(id)) nodes.set(id, {});
      return nodes.get(id);
    };
    const initial = [{ family: 'sf', variant: 'lite', strength: 8 }];
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
    };
    const code =
      extract(source.analysis, 'async function profileRowsForUse(', 'function loadEngineProfiles(') +
      extract(source.analysis, 'let profileLoadAbort:', 'function saveCurrentEngineProfile(');
    const api = new Function(
      'deps',
      `let { ${Object.keys(deps)} } = deps; ${code}; return {
      apply: applyEngineProfile, rows: () => engineRows,
      manual() { profileLoadAbort?.abort(); engineRows = [{family:'sf',variant:'full',strength:12}]; }
    };`,
    )(deps);
    const old = api.apply({ name: 'old', rows: ['bt4', 't3'].map((variant) => ({ family: 'lc0', variant, strength: 10 })), multiPv: 1 });
    await until(() => prompts.length === 1);
    if (replacement === 'cpu-profile') await api.apply({ name: 'new', rows: [{ family: 'sf', variant: 'full', strength: 12 }], multiPv: 2 });
    else if (replacement === 'manual') api.manual();
    else mountAbort.abort();
    assert.equal(prompts[0].signal.aborted, true);
    probe.resolve(true);
    await old;
    assert.deepEqual(
      prompts.map((entry) => entry.key),
      ['bt4'],
      'the old second row must never prompt',
    );
    assert.deepEqual(api.rows(), replacement === 'unmount' ? initial : [{ family: 'sf', variant: 'full', strength: 12 }]);
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
