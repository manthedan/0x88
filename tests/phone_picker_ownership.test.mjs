import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';

const source = Object.fromEntries(['arena', 'play'].map((name) => [name, readFileSync(new URL(`../src/lc0/${name}Browser.ts`, import.meta.url), 'utf8')]));
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

for (const action of ['cpu', 'startup', 'replacement-picker']) {
  test(`Play ${action} aborts obsolete picker metadata before another dialog or admission`, async () => {
    const ctx = {
      pendingPreferredEngineId: 'leela-queen-odds',
      lastEngineId: 'maia3',
      opponent: { active: { engineId: 'maia3' } },
      abort: new AbortController(),
      consentAbort: new AbortController(),
      disposed: false,
      engineThinking: false,
      preparingEngine: false,
      engineReady: false,
      gameOver: null,
      board: { turn: 'w' },
      humanColor: 'w',
    };
    const node = { value: 'lc0-bt4' };
    const probes = [];
    let handler;
    let admissions = 0;
    const deps = {
      ctx,
      BIG_NETS,
      trackListener: (_ctx, _node, _event, listener) => {
        handler = listener;
      },
      selectEl: () => node,
      ctxDraftEngine: () => ({ id: node.value, family: node.value === 'sf-lite' ? 'sf' : 'lc0', variant: node.value === 'lc0-t3' ? 't3' : 'bt4' }),
      requestLargeNetConsent: (config, signal) => {
        const pending = Promise.withResolvers();
        probes.push({ config, signal, pending });
        return pending.promise;
      },
      ctxEngineOptionState: () => assert.fail('an obsolete picker must not validate or commit its model'),
      ctxRenderLevelOptions() {},
      ctxSetEngineNote() {},
      ctxVerifySelectedEngineReady: () => admissions++,
      ctxEngineTurn: () => assert.fail('the human moves first'),
    };
    playScope(ctx, deps);
    const code =
      extract(source.play, 'function ctxCancelPickerConsent(', 'function ctxCancelEngineTurn(') +
      extract(source.play, 'function ctxPrepareOrStartEngine(', 'function ctxResetGameStateForMount(') +
      extract(
        source.play,
        "  ctx.scope.listen(ctx.dom.selectEl('engineSelect'), 'change', async () => {",
        "  ctx.scope.listen(ctx.dom.selectEl('levelSelect')",
      );
    const prepare = new Function(
      'deps',
      `const { ${Object.keys(deps)} } = deps;
      function ctxMaybeQueueRestart(ctx) { ctx.opponent.active.engineId = node.value; ctxPrepareOrStartEngine(ctx); }
      const node = selectEl(); ${code}; return ctxPrepareOrStartEngine;`,
    )(deps);
    const old = handler();
    assert.equal(probes.length, 1);
    assert.equal(probes[0].signal.aborted, false);
    let replacement;
    if (action === 'cpu') {
      node.value = 'sf-lite';
      await handler();
    } else if (action === 'startup') prepare(ctx);
    else {
      node.value = 'lc0-t3';
      replacement = handler();
    }
    assert.equal(probes[0].signal.aborted, true, 'superseding work cancels the readonly lookup before it can show a modal');
    assert.equal(ctx.consentAbort.signal.aborted, false, 'restored-download consent retains its independent mount scope');
    probes[0].pending.resolve(true);
    await old;
    if (replacement) {
      assert.equal(ctx.pickerConsent.resumeFallback, true, 'replacement pickers inherit the abandoned saved-engine preparation');
      assert.equal(probes[1].signal.aborted, false);
      assert.equal(admissions, 0, 'the old picker cannot prepare over its new owner');
      probes[1].pending.resolve(false);
      await replacement;
    }
    assert.equal(admissions, 1, 'the current/reverted opponent is prepared exactly once');
    assert.equal(ctx.opponent.active.engineId, action === 'cpu' ? 'sf-lite' : 'maia3');
    assert.equal(ctx.pickerConsent, null);
  });
}

function arenaPicker(requestConsent, overrides = {}) {
  const rows = [
    { family: 'lc0', variant: 'small', strength: 8 },
    { family: 'sf', variant: 'lite', strength: 8 },
    { family: 'sf', variant: 'full', strength: 8 },
  ];
  const callbacks = new Map();
  const nodes = new Map();
  const el = (id) => {
    if (!nodes.has(id))
      nodes.set(id, { textContent: '', addEventListener: (event, handler) => callbacks.set(`${id}:${event}`, handler), querySelector: () => null });
    return nodes.get(id);
  };
  const approvals = new Set();
  const deps = {
    BIG_NETS,
    running: false,
    startPending: false,
    abort: null,
    mountAbort: new AbortController(),
    seatRows: rows,
    arenaDownloadApprovals: approvals,
    ARENA_BIG_NET_KEYS: ['bt4', 't3'],
    el,
    arenaBigNetSeatBlocked: () => false,
    isLc0BigNetVariant: (key) => key === 'bt4' || key === 't3',
    arenaTournamentMode: () => 'match',
    requestLargeNetDownloadConsent: requestConsent,
    isMemoryConstrainedDevice: () => true,
    bigNetSelectableSync: () => true,
    defaultVariant: () => 'lite',
    strengthMeta: () => ({ def: 8 }),
    normalizeSeatRowForDeploy() {},
    reconcileActiveRuntimeFallbackWarning() {},
    persistArenaSeatRows() {},
    activeSeatRows: () => rows,
    peekBigNetSearcher: () => null,
    releaseBigNetSearcher() {},
    buildEngines() {},
    populateSeats() {},
    selectedSeatsNeedLc0Evaluator: () => false,
    loadingLc0: false,
    setArenaStartDisabled() {},
    syncArenaPhase() {},
    syncSeatRowsFromDom() {},
    isReadyMessage: () => false,
    refreshSeatControls() {},
    renderRecklessRuntimeInfo() {},
    renderViridithasRuntimeInfo() {},
    renderBerserkRuntimeInfo() {},
    renderPlentyChessRuntimeInfo() {},
    renderStormphraxRuntimeInfo() {},
    // The handler binds these when it is built, so overrides must apply first.
    ...overrides,
  };
  const code =
    extract(source.arena, 'let arenaPickerConsentAbort:', 'async function bigNetFor(') +
    extract(source.arena, 'async function startMatchUnchecked(', '  await checkpointWrites;', '  admissions++;\n}') +
    extract(source.arena, "  el('arenaSeatList').addEventListener('change'", "  el('arenaSeatList').addEventListener('input'") +
    extract(source.arena, "  el('arenaSeatList').addEventListener('click'", "  el('arenaConcurrencySelect').addEventListener(");
  const api = new Function(
    'deps',
    `let { ${Object.keys(deps)} } = deps; let admissions = 0; ${code}; return {start:startMatchUnchecked, admissions:()=>admissions};`,
  )(deps);
  const event = (name, value) => ({ target: { value, dataset: { seat: '0' }, classList: { contains: (entry) => entry === name } } });
  return { rows, callbacks, approvals, deps, api, pick: (variant) => callbacks.get('arenaSeatList:change')(event('seat-var', variant)), event };
}

for (const action of ['cpu', 'startup', 'remove', 'add', 'mode']) {
  test(`Arena ${action} aborts obsolete picker metadata without changing the new roster`, async () => {
    const permission = Promise.withResolvers();
    let consent;
    const { rows, callbacks, approvals, deps, api, pick, event } = arenaPicker(
      (_config, signal) => {
        consent = signal;
        return permission.promise;
      },
      { bigNetSelectableSync: () => assert.fail('an obsolete picker must never validate or commit its model') },
    );
    const old = pick('bt4');
    assert.equal(consent.aborted, false);
    if (action === 'cpu') await callbacks.get('arenaSeatList:change')(event('seat-fam', 'sf'));
    else if (action === 'startup') await api.start();
    else if (action === 'remove') callbacks.get('arenaSeatList:click')(event('seat-remove', ''));
    else if (action === 'add') callbacks.get('addSeat:click')();
    else callbacks.get('tournamentModeSelect:change')();
    assert.equal(consent.aborted, true, 'roster edits and startup cancel metadata before an obsolete modal can open');
    assert.equal(deps.mountAbort.signal.aborted, false);
    permission.resolve('download');
    await old;
    assert.equal(
      rows.some((row) => row.variant === 'bt4'),
      false,
    );
    assert.equal(approvals.size, 0, 'stale requests cannot save a startup approval');
    assert.equal(api.admissions(), action === 'startup' ? 1 : 0);
  });
}

for (const consent of ['cached', 'download']) {
  test(`Arena picker consent (${consent}) ${consent === 'cached' ? 'leaves Start to recheck the cache' : 'approves the Start download'}`, async () => {
    const { rows, approvals, pick } = arenaPicker(async () => consent);
    await pick('bt4');
    assert.equal(rows[0].variant, 'bt4');
    // A cache hit is not download approval: storage can evict the model before Start.
    assert.deepEqual([...approvals], consent === 'download' ? ['bt4'] : []);
  });
}
