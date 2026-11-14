import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { analysisCanPrepare, analysisIsAnalyzing, analysisOperationId, analysisRunStateSnapshot, appendAnalysisTransitionLog, initialAnalysisRunState, transitionAnalysis } from '../src/lc0/analysisRunState.ts';

const none = { reloadLc0: false, disposeUnused: false };
const both = { reloadLc0: true, disposeUnused: true };
const snapshot = Object.freeze({ selectionKey: 'history/settings/model', rowKeys: Object.freeze(['row']) });

function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

// Timers and preparation/maintenance completions are deferred independently.
function machine() {
  let state = initialAnalysisRunState(), id = 0, now = 0;
  const timers = new Map(), effects = [], publishers = new Set();
  const send = (event) => {
    const previous = freeze(state);
    const result = transitionAnalysis(previous, { revision: state.revision + 1, ...event, nextId: ++id });
    state = result.state;
    for (const effect of result.effects) {
      effects.push(effect);
      if (effect.type === 'schedule') timers.set(effect.ticket, now + effect.delayMs);
      if (effect.type === 'clearTimer') timers.delete(effect.ticket);
      if (effect.type === 'cancel') publishers.delete(effect.operationId);
      if (effect.type === 'search') {
        assert.equal(state.visible, true, 'never search while hidden');
        assert.deepEqual(state.modes, [], 'never search while another mode owns admission');
        assert.equal(state.execution.tag, 'running');
        assert.equal(effect.owner.runId, state.execution.owner.runId);
        publishers.add(effect.owner.runId);
      }
    }
    if (event.type === 'SETTLED') publishers.delete(event.operationId);
    assert.ok(publishers.size <= 1, 'at most one publishing owner');
    assert.equal(analysisIsAnalyzing(state), ['preparing', 'running'].includes(state.execution.tag));
    return result.effects;
  };
  return {
    send, effects,
    state: () => state,
    owner: () => state.execution.owner,
    operation: () => analysisOperationId(state.execution),
    request: (force = false, delayMs = 0) => send({ type: 'REQUEST', intent: { requestId: ++id, force }, delayMs }),
    settings: (maintenance = none, request = true) => send({ type: 'SETTINGS_CHANGED', maintenance, request, intent: { requestId: ++id, force: false } }),
    prepared: (owner = state.execution.owner, revision = owner.preparationRevision) => send({ type: 'PREPARED', owner, revision, snapshot }),
    settle: (operationId = analysisOperationId(state.execution), failed = false) => send({ type: 'SETTLED', operationId, failed }),
    tick: (ms) => {
      now += ms;
      for (const [ticket, at] of [...timers]) if (at <= now) {
        timers.delete(ticket);
        send({ type: 'DEBOUNCE_FIRED', ticket });
      }
    },
  };
}

for (const [name, sequence] of [
  ['Stop during debounce', ['request', 'stop', 'tick']],
  ['Stop during preparation', ['requestNow', 'remember', 'stop', 'preparedOld', 'settleOld']],
  ['hide during preparation', ['requestNow', 'remember', 'hide', 'preparedOld', 'settleOld']],
  ['dispose during preparation', ['requestNow', 'remember', 'dispose', 'preparedOld', 'settleOld', 'requestNow']],
  ['Stop then hide before finally', ['requestNow', 'remember', 'stop', 'hide', 'settleOld', 'show']],
]) {
  test(`event sequence: ${name} cannot launch a stale search`, () => {
    const m = machine();
    let old;
    for (const action of sequence) {
      if (action === 'request') m.request(false, 250);
      if (action === 'requestNow') m.request();
      if (action === 'remember') old = m.owner();
      if (action === 'stop') m.send({ type: 'STOP', preserveMaintenance: true });
      if (action === 'hide') m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
      if (action === 'show') m.send({ type: 'VISIBILITY', visible: true, resumeSearch: false });
      if (action === 'dispose') m.send({ type: 'DISPOSE' });
      if (action === 'tick') m.tick(1000);
      if (action === 'preparedOld') m.prepared(old);
      if (action === 'settleOld') m.settle(old.runId);
    }
    assert.equal(m.effects.filter((e) => e.type === 'search').length, 0);
  });
}

test('fake-clock debounce ignores replaced tickets and force survives coalescing and hide', () => {
  const m = machine();
  m.request(true, 250);
  const oldTicket = m.state().execution.ticket;
  m.tick(100);
  m.request(false, 250);
  m.send({ type: 'DEBOUNCE_FIRED', ticket: oldTicket });
  assert.equal(m.state().execution.tag, 'scheduled');
  m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
  m.request(false, 250);
  m.tick(1000);
  assert.equal(m.state().wanted.force, true);
  assert.equal(m.effects.some((e) => e.type === 'prepare'), false);
  m.send({ type: 'VISIBILITY', visible: true, resumeSearch: true });
  assert.equal(m.state().execution.intent.force, true);
  assert.equal(m.prepared()[0].type, 'search');
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.settle();
  m.request();
  assert.equal(m.state().execution.intent.force, false, 'Stop drops the old force');
});

test('only the matching owner settles and a replacement does not wait for old physical work', () => {
  const m = machine();
  m.request();
  const old = m.owner();
  m.prepared();
  m.request();
  const replacement = m.owner();
  assert.notEqual(old.runId, replacement.runId);
  assert.deepEqual(m.prepared(old), []);
  m.prepared();
  m.settings(both);
  const state = m.state();
  assert.deepEqual(m.settle(old.runId), []);
  assert.equal(m.state(), state, 'old finally cannot consume pending work');
  m.settle(replacement.runId);
  assert.equal(m.state().execution.tag, 'maintaining');
  assert.deepEqual(m.state().execution.maintenance, both);
});

for (const invalidation of ['revision', 'queued request', 'reload', 'dispose', 'owner revision', 'mode']) {
  test(`PREPARED rejects ${invalidation} without a search effect`, () => {
    const m = machine();
    m.request();
    const owner = m.owner();
    let event = { type: 'PREPARED', owner, revision: owner.preparationRevision, snapshot };
    if (invalidation === 'revision') m.settings(none, false);
    if (invalidation === 'queued request') m.request();
    if (invalidation === 'reload') m.settings({ ...none, reloadLc0: true });
    if (invalidation === 'dispose') m.settings({ ...none, disposeUnused: true });
    if (invalidation === 'owner revision') event.owner = { ...owner, preparationRevision: -1 };
    if (invalidation === 'mode') m.send({ type: 'MODE_ENTER', mode: 'review' });
    assert.deepEqual(m.send(event), []);
    assert.equal(m.effects.some((e) => e.type === 'search'), false);
  });
}

test('queued maintenance survives hide, Stop, stale finally and maintenance settlement', () => {
  const m = machine();
  m.request(true);
  const old = m.owner();
  m.settings(both);
  m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.settle(old.runId);
  assert.deepEqual(m.state().maintenance, both);
  assert.equal(m.state().wanted, null);
  assert.equal(m.effects.some((e) => e.type === 'maintain'), false);
  m.send({ type: 'VISIBILITY', visible: true, resumeSearch: false });
  const operation = m.operation();
  m.settings(both, false);
  m.settle(old.runId);
  assert.deepEqual(m.state().maintenance, both);
  m.settle(operation);
  assert.equal(m.state().execution.tag, 'maintaining');
  assert.deepEqual(m.state().execution.maintenance, both);
  m.settle();
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'search'), false);
});

test('a hidden maintenance operation is replaced without letting its old callback consume work', () => {
  const m = machine();
  m.settings(both);
  const old = m.operation();
  m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
  m.send({ type: 'VISIBILITY', visible: true, resumeSearch: true });
  const replacement = m.operation();
  assert.notEqual(old, replacement);
  m.settings(both);
  const state = m.state();
  m.settle(old);
  assert.equal(m.state(), state);
  m.settle(replacement);
  assert.equal(m.state().execution.tag, 'maintaining');
  m.settle();
  assert.equal(m.state().execution.tag, 'preparing');
  assert.equal(m.prepared()[0].type, 'search');
});

test('fresh settings replace a cancelling owner without waiting for unabortable preparation', () => {
  const m = machine();
  m.request();
  const old = m.owner();
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.settings(both);
  const replacement = m.operation();
  assert.equal(m.state().execution.tag, 'maintaining');
  assert.notEqual(replacement, old.runId);
  const state = m.state();
  m.settle(old.runId);
  assert.equal(m.state(), state);
  m.settle(replacement);
  assert.equal(m.prepared()[0].type, 'search');
});

test('failed backend loads do not loop, but a failed superseded load still applies new maintenance', () => {
  const m = machine();
  m.request();
  m.send({ type: 'NEEDS_BACKEND', owner: m.owner() });
  m.settings(both);
  m.settle(m.operation(), true);
  assert.equal(m.state().execution.tag, 'maintaining');
  m.settle(m.operation(), true);
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.state().wanted, null);
});

test('Stop during a backend load consumes maintenance but never resurrects search', () => {
  const m = machine();
  m.send({ type: 'BACKEND_STARTED', operationId: 1000, intent: null });
  m.request(true);
  m.settings(both);
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.settle(1000);
  assert.equal(m.state().execution.tag, 'maintaining');
  m.settle();
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'prepare'), false);
});

for (const mode of ['review', 'retry', 'critical']) {
  test(`${mode} handoff retires the owner without applying its pending maintenance`, () => {
    const m = machine();
    m.request();
    const old = m.owner();
    m.settings(both);
    m.send({ type: 'MODE_ENTER', mode });
    m.settle(old.runId);
    m.send({ type: 'SETTINGS_CHANGED', maintenance: none, request: false, intent: { requestId: 100, force: false } });
    m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
    m.send({ type: 'VISIBILITY', visible: true, resumeSearch: false });
    assert.deepEqual(m.state().maintenance, both);
    assert.equal(m.effects.some((e) => e.type === 'maintain'), false);
    m.send({ type: 'MODE_EXIT', mode });
    m.request();
    assert.equal(m.state().execution.tag, 'maintaining');
  });
}

test('admission resume drains queued settings but not a stopped search', () => {
  const m = machine();
  m.send({ type: 'MODE_ENTER', mode: 'critical' });
  m.send({ type: 'SETTINGS_CHANGED', maintenance: both, request: true, intent: { requestId: 100, force: false } });
  assert.equal(m.state().execution.tag, 'idle');
  m.send({ type: 'MODE_EXIT', mode: 'review', resume: true });
  assert.equal(m.state().execution.tag, 'idle');
  m.send({ type: 'MODE_EXIT', mode: 'critical', resume: true });
  assert.equal(m.state().execution.tag, 'maintaining');
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.settle();
  m.send({ type: 'MODE_EXIT', mode: 'critical', resume: true });
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'prepare'), false);
});

for (const mode of ['review', 'retry', 'critical']) {
  for (const phase of ['scheduled', 'preparing', 'running', 'maintaining', 'hidden']) {
    test(`${mode} enters during ${phase}, retires stale work and drops old force`, () => {
      const m = machine();
      if (phase === 'maintaining') m.settings(both);
      else m.request(true, phase === 'scheduled' ? 250 : 0);
      if (phase === 'running') m.prepared();
      if (phase === 'hidden') m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
      const old = m.owner(), operation = m.operation();
      const effects = m.send({ type: 'MODE_ENTER', mode });
      assert.equal(m.state().execution.tag, 'idle');
      assert.deepEqual(m.state().modes, [mode]);
      assert.equal(m.state().wanted, null);
      assert.equal(effects[0]?.type, phase === 'scheduled' ? 'clearTimer' : phase === 'hidden' ? undefined : 'cancel');
      if (old) {
        assert.equal(analysisCanPrepare(m.state(), old), false);
        assert.deepEqual(m.prepared(old), []);
      }
      if (operation !== null) assert.deepEqual(m.settle(operation), []);
      m.tick(1000);
      assert.deepEqual(m.send({ type: 'MODE_EXIT', mode, resume: mode === 'critical' }), []);
      m.send({ type: 'VISIBILITY', visible: true, resumeSearch: true });
      assert.equal(m.state().execution.tag, 'idle', 'enter never requeues interrupted work, even maintenance');
      m.request();
      assert.equal(m.state().execution.intent.force, false);
    });
  }

  test(`${mode} requests and settings retain their existing exit policy`, () => {
    const m = machine();
    m.send({ type: 'MODE_ENTER', mode });
    m.request(true, 250);
    m.request(false, 250);
    assert.equal(m.state().execution.tag, 'idle');
    assert.equal(m.state().wanted?.force ?? null, mode === 'critical' ? true : null);
    m.settings();
    assert.equal(m.state().wanted.force, mode === 'critical');
    m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
    m.send({ type: 'MODE_EXIT', mode, resume: mode === 'critical' });
    assert.equal(m.state().execution.tag, 'idle', 'mode exit cannot bypass visibility');
    m.send({ type: 'VISIBILITY', visible: true, resumeSearch: true });
    assert.equal(m.state().execution.intent.force, mode === 'critical');
    assert.equal(m.prepared()[0].type, 'search');
  });

  test(`${mode} exit leaves queued maintenance deferred unless explicitly resumed`, () => {
    const m = machine();
    m.send({ type: 'MODE_ENTER', mode });
    m.settings(both);
    m.send({ type: 'MODE_EXIT', mode, resume: mode === 'critical' });
    assert.equal(m.state().execution.tag, mode === 'critical' ? 'maintaining' : 'idle');
    if (mode !== 'critical') m.request();
    assert.equal(m.state().execution.tag, 'maintaining');
    m.settle();
    assert.equal(m.state().execution.intent.force, true, 'reload still forces the queued search');
  });

  test(`Stop during ${mode} clears search, preserving only queued maintenance`, () => {
    const m = machine();
    m.send({ type: 'MODE_ENTER', mode });
    m.settings(both);
    m.send({ type: 'STOP', preserveMaintenance: true });
    assert.deepEqual(m.state().modes, [mode], 'Stop does not release another mode’s resources');
    m.send({ type: 'MODE_EXIT', mode, resume: mode === 'critical' });
    if (mode !== 'critical') m.send({ type: 'VISIBILITY', visible: true, resumeSearch: false });
    assert.equal(m.state().execution.tag, 'maintaining');
    m.settle();
    assert.equal(m.state().execution.tag, 'idle');
    assert.equal(m.effects.some((effect) => effect.type === 'prepare'), false);
  });
}

test('critical drain cannot release review ownership or consume its maintenance', () => {
  const m = machine();
  m.send({ type: 'MODE_ENTER', mode: 'critical' });
  m.send({ type: 'MODE_ENTER', mode: 'review' });
  m.settings(both);
  const state = m.state();
  assert.deepEqual(m.send({ type: 'MODE_ENTER', mode: 'review' }), []);
  assert.equal(m.state(), state, 'duplicate enter is harmless');
  assert.deepEqual(m.send({ type: 'MODE_EXIT', mode: 'critical', resume: true }), []);
  assert.deepEqual(m.state().modes, ['review']);
  assert.equal(m.state().execution.tag, 'idle');
  m.send({ type: 'MODE_EXIT', mode: 'review' });
  assert.equal(m.state().execution.tag, 'idle');
  m.request();
  assert.equal(m.state().execution.tag, 'maintaining');
});

test('diagnostic snapshots are detached, deeply frozen and omit run payloads in every phase', () => {
  const m = machine();
  const inspect = () => {
    const state = m.state(), view = analysisRunStateSnapshot(state);
    const checkFrozen = (value) => {
      if (!value || typeof value !== 'object') return;
      assert.equal(Object.isFrozen(value), true);
      Object.values(value).forEach(checkFrozen);
    };
    checkFrozen(view);
    assert.notEqual(view.execution, state.execution);
    assert.notEqual(view.modes, state.modes);
    assert.notEqual(view.maintenance, state.maintenance);
    assert.equal(view.execution.tag, state.execution.tag);
    assert.equal(view.execution.runId, analysisOperationId(state.execution));
    assert.doesNotMatch(JSON.stringify(view), /history\/settings\/model|rowKeys|selectionKey|snapshot|fen/i);
    assert.throws(() => { view.visible = false; }, TypeError);
    assert.throws(() => { view.modes.push('review'); }, TypeError);
    return view;
  };
  inspect();
  m.request(true, 250); inspect();
  m.tick(250); inspect();
  m.prepared();
  const running = inspect();
  assert.throws(() => { running.execution.owner.runId = -1; }, TypeError);
  assert.throws(() => { running.execution.intent.force = false; }, TypeError);
  m.settings(both); inspect();
  m.settle(); inspect();
  m.settings(both); inspect();
  m.send({ type: 'MODE_ENTER', mode: 'critical' }); inspect();
  m.settings(both); const queued = inspect();
  assert.throws(() => { queued.wanted.force = true; }, TypeError);
  m.send({ type: 'MODE_EXIT', mode: 'critical', resume: true });
  m.send({ type: 'STOP', preserveMaintenance: true }); inspect();
  m.send({ type: 'DISPOSE' }); inspect();
  assert.equal(running.execution.tag, 'running', 'older snapshots never change');
  assert.deepEqual(queued.modes, ['critical']);
});

test('transition log is bounded, immutable, detached and contains no event payloads', () => {
  let state = initialAnalysisRunState(), log = Object.freeze([]), first;
  for (let nextId = 1; nextId <= 75; nextId++) {
    const event = { type: 'REQUEST', intent: { requestId: nextId, force: true }, delayMs: 250, nextId };
    const next = transitionAnalysis(freeze(state), event);
    const previousLog = log;
    log = appendAnalysisTransitionLog(log, state, event, next);
    assert.notEqual(log, previousLog);
    assert.equal(log.length, Math.min(nextId, 50));
    if (nextId === 1) first = log;
    state = next.state;
    event.intent.force = false;
    next.effects.length = 0;
  }
  assert.equal(first.length, 1);
  assert.equal(log.length, 50);
  assert.deepEqual(log[0], { event: 'REQUEST', mode: null, from: 'scheduled', to: 'scheduled', effects: ['clearTimer', 'schedule'], runId: null, revision: 0 });
  assert.deepEqual(first[0].effects, ['schedule']);
  assert.equal(Object.isFrozen(log), true);
  assert.equal(Object.isFrozen(log[0]), true);
  assert.equal(Object.isFrozen(log[0].effects), true);
  assert.throws(() => log.pop(), TypeError);
  assert.throws(() => { log[0].from = 'running'; }, TypeError);
  assert.throws(() => log[0].effects.push('search'), TypeError);
  assert.doesNotMatch(JSON.stringify(log), /intent|snapshot|fen|selectionKey/i);
  const event = { type: 'MODE_ENTER', mode: 'critical', nextId: 100 };
  const next = transitionAnalysis(state, event);
  log = appendAnalysisTransitionLog(log, state, event, next);
  assert.deepEqual(log.at(-1), { event: 'MODE_ENTER', mode: 'critical', from: 'scheduled', to: 'idle', effects: ['clearTimer'], runId: null, revision: 0 });
});

test('smoke hooks keep existing fields, expose frozen diagnostics and tear down only their own mount', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const from = source.indexOf('  const schedulingStateHook = '), to = source.indexOf('  const pvResize = ', from);
  assert.ok(from >= 0 && to > from);
  const cleanup = [...source.matchAll(/^    if \(smokeGlobal\.__analysis.*$/gm)].map(([line]) => line).join('\n');
  assert.equal(cleanup.split('\n').length, 4);
  const code = ts.transpile(`
    const { analysisIsAnalyzing, analysisOperationId, analysisRunStateSnapshot } = helpers;
    const analysisRunState = helpers.state, analysisTransitionLog = helpers.log;
    const selectedLc0Runtime = () => 'onnx', workerRuntime = 'onnx';
    const hiddenTab = { hidden: false, suspended: [] }, tree = { current: { fen: 'current FEN' } };
    const analysisWorkerProgressMessages = 2, analysisUiProgressSamples = 1;
    const document = { getElementById: () => ({ textContent: 'ready' }) };
    const hook = () => {}, applyHiddenTab = () => {};
    ${source.slice(from, to)}
    return () => { ${cleanup} };
  `, { target: ts.ScriptTarget.ES2022 });
  const global = {}, m = machine();
  m.request(); m.prepared();
  const mount = () => new Function('globalThis', 'helpers', code)(global, {
    analysisIsAnalyzing, analysisOperationId, analysisRunStateSnapshot, state: m.state(), log: Object.freeze([]),
  });
  const unmountOld = mount(), oldHook = global.__analysisSchedulingState;
  const view = oldHook();
  assert.deepEqual(Object.keys(view).sort(), [
    'analyzing', 'searchActive', 'reloadActive', 'reloadQueued', 'disposeQueued', 'phase', 'runId', 'revision',
    'runtime', 'workerRuntime', 'hidden', 'suspended', 'fen', 'workerProgressMessages', 'uiProgressSamples', 'message', 'state',
  ].sort());
  assert.equal(view.fen, 'current FEN');
  assert.equal(view.state.execution.tag, 'running');
  assert.equal(Object.isFrozen(view), true);
  assert.equal(Object.isFrozen(view.state), true);
  assert.equal(Object.isFrozen(global.__analysisTransitionLog()), true);
  const unmountNew = mount();
  unmountOld();
  assert.notEqual(global.__analysisSchedulingState, oldHook);
  assert.equal(typeof global.__analysisTransitionLog, 'function');
  unmountNew();
  assert.deepEqual(global, {});
});

test('transition log records run identity and revision without retaining PREPARED payloads', () => {
  const m = machine();
  m.settings(none, false);
  m.request();
  const before = m.state(), owner = m.owner();
  const event = { type: 'PREPARED', owner, revision: before.revision, snapshot };
  const effects = m.send(event);
  let log = appendAnalysisTransitionLog([], before, event, { state: m.state(), effects });
  assert.deepEqual(log[0], { event: 'PREPARED', mode: null, from: 'preparing', to: 'running', effects: ['search'], runId: owner.runId, revision: before.revision });
  const running = m.state(), enter = { type: 'MODE_ENTER', mode: 'retry' };
  const cancelled = m.send(enter);
  log = appendAnalysisTransitionLog(log, running, enter, { state: m.state(), effects: cancelled });
  assert.equal(log.at(-1).runId, owner.runId);
  assert.doesNotMatch(JSON.stringify(log), /selectionKey|rowKeys|history\/settings\/model/);
});

test('navigation drops only the request, retaining active backend ownership', () => {
  const m = machine();
  m.settings(both);
  const operation = m.operation();
  m.send({ type: 'DROP_REQUEST' });
  assert.equal(m.operation(), operation);
  assert.equal(m.state().wanted, null);
  m.settle();
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'prepare'), false);
  m.request(true, 250);
  m.send({ type: 'DROP_REQUEST' });
  m.tick(1000);
  assert.equal(m.state().execution.tag, 'idle');
});

test('initial setup continues while hidden and only defers its later requested search', () => {
  const m = machine();
  m.send({ type: 'BACKEND_STARTED', operationId: 1000, intent: null });
  m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
  assert.equal(m.operation(), 1000);
  m.request();
  m.settle(1000);
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'prepare'), false);
  m.send({ type: 'VISIBILITY', visible: true, resumeSearch: true });
  assert.equal(m.state().execution.tag, 'preparing');
});

test('cached navigation with auto off drops a hidden search, preserving only maintenance', () => {
  const m = machine();
  m.request(true);
  m.settings(both);
  m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false });
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.send({ type: 'VISIBILITY', visible: true, resumeSearch: false });
  m.settle();
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'search'), false);
});
