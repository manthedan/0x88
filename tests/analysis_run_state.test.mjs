import assert from 'node:assert/strict';
import test from 'node:test';
import { analysisIsAnalyzing, analysisOperationId, initialAnalysisRunState, transitionAnalysis } from '../src/lc0/analysisRunState.ts';

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
    if (invalidation === 'mode') event.admit = false;
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
    m.send({ type: 'STOP', preserveMaintenance: true, retire: true });
    m.settle(old.runId);
    m.send({ type: 'SETTINGS_CHANGED', maintenance: none, request: false, intent: { requestId: 100, force: false }, admit: false });
    m.send({ type: 'VISIBILITY', visible: false, resumeSearch: false, admit: false });
    m.send({ type: 'VISIBILITY', visible: true, resumeSearch: false, admit: false });
    assert.deepEqual(m.state().maintenance, both);
    assert.equal(m.effects.some((e) => e.type === 'maintain'), false);
    m.request();
    assert.equal(m.state().execution.tag, 'maintaining');
  });
}

test('admission resume drains queued settings but not a stopped search', () => {
  const m = machine();
  m.send({ type: 'SETTINGS_CHANGED', maintenance: both, request: true, intent: { requestId: 100, force: false }, admit: false });
  assert.equal(m.state().execution.tag, 'idle');
  m.send({ type: 'RESUME', admit: false });
  assert.equal(m.state().execution.tag, 'idle');
  m.send({ type: 'RESUME' });
  assert.equal(m.state().execution.tag, 'maintaining');
  m.send({ type: 'STOP', preserveMaintenance: true });
  m.settle();
  m.send({ type: 'RESUME' });
  assert.equal(m.state().execution.tag, 'idle');
  assert.equal(m.effects.some((e) => e.type === 'prepare'), false);
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
