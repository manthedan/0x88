import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { phoneSafeEngineVariant } from '../src/lc0/phoneLargeNetPolicy.ts';

function handler(surface, name, deps) {
  const source = readFileSync(new URL(`../src/lc0/${surface}Browser.ts`, import.meta.url), 'utf8');
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf('function wireEvents()', start);
  assert.ok(start >= 0 && end > start);
  return new Function('deps', `const { ${Object.keys(deps)} } = deps; ${stripTypeScriptTypes(source.slice(start, end))}; return ${name};`)(deps);
}
function retainedPool() {
  const pool = new BigNetSessionPool(() => ({
    disposeCalls: 0,
    dispose() {
      this.disposeCalls++;
    },
  }));
  const bt4 = pool.acquire('bt4');
  const t3 = pool.acquire('t3');
  const queen = pool.acquire('lqo');
  return { pool, bt4, t3, queen };
}

test('Analysis revocation retires large workers even when maintenance cannot advance', () => {
  const { pool, bt4, t3, queen } = retainedPool();
  const events = [];
  const revoke = handler('analysis', 'revokeAnalysisPhoneLargeNets', {
    phoneLargeNetsBlocked: () => true,
    ANALYSIS_BIG_NET_KEYS: ['bt4', 't3'],
    disposeBigNetSearcherNow: (key) => pool.disposeNow(key),
    cancelActiveAnalysisRun: () => {
      assert.equal(pool.has('bt4'), false);
      assert.equal(pool.has('t3'), false);
      events.push('cancel');
    },
    cancelAllAnalysisAssetPrefetches: () => events.push('cancel-prefetch'),
    renderEngineList: () => events.push('render'),
    persistManualEngineRows: () => events.push('persist'),
    // Deliberately never execute deferred work: hidden/blocking Analysis cannot do so.
    deferAnalysisSettingsRefresh: () => events.push('deferred'),
  });
  revoke();
  assert.equal(bt4.disposeCalls, 1);
  assert.equal(t3.disposeCalls, 1);
  assert.equal(queen.disposeCalls, 0);
  assert.deepEqual(events, ['cancel', 'cancel-prefetch', 'render', 'persist', 'deferred']);
  pool.disposeAllNow();
});

for (const state of ['running', 'starting', 'idle']) {
  for (const variant of ['cpu-only', 'small', 'bt4', 't3']) {
    test(`Arena ${state} revocation preserves unaffected ${variant} rosters while retiring gated leftovers`, () => {
      const { pool, bt4, t3, queen } = retainedPool();
      const row = variant === 'cpu-only' ? { family: 'sf', variant: 'full' } : { family: 'lc0', variant };
      let aborted = 0;
      const updates = [];
      const revoke = handler('arena', 'revokeArenaPhoneLargeNets', {
        phoneLargeNetsBlocked: () => true,
        activeSeatRows: () => [row, { family: 'sf', variant: 'lite' }],
        phoneSafeEngineVariant: (family, variant) => phoneSafeEngineVariant(family, variant, true, false),
        ARENA_BIG_NET_KEYS: ['bt4', 't3'],
        arenaDownloadApprovals: new Set(),
        disposeBigNetSearcherNow: (key) => pool.disposeNow(key),
        running: state === 'running',
        startPending: state === 'starting',
        abort: { abort: () => aborted++ },
        renderSeatSelectors: () => updates.push('render'),
        persistArenaSeatRows: () => updates.push('persist'),
        buildEngines: () => updates.push('rebuild'),
        populateSeats: () => updates.push('populate'),
      });
      revoke();
      const affected = variant === 'bt4' || variant === 't3';
      assert.equal(aborted, affected ? 1 : 0);
      assert.deepEqual(updates, affected || state === 'idle' ? ['render', 'persist', 'rebuild', 'populate'] : []);
      assert.equal(bt4.disposeCalls, 1);
      assert.equal(t3.disposeCalls, 1);
      assert.equal(queen.disposeCalls, 0);
      pool.disposeAllNow();
    });
  }
}
