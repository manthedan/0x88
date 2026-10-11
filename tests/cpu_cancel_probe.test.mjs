import assert from 'node:assert/strict';
import { test } from 'node:test';
import { recordCompletedProbeSearch } from '../scripts/lib/cpu_cancel_probe_result.mjs';

test('CPU cancel probe records a deeper completed request when its cancel timer misses', () => {
  const row = { aborted: false };
  const infos = [{ depth: 2, nodes: 100 }];
  recordCompletedProbeSearch(row, infos, 2, true);
  assert.equal(row.cancelMissed, true);
  assert.equal(row.aborted, false);
  assert.equal(row.info, infos);
  assert.equal(row.info[0].depth, 2);
});

test('CPU cancel probe validates normal completions and rejects incomplete depth without recording success', () => {
  const normal = {};
  recordCompletedProbeSearch(normal, [{ depth: 12 }], 12, false);
  assert.equal(normal.cancelMissed, false);
  const incomplete = {};
  assert.throws(() => recordCompletedProbeSearch(incomplete, [{ depth: 1 }], 2, true), assert.AssertionError);
  assert.throws(() => recordCompletedProbeSearch(incomplete, [], 2, true), assert.AssertionError);
  assert.deepEqual(incomplete, {});
});
