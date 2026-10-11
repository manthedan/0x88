import assert from 'node:assert/strict';

/** Record a completed request, including a scheduled cancellation that missed it. */
export function recordCompletedProbeSearch(row, infos, searchDepth, cancelScheduled) {
  assert.equal(infos[0]?.depth, searchDepth);
  row.info = infos;
  row.cancelMissed = cancelScheduled;
}
