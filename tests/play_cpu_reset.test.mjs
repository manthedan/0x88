import assert from 'node:assert/strict';
import test from 'node:test';
import { PendingCpuNewGames } from '../src/lc0/cpuEngineResidency.ts';

// New game B starts while A's ucinewgame is still in flight. A's completion
// must not acknowledge B (nor a replacement engine using the same id).
for (const replace of [false, true]) {
  test(`an older reset cannot clear a newer mark (replacement=${replace})`, async () => {
    const pending = new PendingCpuNewGames();
    const reset = Promise.withResolvers();
    pending.markAll(['sf-lite']);
    const old = pending.resetIfPending('sf-lite', { newGame: () => reset.promise });
    if (replace) pending.forget('sf-lite');
    pending.markAll(['sf-lite']);
    reset.resolve();
    await old;
    assert.equal(pending.has('sf-lite'), true, 'game B still needs its own acknowledged reset');
    let calls = 0;
    await pending.resetIfPending('sf-lite', { newGame: async () => calls++ });
    assert.equal(calls, 1);
    assert.equal(pending.has('sf-lite'), false);
  });
}
