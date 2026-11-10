import assert from 'node:assert/strict';
import { test } from 'node:test';
import { arenaMoveDelay, guardArenaUnload } from '../src/lc0/arenaNavigation.ts';

test('unload is guarded only for an active run', () => {
  for (const active of [false, true]) {
    const event = {
      returnValue: undefined,
      prevented: false,
      preventDefault() {
        this.prevented = true;
      },
    };
    guardArenaUnload(event, active);
    assert.equal(event.prevented, active);
    assert.equal(event.returnValue, active ? '' : undefined);
  }
});

test('hidden tournaments bypass pacing, and hiding cancels a pending visible delay', async () => {
  const visibility = new EventTarget();
  visibility.hidden = true;
  const controller = new AbortController();
  await arenaMoveDelay(60_000, controller.signal, visibility);
  visibility.hidden = false;
  const pending = arenaMoveDelay(60_000, controller.signal, visibility);
  visibility.hidden = true;
  visibility.dispatchEvent(new Event('visibilitychange'));
  await pending;
  visibility.hidden = false;
  const aborted = arenaMoveDelay(60_000, controller.signal, visibility);
  controller.abort();
  await aborted;
});
