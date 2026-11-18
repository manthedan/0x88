import assert from 'node:assert/strict';
import test from 'node:test';
import { disposeOrtWorker } from '../src/lc0/disposeOrtWorker.ts';
import { awaitEngineRetirements, ENGINE_RETIREMENT_CAP_MS, hasEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';

class Worker extends EventTarget {
  terminated = false;
  postMessage() {}
  terminate() {
    this.terminated = true;
  }
  reply(data) {
    this.dispatchEvent(new MessageEvent('message', { data }));
  }
}

for (const failure of ['release', 'transport'])
  test(`a ${failure} error stays diagnostic, terminates the worker, and ends the replacement wait`, async () => {
    const worker = new Worker();
    const closing = disposeOrtWorker(worker);
    const diagnostic = assert.rejects(closing, /failed/);
    trackEngineRetirement(closing);
    let admitted = false;
    const replacement = awaitEngineRetirements().then(() => {
      admitted = true;
    });
    await new Promise(setImmediate);
    assert.equal(admitted, false, 'the replacement waits for the release to settle');
    if (failure === 'release') worker.reply({ type: 'error', id: -1, error: 'session release failed' });
    else worker.dispatchEvent(new Event('error'));
    await diagnostic;
    await replacement;
    assert.equal(worker.terminated, true);
    assert.equal(hasEngineRetirements(), false);
  });

test('a waiter never waits on a release registered after it started (no self-wait)', async () => {
  const earlier = Promise.withResolvers();
  trackEngineRetirement(earlier.promise);
  let admitted = false;
  const waiter = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  // A replacement's own cleanup that cannot finish before the replacement is admitted.
  const later = trackEngineRetirement(waiter);
  earlier.resolve();
  await Promise.race([waiter, new Promise((resolve) => setTimeout(resolve, 50))]);
  assert.equal(admitted, true, 'the waiter is admitted once the releases it observed have settled');
  await later;
  await awaitEngineRetirements();
  assert.equal(hasEngineRetirements(), false);
});

test('a stalled release stops holding the barrier after the cap, with a warning', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const warnings = [];
  t.mock.method(console, 'warn', (message) => warnings.push(String(message)));
  trackEngineRetirement(new Promise(() => {}));
  let admitted = false;
  const waiter = awaitEngineRetirements().then(() => {
    admitted = true;
  });
  t.mock.timers.tick(ENGINE_RETIREMENT_CAP_MS - 1);
  await new Promise(setImmediate);
  assert.equal(admitted, false, 'a slow release still holds the barrier before the cap');
  t.mock.timers.tick(1);
  await waiter;
  assert.match(warnings.join('\n'), /still pending/);
  assert.equal(hasEngineRetirements(), false, 'later engine starts do not pay the cap again');
});

test('a caller awaiting its own release still waits for the real completion past the cap', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.mock.method(console, 'warn', () => {});
  const release = Promise.withResolvers();
  let released = false;
  const direct = trackEngineRetirement(release.promise).then(() => {
    released = true;
  });
  t.mock.timers.tick(ENGINE_RETIREMENT_CAP_MS);
  await awaitEngineRetirements();
  assert.equal(released, false, 'e.g. GPU-IO setup failure must not create its CPU-IO session while the first is live');
  release.resolve();
  await direct;
});
