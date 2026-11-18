import assert from 'node:assert/strict';
import test from 'node:test';
import { disposeOrtWorker } from '../src/lc0/disposeOrtWorker.ts';
import { awaitEngineRetirements, hasEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';

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
