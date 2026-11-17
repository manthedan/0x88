import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { disposeOrtWorker, RetiredOrtWorkerError } from '../src/lc0/disposeOrtWorker.ts';
import { awaitEngineRetirements, trackEngineRetirement } from '../src/lc0/engineRetirement.ts';

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
  test(`a ${failure} error after successful termination remains diagnostic but does not block replacement startup`, async () => {
    const worker = new Worker();
    const closing = disposeOrtWorker(worker);
    const diagnostic = assert.rejects(closing, (error) => error instanceof RetiredOrtWorkerError && /failed/.test(error.message));
    const boundary = trackEngineRetirement(closing);
    const replacement = awaitEngineRetirements();
    if (failure === 'release') worker.reply({ type: 'error', id: -1, error: 'session release failed' });
    else worker.dispatchEvent(new Event('error'));
    await diagnostic;
    await boundary;
    await replacement;
    assert.equal(worker.terminated, true);
  });

test('a failed worker termination stays latched even if it settles before the next route starts', async () => {
  const barrier = await import('../src/lc0/engineRetirement.ts?failed-termination');
  const worker = new Worker(),
    failure = new Error('termination failed');
  worker.terminate = () => {
    throw failure;
  };
  const closing = disposeOrtWorker(worker);
  const boundary = barrier.trackEngineRetirement(closing);
  const rejected = assert.rejects(boundary, (error) => error === failure && !(error instanceof RetiredOrtWorkerError));
  worker.reply({ type: 'disposed', id: -1 });
  await rejected;
  await new Promise(setImmediate);
  assert.equal(barrier.hasEngineRetirements(), true);
  await assert.rejects(barrier.awaitEngineRetirements(), (error) => error === failure);
});

test('Analysis exposes failed retirement without leaving Analyze disabled or starting a fallback model', async () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const start = source.indexOf('async function loadLc0BackendOwned('),
    end = source.indexOf('async function reloadLc0Backend(', start);
  const elements = new Map();
  const el = (id) => {
    if (!elements.has(id))
      elements.set(id, {
        disabled: true,
        toggleAttribute(_name, disabled) {
          this.disabled = disabled;
        },
      });
    return elements.get(id);
  };
  const code = stripTypeScriptTypes(source.slice(start, end));
  const load = new Function('awaitEngineRetirements', 'el', 'selectEl', 'initWorker', `${code};return loadLc0BackendOwned;`)(
    async () => {
      throw new Error('cannot terminate worker');
    },
    el,
    el,
    () => assert.fail('replacement admitted beside an unterminated owner'),
  );
  assert.equal(await load(() => false), false);
  assert.equal(el('analyze').disabled, false);
  assert.equal(el('lc0RuntimeSelect').disabled, false);
  assert.match(el('message').textContent, /could not be retired.*cannot terminate/);
});
