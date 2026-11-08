import assert from 'node:assert/strict';
import test from 'node:test';
import { withEngineTimeout } from '../src/lc0/engineTimeout.ts';
import { EngineResourceBroker } from '../src/lc0/resourceBroker.ts';

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('cancelled cold review startups release both broker slots before initialization drains', async () => {
  const broker = new EngineResourceBroker({ policy: 'bounded', maxConcurrentCpu: 2, environment: () => ({ hardwareConcurrency: 8, crossOriginIsolated: true }) });
  for (const id of ['sf', 'other']) broker.register(id, { resourceClass: 'cpu', maxThreads: 1 });
  let finishInit;
  const coldInit = new Promise((resolve) => { finishInit = resolve; });
  let searches = 0;
  const run = async (signal) => {
    const lease = await broker.acquire({ engineId: 'sf', signal });
    try {
      return await withEngineTimeout('Stockfish', signal, 1000, async (workSignal) => {
        await coldInit; // Models Stockfish's non-abortable init before newGame checks its signal.
        if (workSignal.aborted) throw new DOMException('Aborted', 'AbortError');
        searches++;
      });
    } finally {
      lease.release();
    }
  };
  const controllers = [new AbortController(), new AbortController()];
  const tasks = controllers.map((controller) => run(controller.signal));
  await tick();
  assert.equal(broker.snapshot().activeLeases.length, 2);
  const rejections = tasks.map((task) => assert.rejects(task, { name: 'AbortError' }));
  controllers.forEach((controller) => controller.abort());
  await Promise.all(rejections);
  assert.equal(broker.snapshot().activeLeases.length, 0);
  const other = await broker.acquire({ engineId: 'other' });
  assert.equal(other.engineId, 'other', 'another reviewer is not blocked by cancelled startup');
  other.release();
  finishInit();
  await tick();
  assert.equal(searches, 0, 'late initialization must not start an abandoned search');
});

test('engine timeout also bounds a startup that ignores AbortSignal', async () => {
  let workSignal;
  await assert.rejects(withEngineTimeout('Cold engine', new AbortController().signal, 5, (signal) => {
    workSignal = signal;
    return new Promise(() => {});
  }), /Cold engine timed out/);
  assert.equal(workSignal.aborted, true);
});

test('engine timeout returns successful work and does not start already-aborted work', async () => {
  assert.equal(await withEngineTimeout('Ready engine', new AbortController().signal, 100, async () => 42), 42);
  const controller = new AbortController();
  controller.abort();
  let started = false;
  await assert.rejects(withEngineTimeout('Stopped engine', controller.signal, 100, async () => { started = true; }), { name: 'AbortError' });
  assert.equal(started, false);
});
