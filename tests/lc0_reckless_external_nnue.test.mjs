import assert from 'node:assert/strict';
import { test } from 'node:test';
import { RecklessEngine, RECKLESS_EXTERNAL_NNUE_FILE } from '../src/lc0/recklessEngine.ts';

const FEN = '8/8/8/8/8/8/4P3/4K3 w - - 0 1';

test('persistent Reckless WASI preopens an external NNUE and reports download progress', async () => {
  class MockWorker {
    static messages = [];

    constructor() {
      this.onmessage = null;
      this.onerror = null;
    }

    postMessage(message) {
      MockWorker.messages.push(message);
      if (message.type !== 'start-persistent') return;
      queueMicrotask(() => {
        this.onmessage?.({ data: { type: 'preopen-progress', url: message.preopenFiles[0].url, loadedBytes: 64, totalBytes: 128 } });
        this.onmessage?.({ data: { type: 'persistent-line', stream: 'stdout', line: 'readyok' } });
      });
    }

    terminate() {}
  }

  const previousWorker = globalThis.Worker;
  const previousIsolation = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
  globalThis.Worker = MockWorker;
  Object.defineProperty(globalThis, 'crossOriginIsolated', { configurable: true, value: true });
  let statusNotifications = 0;
  try {
    const engine = new RecklessEngine({}, '/reckless/reckless-simd128-external.wasm', {
      nnueUrl: '/reckless/reckless-v60-7f587dfb.nnue',
      onStatus: () => { statusNotifications += 1; },
      disablePersistentFallback: true,
    });
    await engine.prewarm();
    const start = MockWorker.messages.find((message) => message.type === 'start-persistent');
    assert.deepEqual(start.preopenFiles, [{ name: RECKLESS_EXTERNAL_NNUE_FILE, url: '/reckless/reckless-v60-7f587dfb.nnue' }]);
    assert.equal(engine.runtimeStatus().mode, 'persistent');
    assert.deepEqual(engine.runtimeStatus().browserApiLoad, {
      phase: 'nnue-download',
      nnueUrl: '/reckless/reckless-v60-7f587dfb.nnue',
      loadedBytes: 64,
      totalBytes: 128,
    });
    assert.equal(statusNotifications, 1);
    engine.dispose();
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
    if (previousIsolation) Object.defineProperty(globalThis, 'crossOriginIsolated', previousIsolation);
    else delete globalThis.crossOriginIsolated;
  }
});

test('one-shot Reckless fallback also preopens the external NNUE', async () => {
  class MockWorker {
    static messages = [];

    constructor() {
      this.onmessage = null;
      this.onerror = null;
    }

    postMessage(message) {
      MockWorker.messages.push(message);
      if (message.type === 'run') {
        queueMicrotask(() => this.onmessage?.({
          data: {
            type: 'result',
            id: message.id,
            exitCode: 0,
            stdout: ['info depth 1 score cp 1 pv e2e4', 'bestmove e2e4'],
            stderr: [],
          },
        }));
      }
    }

    terminate() {}
  }

  const previousWorker = globalThis.Worker;
  globalThis.Worker = MockWorker;
  try {
    const engine = new RecklessEngine({}, '/reckless/reckless-simd128-external.wasm', {
      nnueUrl: '/reckless/reckless-v60-7f587dfb.nnue',
      forceOneShot: true,
    });
    assert.equal(await engine.bestMove(FEN), 'e2e4');
    const run = MockWorker.messages.find((message) => message.type === 'run');
    assert.deepEqual(run.preopenFiles, [{ name: RECKLESS_EXTERNAL_NNUE_FILE, url: '/reckless/reckless-v60-7f587dfb.nnue' }]);
    engine.dispose();
  } finally {
    if (previousWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = previousWorker;
  }
});
