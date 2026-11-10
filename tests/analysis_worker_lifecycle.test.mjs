import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

// Exercise the actual controller functions with only browser/transport effects
// stubbed. Importing the page would mount its DOM and load model assets.
const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const functions = source.slice(source.indexOf('function postWorker<T>'), source.indexOf('function analysisProgressText('));
function controller() {
  const workers = [];
  class Worker {
    listeners = new Map();
    terminated = false;
    constructor() {
      workers.push(this);
    }
    addEventListener(type, listener) {
      const listeners = this.listeners.get(type) ?? [];
      listeners.push(listener);
      this.listeners.set(type, listeners);
    }
    emit(type, data) {
      for (const listener of this.listeners.get(type) ?? []) listener(data);
    }
    postMessage(message) {
      if (message.type === 'init') queueMicrotask(() => this.emit('message', { data: { id: message.id, type: 'ready', backend: 'fake' } }));
    }
    terminate() {
      this.terminated = true;
    }
  }
  const code = ts.transpile(
    `
    let searchWorker = null, workerReady = false, workerBackend = '', workerRuntime = '', workerSeq = 0;
    let activeWorkerSearchId = null, workerSearchWarm = false, analysisWorkerProgressMessages = 0;
    const workerPending = new Map();
    const selectedLc0Runtime = () => 'onnx', lc0InitMessage = () => ({ type: 'init' });
    const publishBrowserRuntimeAudit = () => {}, lc0ResolvedRuntime = () => 'fake', multiPv = () => 1;
    const showModelProgress = () => {}, MODEL_URL = '/unused', LC0_WHOLE_MODEL_WEBGPU_RUNTIME = 'whole';
    const LC0_WHOLE_MODEL_MANIFEST_URL = '', PACK_URL = '';
    ${functions.replaceAll('import.meta.url', "'file:///test/analysisBrowser.ts'")}
    return { initWorker, postWorker, resetSearchWorker, snapshot: () => ({ workerReady, workerBackend, workerRuntime }) };
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  return { ...new Function('Worker', code)(Worker), workers };
}

for (const type of ['error', 'messageerror']) {
  test(`S01: Analysis post-init ${type} invalidates the backend and retry creates a new worker`, async () => {
    const api = controller();
    await api.initWorker();
    const worker = api.workers[0];
    const failed = assert.rejects(api.postWorker({ type: 'search' }), (error) => error.name !== 'AbortError' && /crash|decoded/.test(error.message));
    worker.emit(type, { message: 'injected Analysis crash' });
    // This assertion fails before the fix: workerReady remains true.
    assert.equal(api.snapshot().workerReady, false);
    assert.equal(worker.terminated, true);
    await failed;
    await api.initWorker();
    assert.equal(api.workers.length, 2);
    worker.emit('error', { message: 'stale crash' });
    assert.equal(api.snapshot().workerReady, true);
    api.resetSearchWorker('cleanup');
  });
}
