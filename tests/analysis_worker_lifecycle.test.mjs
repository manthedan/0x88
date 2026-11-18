import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { disposeOrtWorker } from '../src/lc0/disposeOrtWorker.ts';
import { terminateEngineWorker } from '../src/lc0/engineRetirement.ts';

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
    removeEventListener(type, listener) {
      this.listeners.set(
        type,
        (this.listeners.get(type) ?? []).filter((fn) => fn !== listener),
      );
    }
    emit(type, data) {
      for (const listener of this.listeners.get(type) ?? []) listener(data);
    }
    postMessage(message) {
      if (message.type === 'dispose') queueMicrotask(() => this.emit('message', { data: { id: message.id, type: 'disposed' } }));
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
    const trackEngineRetirement = (promise) => promise;
    const selectedLc0Runtime = () => 'onnx', lc0InitMessage = () => ({ type: 'init' });
    const publishBrowserRuntimeAudit = () => {}, lc0ResolvedRuntime = () => 'fake', multiPv = () => 1;
    const showModelProgress = () => {}, MODEL_URL = '/unused', LC0_WHOLE_MODEL_WEBGPU_RUNTIME = 'whole';
    const LC0_WHOLE_MODEL_MANIFEST_URL = '', PACK_URL = '';
    ${functions.replaceAll('import.meta.url', "'file:///test/analysisBrowser.ts'")}
    return { initWorker, postWorker, resetSearchWorker, snapshot: () => ({ workerReady, workerBackend, workerRuntime }) };
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  return { ...new Function('Worker', 'disposeOrtWorker', 'terminateEngineWorker', code)(Worker, disposeOrtWorker, terminateEngineWorker), workers };
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

// Leaving Analysis mid-load: Svelte removes the page's DOM before the mount cleanup dispatches
// DISPOSE, which cancels the running operation. That cancel must not throw on the missing
// elements, and must still stop the worker search and big-net inits.
function cancelController({ page, hybrid = false, reload = false }) {
  const slice = (start, end) => {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first, `${start} .. ${end}`);
    return source.slice(first, last);
  };
  const posted = [];
  const calls = [];
  const searchWorker = { postMessage: (message) => posted.push(message) };
  const controller = new AbortController();
  const document = {
    getElementById: (id) => page?.get(id) ?? null,
    createElement: () => ({}),
  };
  const code = ts.transpile(
    `
    function el(id) { const node = document.getElementById(id); if (!node) throw new Error('Missing #' + id); return node; }
    const maybeEl = (id) => document.getElementById(id), selectEl = el;
    const hasBoundedSearchProgress = () => false, searchProgressHtml = () => '';
    const renderLoadingProgress = (node, items) => { node.hidden = false; node.items = items; };
    const hideLoadingProgress = (node) => { node.hidden = true; };
    const searchProgressByEngine = new Map([['lc0', { completed: 1, requested: 4 }]]);
    const loadingProgressItems = new Map([['model', { id: 'model', label: 'Lc0 BT4' }]]);
    const mountAbort = new AbortController();
    if (!page) mountAbort.abort();
    const analysisOperations = new Map([[1, operation]]);
    let analysisAbort = reload ? null : operation, analysisReloadAbort = reload ? operation : null;
    let activeWorkerSearchId = 7, workerSearchWarm = !hybrid, workerRuntime = hybrid ? 'hybrid-small' : 'onnx';
    let searchWorker = worker;
    const ANALYSIS_BIG_NET_KEYS = ['bt4', 't3'];
    const peekBigNetSearcher = (key) => ({ cancel: () => calls.push('cancel ' + key) });
    const retireAnalysisRunCentipawnKeys = () => calls.push('retire centipawn');
    const disposeRuntimeResources = () => { calls.push('dispose runtime'); analysisAbort = null; };
    const resetSearchWorker = (reason) => { calls.push('reset worker: ' + reason); searchWorker = null; activeWorkerSearchId = null; };
    ${slice('function modelProgressEl(', 'const loadingProgressItems')}
    ${slice('function renderLoadingProgressItems(', 'function showModelProgress(')}
    ${slice('function hideModelProgress(', 'function storageGet(')}
    ${slice('function renderAnalysisSearchProgress(', 'function progressKey(')}
    ${slice('function cancelAnalysisOperation(', 'function scheduleAnalyzeCurrent(')}
    return { cancel: () => cancelAnalysisOperation(1), loadingProgressItems, searchProgressByEngine };
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  const api = new Function('document', 'page', 'operation', 'worker', 'calls', 'hybrid', 'reload', code)(
    document,
    page,
    controller,
    searchWorker,
    calls,
    hybrid,
    reload,
  );
  return { ...api, posted, calls, controller };
}

test('leaving Analysis mid-search cancels the worker search and big nets without touching the removed page', () => {
  const api = cancelController({ page: null });
  assert.doesNotThrow(() => api.cancel()); // threw "Missing #message" before the fix
  assert.equal(api.controller.signal.aborted, true);
  assert.deepEqual(api.posted, [{ type: 'cancel', target: 7 }]);
  assert.deepEqual(api.calls, ['retire centipawn', 'cancel bt4', 'cancel t3']);
  // Progress state is module-wide and must not leak into the next mount.
  assert.equal(api.loadingProgressItems.size, 0);
  assert.equal(api.searchProgressByEngine.size, 0);
});

test('leaving Analysis during hybrid init or a backend reload releases the engines without the page', () => {
  const hybrid = cancelController({ page: null, hybrid: true });
  assert.doesNotThrow(() => hybrid.cancel());
  assert.deepEqual(hybrid.calls, ['retire centipawn', 'reset worker: LC0 hybrid worker initialization cancelled', 'cancel bt4', 'cancel t3']);
  const reload = cancelController({ page: null, reload: true });
  assert.doesNotThrow(() => reload.cancel());
  assert.deepEqual(reload.calls, ['dispose runtime']);
});

test('a cancel on the mounted page still updates its controls and status', () => {
  const node = () => ({
    textContent: '',
    disabled: true,
    toggled: {},
    toggleAttribute(name, on) {
      this.toggled[name] = on;
    },
    insertAdjacentElement() {},
  });
  const page = new Map(['message', 'analyze', 'stop', 'backend', 'lc0RuntimeSelect', 'downloadProgress'].map((id) => [id, node()]));
  page.get('message').textContent = 'Freeing memory…';
  const api = cancelController({ page, hybrid: true });
  api.cancel();
  assert.equal(page.get('message').textContent, 'Analysis stopped during LC0 hybrid backend initialization.');
  assert.equal(page.get('analyze').toggled.disabled, false);
  assert.equal(page.get('stop').toggled.disabled, true);
  assert.equal(page.get('lc0RuntimeSelect').disabled, false);
  assert.equal(page.get('downloadProgress').hidden, true);
});
