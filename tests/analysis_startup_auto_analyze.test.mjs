import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { untilAborted } from '../src/lc0/modelCache.ts';

// The end of Analysis init(), after the startup asset checks have been awaited.
const text = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const initStart = text.indexOf('async function init(mountSignal: AbortSignal)');
const anchor = '  if (!isV0DeployProfile()) void refreshCentipawnHybridManifestStatus();\n';
const tailStart = text.indexOf(anchor, initStart) + anchor.length;
const tail = text.slice(tailStart, text.indexOf('\n}\n', tailStart));

function startup({ autoAnalyze, smallLc0 }) {
  const calls = [];
  const element = () => ({
    textContent: '',
    disabled: true,
    toggleAttribute(_name, on) {
      this.disabled = on;
    },
  });
  const elements = new Map([
    ['analyze', element()],
    ['message', element()],
  ]);
  const deps = {
    automaticAnalysisAllowed: () => true,
    activeEngineRows: () => [smallLc0 ? { family: 'lc0', variant: 'small' } : { family: 'sf', variant: 'lite' }],
    isLc0BigNetVariant: () => false,
    inputEl: (id) => (assert.equal(id, 'autoAnalyze'), { checked: autoAnalyze }),
    el: (id) => elements.get(id),
    selectEl: () => ({ disabled: true }),
    loadLc0Backend: async (runAutoAnalyze) => calls.push(`backend:${runAutoAnalyze}`),
    analyzeCurrent: async () => calls.push('analyze'),
    mountSignal: new AbortController().signal,
  };
  const body = stripTypeScriptTypes(`async function run(){${tail}\n}`);
  const run = new Function(...Object.keys(deps), `return (${body})();`);
  return run(...Object.values(deps)).then(() => ({ calls, analyzeDisabled: elements.get('analyze').disabled }));
}

test('Analysis yields before the initial render and skips it if the route unmounts', async () => {
  const start = text.indexOf('  await Promise.all([...new Set(engineRows.map(', initStart);
  const end = text.indexOf('  installRuntimeAuditPanel();', start);
  const body = stripTypeScriptTypes(`async function render(){${text.slice(start, end)}}`);
  const mount = new AbortController();
  const calls = [];
  const timers = [];
  const render = new Function(
    'deps',
    `const { mountSignal, resolveAnalysisCpuDefault, setTimeout, renderEngineProfiles } = deps;
    const engineRows = [], lastProfile = undefined, isStaleMount = (signal) => signal.aborted;
    return ${body};`,
  )({
    mountSignal: mount.signal,
    resolveAnalysisCpuDefault: assert.fail,
    setTimeout: (callback) => timers.push(callback),
    renderEngineProfiles: () => calls.push('render'),
  });
  const pending = render();
  await Promise.resolve();
  assert.deepEqual(calls, []);
  assert.equal(timers.length, 1);
  mount.abort();
  timers[0]();
  await pending;
  assert.deepEqual(calls, []);
});

test('a saved setup without Lc0 Small honours Auto-analyze turned off during startup checks', async () => {
  assert.ok(tailStart > initStart, 'init() startup tail found');
  const off = await startup({ autoAnalyze: false, smallLc0: false });
  assert.deepEqual(off.calls, [], 'no automatic request');
  assert.equal(off.analyzeDisabled, false, 'manual Analyze is enabled');
  const on = await startup({ autoAnalyze: true, smallLc0: false });
  assert.deepEqual(on.calls, ['analyze']);
});

test('a saved setup with Lc0 Small dispatches analysis before backend readiness when Auto-analyze is on', async () => {
  assert.deepEqual((await startup({ autoAnalyze: false, smallLc0: true })).calls, ['backend:true']);
  assert.deepEqual((await startup({ autoAnalyze: true, smallLc0: true })).calls, ['analyze']);
});

for (const outcome of ['Lc0 first line', 'stalled Lc0', 'cancelled']) {
  test(`Stockfish Lite cold startup: ${outcome}`, async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const gateStart = text.indexOf('async function waitForSmallFirstLine(');
    const gate = gateStart < 0 ? '' : text.slice(gateStart, text.indexOf('async function analyzeCurrent(', gateStart));
    const createStart = text.indexOf('    let releaseSmallFirstLine');
    const create = text.slice(createStart < 0 ? text.indexOf('    const smallFirstLine =') : createStart, text.indexOf('    const tasks: Promise<void>[]'));
    const sfStart = text.indexOf('pushTask(index, label, async (beginSearch) => {', text.indexOf("} else if (row.family === 'sf') {", text.indexOf(create)));
    const waitStart = text.indexOf('\n', sfStart);
    const waitEnd = text.indexOf('          return engines.withCpuLease(', waitStart);
    const start = stripTypeScriptTypes(`${gate}
      function setup() {
        const runRows = [{ family: 'lc0', variant: 'small' }, { family: 'sf', variant: 'lite' }];
        const needsSmallLc0Backend = true, smallLc0Pending = true, workerReady = false, workerSearchWarm = false;
        ${create}
        return { publish: () => releaseSmallFirstLine(), async stockfish() { ${text.slice(waitStart, waitEnd)} boot(); } };
      }`);
    const controller = new AbortController();
    const boot = t.mock.fn();
    const api = new Function('untilAborted', 'controller', 'boot', 'Promise', `${start}; return setup();`)(
      untilAborted,
      controller,
      boot,
      outcome === 'stalled Lc0'
        ? Promise
        : class SafariPromise extends Promise {
            static withResolvers = undefined;
          },
    );
    const running = api.stockfish();
    await new Promise(setImmediate);
    assert.equal(boot.mock.callCount(), 0);
    if (outcome === 'cancelled') {
      controller.abort();
      await assert.rejects(running, { name: 'AbortError' });
      assert.equal(boot.mock.callCount(), 0);
    } else {
      if (outcome === 'Lc0 first line') api.publish();
      else {
        t.mock.timers.tick(2999);
        await new Promise(setImmediate);
        assert.equal(boot.mock.callCount(), 0);
        t.mock.timers.tick(1);
      }
      await new Promise(setImmediate);
      assert.equal(boot.mock.callCount(), 1);
      await running;
    }
  });
}

test('Analysis fetches the opening index once when the position leaves the starting board', async () => {
  const start = text.indexOf('let openingNameIndex:');
  const controller = text.slice(start, text.indexOf('let tree =', start));
  const tree = { root: { fen: 'start' } };
  tree.current = tree.root;
  const label = {};
  let requests = 0;
  const deps = {
    tree,
    START_FEN: 'start',
    retry: false,
    mountAbort: new AbortController(),
    el: () => label,
    fetch: async () => {
      requests++;
      return Response.json({ named: ['B00', 'King’s Pawn Game'] });
    },
    resolvePublicAssetUrl: (url) => url,
    isStaleMount: () => false,
    openingNameAt: (index) => index.named,
    fitMoveList() {},
    keepCurrentMoveInView() {},
  };
  const render = new Function(...Object.keys(deps), `${stripTypeScriptTypes(controller)}; return renderOpeningName;`)(...Object.values(deps));
  render();
  assert.equal(requests, 0);
  tree.current = { fen: 'after e4' };
  render();
  render();
  await new Promise(setImmediate);
  assert.equal(requests, 1);
  assert.equal(label.textContent, 'B00 · King’s Pawn Game');
});
