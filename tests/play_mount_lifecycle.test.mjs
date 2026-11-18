// Play remounts without a page reload: each mount gets a fresh scope, root
// and note state, while the engine host outlives mounts. Behavioural tests on
// fake roots; the controller itself needs the WebGPU graph, so only its
// structural guarantees are checked from source at the end.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import test from 'node:test';
import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';
import { createRootDom } from '../src/lc0/browserDom.ts';
import { createMountScope } from '../src/lc0/mountScope.ts';
import { createPlayEngines } from '../src/lc0/play/playEngines.ts';
import { createEngineUiPort, setEngineNote } from '../src/lc0/play/playEngineUi.ts';
import { routeController } from '../src/lib/routeController.ts';

class FakeElement extends EventTarget {
  constructor(id = '', children = []) {
    super();
    Object.assign(this, { id, children, parent: null, textContent: '', innerHTML: '', hidden: false, dataset: {}, attributes: new Map() });
    const classes = new Set();
    this.classList = { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)), contains: (name) => classes.has(name) };
    for (const child of children) child.parent = this;
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  contains(node) {
    for (let current = node; current; current = current.parent) if (current === this) return true;
    return false;
  }
  querySelector(selector) {
    const match = /^\[id=(".*")\]$/.exec(selector);
    if (!match) return null;
    const id = JSON.parse(match[1]);
    for (const child of this.children) {
      if (child.id === id) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
  append(child) {
    child.parent = this;
    this.children.push(child);
  }
  remove() {
    if (!this.parent) return;
    this.parent.children.splice(this.parent.children.indexOf(this), 1);
    this.parent = null;
  }
}

const body = new FakeElement('body');
function mountRoot() {
  const root = new FakeElement(
    'main',
    ['engineNote', 'retryEngine', 'dlProgress', 'status'].map((id) => new FakeElement(id)),
  );
  body.append(root);
  return root;
}
const noteOf = (root) => root.querySelector('[id="engineNote"]').textContent;

/** What mountPlayBrowser builds per mount: scope, root lookups, note model state and the engine port. */
function playMount(root) {
  const scope = createMountScope(root, new URL('https://0x88.test/app/play'));
  const state = { dom: createRootDom(root), disposed: false, enginePreparationGeneration: 0, engineNote: '' };
  const ui = createEngineUiPort(state);
  scope.defer(() => {
    state.disposed = true;
  });
  return { scope, state, ui };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

test('a mount scope removes its listeners, cancels its timers and runs every cleanup once', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const target = new EventTarget();
  const scope = createMountScope(new FakeElement('main'), new URL('https://0x88.test/app/play'));
  const order = [];
  let clicks = 0,
    fired = 0;
  scope.listen(target, 'click', () => clicks++);
  scope.timeout(() => fired++, 1000);
  const cancelled = scope.timeout(() => fired++, 10);
  scope.clearTimeout(cancelled);
  scope.defer(() => order.push('first'));
  scope.defer(() => {
    throw new Error('second cleanup failed');
  });
  scope.defer(() => order.push('third'));
  target.dispatchEvent(new Event('click'));
  t.mock.timers.tick(10);
  assert.equal(fired, 0, 'a cleared timer never fires');
  assert.throws(() => scope.dispose(), /second cleanup failed/);
  assert.deepEqual(order, ['third', 'first'], 'newest first; a failing cleanup does not skip older ones');
  assert.equal(scope.live(), false);
  scope.dispose();
  assert.deepEqual(order, ['third', 'first'], 'dispose is idempotent');
  target.dispatchEvent(new Event('click'));
  t.mock.timers.tick(5000);
  assert.equal(clicks, 1, 'listeners are gone after dispose');
  assert.equal(fired, 0, 'timers never fire after dispose');
});

test('routeController gives every attach a fresh scope and never mounts a detached root', async (t) => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { href: 'https://0x88.test/app/play?model=/a.onnx' } });
  t.after(() => (previous ? Object.defineProperty(globalThis, 'location', previous) : delete globalThis.location));
  const scopes = [];
  const gate = deferred();
  let loads = 0;
  const attach = routeController(
    async () => {
      if (++loads === 1) await gate.promise;
      return (scope) => scopes.push(scope);
    },
    { label: 'play', statusId: 'status' },
  );
  const first = mountRoot();
  const detachFirst = attach(first);
  detachFirst();
  gate.resolve();
  await new Promise(setImmediate);
  assert.equal(scopes.length, 0, 'a load that lands after detach must not mount');

  const second = mountRoot();
  const detachSecond = attach(second);
  await new Promise(setImmediate);
  globalThis.location.href = 'https://0x88.test/app/play?model=/b.onnx';
  const third = mountRoot();
  const detachThird = attach(third);
  await new Promise(setImmediate);
  assert.equal(scopes.length, 2);
  assert.equal(scopes[0].root, second);
  assert.equal(scopes[1].root, third);
  assert.equal(scopes[0].url.searchParams.get('model'), '/a.onnx');
  assert.equal(scopes[1].url.searchParams.get('model'), '/b.onnx', 'a new query is read by the new mount, not frozen at module load');
  assert.notEqual(scopes[0].signal, scopes[1].signal);
  detachSecond();
  assert.equal(scopes[0].live(), false);
  assert.equal(scopes[1].live(), true);
  detachThird();

  const failing = routeController(
    async () => {
      throw new Error('chunk 404');
    },
    { label: 'play', statusId: 'status' },
  );
  const broken = mountRoot();
  failing(broken);
  const errors = t.mock.method(console, 'error', () => {});
  await new Promise(setImmediate);
  assert.match(broken.querySelector('[id="status"]').textContent, /Page failed to initialize: chunk 404/);
  assert.equal(errors.mock.callCount(), 1);
});

test('teardown after Svelte removed the DOM writes to the detached root, never the next mount', () => {
  const rootA = mountRoot();
  const a = playMount(rootA);
  a.ui.note('Freeing memory…');
  rootA.remove(); // Svelte removes effect DOM before teardown runs.
  const rootB = mountRoot();
  const b = playMount(rootB);
  // Teardown-time writes resolve against the captured, detached root.
  assert.doesNotThrow(() => setEngineNote(a.state, ''));
  assert.equal(noteOf(rootA), '');
  a.scope.dispose();
  // The reclaim wait reports after unmount: no throw, and no write anywhere.
  assert.doesNotThrow(() => a.ui.note('Freeing memory…'));
  assert.doesNotThrow(() => a.ui.progress('Lc0 BT4', 1, 2));
  assert.equal(noteOf(rootA), '');
  assert.equal(noteOf(rootB), '');
  assert.equal(b.state.engineNote, '');
  b.scope.dispose();
});

test('the engine note is model state: clearNote never reads the DOM back', () => {
  const root = mountRoot();
  const { state, ui, scope } = playMount(root);
  ui.note('Freeing memory…');
  assert.equal(state.engineNote, 'Freeing memory…');
  setEngineNote(state, 'Maia3 is ready.');
  ui.clearNote('Freeing memory…');
  assert.equal(noteOf(root), 'Maia3 is ready.', 'a newer note survives a stale retraction');
  ui.note('Freeing memory…');
  root.querySelector('[id="engineNote"]').textContent = 'something else drew here';
  ui.clearNote('Freeing memory…');
  assert.equal(state.engineNote, '');
  assert.equal(noteOf(root), '', 'retraction follows the model, not what the node happens to show');
  scope.dispose();
});

test('remount: the engine host is reused, the old mount is retired first, and late loads reach no root', async () => {
  const creates = [];
  let live = 0,
    peak = 0;
  const engines = createPlayEngines({
    loadCentipawn: () => assert.fail('unused'),
    createMaia3: (options) => {
      const gate = deferred();
      creates.push({ gate, options, disposed: false });
      return gate.promise.then(() => {
        live++;
        peak = Math.max(peak, live);
        const entry = creates.find((c) => c.options === options);
        return {
          modelLoad: { bytes: 1 },
          dispose: () => {
            entry.disposed = true;
            live--;
          },
        };
      });
    },
  });
  const rootA = mountRoot();
  const a = playMount(rootA);
  a.scope.defer(() => engines.unmounted(false));
  const loadA = engines.ensureMaia3(1500, a.ui).catch(() => undefined);
  assert.equal(noteOf(rootA), 'Loading Maia3 human model…');
  rootA.remove();
  a.scope.dispose();

  const rootB = mountRoot();
  const b = playMount(rootB);
  // As ctxVerifySelectedEngineReady does: retire everything else, then acquire.
  const prepareB = engines.retireUnusedEngines({ family: 'maia3' }).then(() => engines.ensureMaia3(1500, b.ui));
  await new Promise(setImmediate);
  assert.equal(creates.length, 1, "B's acquisition waits for A's terminal cleanup");
  creates[0].options.onProgress?.(1, 2);
  creates[0].gate.resolve(); // A's load lands after A unmounted.
  await loadA;
  await new Promise(setImmediate);
  assert.equal(creates[0].disposed, true, "A's late evaluator is disposed");
  assert.equal(creates.length, 2, 'B loads its own evaluator only after that');
  creates[1].gate.resolve();
  const evaluatorB = await prepareB;
  assert.ok(evaluatorB);
  assert.equal(peak, 1, 'never two Maia3 models resident');
  assert.equal(noteOf(rootA), 'Loading Maia3 human model…', 'nothing wrote to the unmounted root');
  assert.equal(rootA.querySelector('[id="dlProgress"]').innerHTML, '');
  assert.equal(noteOf(rootB), '', "B's own load cleared its note");
  assert.equal(await engines.ensureMaia3(1500, b.ui), evaluatorB, 'the host keeps serving the same evaluator to the new mount');
  b.scope.dispose();
  engines.unmounted(false);
});

test('an unmounted Play mount is collectable once its engine load settles', async () => {
  setFlagsFromString('--expose_gc');
  const gc = runInNewContext('gc');
  const engines = createPlayEngines({
    loadCentipawn: () => assert.fail('unused'),
    createMaia3: async () => ({ modelLoad: { bytes: 1 }, dispose() {} }),
  });
  const window = new EventTarget(); // outlives the mount, like the real window
  const mount = () => {
    const root = mountRoot();
    const a = playMount(root);
    a.scope.listen(window, 'pagehide', () => a.state.engineNote);
    a.scope.timeout(() => a.state.engineNote, 60_000);
    a.scope.defer(() => engines.unmounted(false));
    const settled = engines.ensureMaia3(1500, a.ui).catch(() => undefined);
    root.remove();
    a.scope.dispose();
    return { ref: new WeakRef(a.state), settled };
  };
  const { ref, settled } = mount();
  await settled;
  for (let i = 0; i < 5 && ref.deref(); i++) {
    await new Promise(setImmediate);
    gc();
  }
  assert.equal(ref.deref(), undefined, 'the engine host and scope must not retain a retired mount');
});

test('Lc0 Small is cached per model URL: a new ?model= retires the old session before loading', async () => {
  const events = [];
  const engines = createPlayEngines({
    loadCentipawn: () => assert.fail('unused'),
    loadLc0Small: async (url) => {
      events.push(`load ${url}`);
      return {
        evaluator: {
          dispose: async () => {
            events.push(`dispose ${url}`);
          },
        },
      };
    },
  });
  const ui = { current: () => () => true, mounted: () => true, note() {}, clearNote() {}, progress() {}, hideProgress() {} };
  const a = await engines.ensureLc0Small({ modelUrl: '/a.onnx', deviceSelection: false }, ui);
  assert.equal(await engines.ensureLc0Small({ modelUrl: '/a.onnx', deviceSelection: false }, ui), a);
  const b = await engines.ensureLc0Small({ modelUrl: '/b.onnx', deviceSelection: false }, ui);
  assert.notEqual(b, a);
  assert.deepEqual(events, ['load /a.onnx', 'dispose /a.onnx', 'load /b.onnx'], 'one Lc0 Small resident at a time');
  await engines.retireUnusedEngines();
});

function recordingLc0Small(events, disposals = new Map()) {
  return createPlayEngines({
    loadCentipawn: () => assert.fail('unused'),
    loadLc0Small: async (url, { deviceSelection, load }) => {
      load.signal.throwIfAborted();
      const name = `${url} ${deviceSelection ? 'device' : 'exact'}`;
      events.push(`load ${name}`);
      return {
        evaluator: {
          dispose: async () => {
            await disposals.get(name)?.promise;
            events.push(`dispose ${name}`);
          },
        },
      };
    },
  });
}
const quietUi = { current: () => () => false, mounted: () => true, note() {}, clearNote() {}, progress() {}, hideProgress() {} };

test('Lc0 Small is cached per request: the same URL with another device rule is a different session', async () => {
  const events = [];
  const engines = recordingLc0Small(events);
  // The default request and an explicit ?model= naming the default file share a URL.
  const device = await engines.ensureLc0Small({ modelUrl: '/a.onnx', deviceSelection: true }, quietUi);
  const exact = await engines.ensureLc0Small({ modelUrl: '/a.onnx', deviceSelection: false }, quietUi);
  assert.notEqual(exact, device);
  assert.deepEqual(events, ['load /a.onnx device', 'dispose /a.onnx device', 'load /a.onnx exact']);
  await engines.retireUnusedEngines();
});

test('a replacement retired before its predecessor finishes disposing still settles, and later starts load', async () => {
  const events = [];
  const slowDispose = deferred();
  const engines = recordingLc0Small(events, new Map([['/a.onnx exact', slowDispose]]));
  await engines.ensureLc0Small({ modelUrl: '/a.onnx', deviceSelection: false }, quietUi);
  // B replaces A and waits for A's session to go; B is retired before that happens.
  const b = engines.ensureLc0Small({ modelUrl: '/b.onnx', deviceSelection: false }, quietUi);
  const bSettled = b.then(
    () => 'loaded',
    (error) => error.name,
  );
  const retired = engines.retireUnusedEngines();
  slowDispose.resolve();
  const settles = (promise) => Promise.race([promise, new Promise((resolve) => setTimeout(() => resolve('hung'), 200))]);
  assert.equal(await settles(bSettled), 'AbortError', 'a retired replacement never loads');
  assert.notEqual(await settles(retired), 'hung', 'retirement must not wait on itself');
  const c = settles(engines.ensureLc0Small({ modelUrl: '/c.onnx', deviceSelection: false }, quietUi));
  assert.notEqual(await c, 'hung', 'later engine starts are not blocked');
  assert.deepEqual(events, ['load /a.onnx exact', 'dispose /a.onnx exact', 'load /c.onnx exact']);
  await settles(engines.retireUnusedEngines());
});

test('Maia3 and Centipawn are cached per resolved model URL: a new asset base retires the old session first', async () => {
  const events = [];
  const engines = createPlayEngines({
    loadCentipawn: async (spec) => {
      events.push(`load ${spec.onnx} ${spec.meta}`);
      return { evaluator: { evaluate: async () => ({}), dispose: async () => events.push(`dispose ${spec.onnx}`) } };
    },
    createMaia3: async ({ fallbackUrls }) => {
      events.push(`load ${fallbackUrls.join(',')}`);
      return { modelLoad: { bytes: 1 }, dispose: async () => events.push(`dispose ${fallbackUrls[0]}`) };
    },
  });
  const oldMaia = await engines.ensureMaia3(1500, quietUi, ['https://old/m.onnx', 'https://old/f.onnx']);
  assert.equal(await engines.ensureMaia3(1500, quietUi, ['https://old/m.onnx', 'https://old/f.onnx']), oldMaia);
  const newMaia = await engines.ensureMaia3(1500, quietUi, ['/m.onnx', '/f.onnx']);
  assert.notEqual(newMaia, oldMaia);
  assert.deepEqual(events, ['load https://old/m.onnx,https://old/f.onnx', 'dispose https://old/m.onnx', 'load /m.onnx,/f.onnx']);
  events.length = 0;
  await engines.retireUnusedEngines({ family: 'centipawn' });
  events.length = 0;
  const oldCp = await engines.ensureCentipawn(quietUi, { onnx: 'https://old/c.onnx', meta: 'https://old/c.json' });
  assert.equal(await engines.ensureCentipawn(quietUi, { onnx: 'https://old/c.onnx', meta: 'https://old/c.json' }), oldCp);
  const newCp = await engines.ensureCentipawn(quietUi, { onnx: '/c.onnx', meta: '/c.json' });
  assert.notEqual(newCp, oldCp);
  await engines.retireUnusedEngines();
  assert.deepEqual(events, ['load https://old/c.onnx https://old/c.json', 'dispose https://old/c.onnx', 'load /c.onnx /c.json', 'dispose /c.onnx']);
});

function sourceFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = `${dir}/${name}`;
    return statSync(path).isDirectory() ? sourceFiles(path) : /\.(ts|svelte)$/.test(name) ? [path] : [];
  });
}

test('location.reload sites only shrink, and Play has none', () => {
  // Ratchet: lower a count (or delete the row) when a reload goes; never raise one.
  const allowed = {
    'src/lc0/analysisBrowser.ts': 1,
    'src/lc0/arenaBrowser.ts': 1,
    'src/lc0/policyOnlyBrowser.ts': 1,
    'src/routes/app/analysis/+page.svelte': 1,
    'src/routes/single-engine/+page.svelte': 1,
  };
  const actual = {};
  for (const file of sourceFiles('src')) {
    const count = readFileSync(file, 'utf8').match(/location\.reload\(/g)?.length ?? 0;
    if (count) actual[file] = count;
  }
  for (const [file, count] of Object.entries(actual)) assert.ok(count <= (allowed[file] ?? 0), `${file}: ${count} location.reload call(s)`);
  assert.equal(actual['src/lc0/playBrowser.ts'], undefined);
  assert.equal(actual['src/routes/app/play/+page.svelte'], undefined);
});

test('Play keeps per-mount state off the module and registers listeners only through its scope', () => {
  const play = readFileSync('src/lc0/playBrowser.ts', 'utf8');
  const page = readFileSync('src/routes/app/play/+page.svelte', 'utf8');
  assert.doesNotMatch(play, /addEventListener\(|removeEventListener\(/, 'listeners go through scope.listen so dispose removes them');
  assert.doesNotMatch(play, /^const (params|pageDom|blockedModelOverride|LC0_SMALL_MODEL|playRuntimeFallbackAudits)\b/m);
  assert.doesNotMatch(play, /new URLSearchParams\(location\.search\)/, 'the URL is read per mount from scope.url');
  assert.doesNotMatch(page, /afterNavigate|onMount/);
  // Prerendering throws on url.search, so the key reads it only in the browser.
  assert.match(page, /\{#key browser \? page\.url\.search : ''\}/);
  assert.match(page, /\{@attach play\}/);
  assert.match(page, /routeController\(/);
});
