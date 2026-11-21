import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { arenaMoveDelay, guardArenaUnload } from '../src/lc0/arenaNavigation.ts';
import { subscribeAssetProbeRetry } from '../src/lc0/assetProbe.ts';
import { BT4_NET, bigNetAssetStatusSync, checkBigNetAsset, T3_NET } from '../src/lc0/bt4Engine.ts';

test('Arena leave confirmation reports checkpoint failure instead of promising saved results', () => {
  const route = readFileSync(new URL('../src/routes/app/arena/+page.svelte', import.meta.url), 'utf8');
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const getter = source.includes('export function arenaLeaveMessage()')
    ? source.slice(source.indexOf('export function arenaLeaveMessage()'), source.indexOf('export function mountArenaBrowser()')).replaceAll('export ', '')
    : '';
  for (const problem of ['', 'Not saved: IndexedDB unavailable']) {
    const code = `
      const checkpointProblem = problem;
      ${stripTypeScriptTypes(getter)}
      ${stripTypeScriptTypes(route.slice(route.indexOf('let mountedUrl:'), route.indexOf('const title =')))}
      isRunActive = () => true;
      if (typeof leaveMessage !== 'undefined') leaveMessage = arenaLeaveMessage;
      navigate({cancel() {}, willUnload: false});
    `;
    let navigate;
    let message;
    new Function('problem', 'window', 'beforeNavigate', 'afterNavigate', 'navigate', code)(
      problem,
      {
        confirm(text) {
          message = text;
          return false;
        },
      },
      (callback) => {
        navigate = callback;
      },
      () => {},
      (event) => navigate(event),
    );
    assert.match(message, problem ? /export.*PGN.*before leaving/i : /Completed games are saved/);
    if (problem) assert.doesNotMatch(message, /Completed games are saved/);
  }
});

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

test('query-only Arena navigation is cancelled before the URL changes and confirmed reload bypasses unload', () => {
  const route = readFileSync(new URL('../src/routes/app/arena/+page.svelte', import.meta.url), 'utf8');
  for (const confirmed of [false, true]) {
    let navigate;
    const events = [];
    const target = new URL('https://example.test/app/arena/');
    const code = `
      ${stripTypeScriptTypes(route.slice(route.indexOf('let mountedUrl:'), route.indexOf('const title =')))}
      mountedUrl = new URL('https://example.test/app/arena/?dev');
      isRunActive = () => true;
      leaveMessage = () => 'Leave?';
      if (typeof allowLeave !== 'undefined') allowLeave = () => events.push('allow-unload');
      invoke({to: {url: target}, willUnload: false, cancel() {events.push('cancel');}});
    `;
    new Function('URL', 'target', 'events', 'window', 'location', 'beforeNavigate', 'afterNavigate', 'invoke', code)(
      URL,
      target,
      events,
      {
        confirm() {
          events.push('confirm');
          return confirmed;
        },
      },
      {
        assign(url) {
          events.push(url);
        },
      },
      (callback) => {
        navigate = callback;
      },
      () => {},
      (event) => navigate(event),
    );
    assert.deepEqual(events, confirmed ? ['cancel', 'confirm', 'allow-unload', target.href] : ['cancel', 'confirm']);
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

function arenaAvailability(prefix, extra = {}) {
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('async function refreshBigNetAvailability('), source.indexOf('async function renderRuntimeBadge('));
  const refreshStart = source.slice(source.indexOf('function refreshSeatControls('), source.indexOf("  selectEl('seatA').disabled"));
  const BIG_NETS = {
    bt4: { ...BT4_NET, modelUrl: `https://example.test/${prefix}-bt4.onnx` },
    t3: { ...T3_NET, modelUrl: `https://example.test/${prefix}-t3.onnx` },
  };
  const deps = {
    BIG_NETS,
    checkBigNetAsset,
    bigNetAssetStatusSync,
    subscribeAssetProbeRetry,
    probeBt4Support: async () => true,
    mountAbort: new AbortController(),
    isStaleMount: (signal) => signal.aborted,
    whileArenaMounted: (callback) => callback,
    activeSeatRows: () => [],
    el: () => ({}),
    document: { activeElement: null },
    buildEngines() {},
    renderSeatSelectors() {},
    ...extra,
  };
  const load = new Function(
    ...Object.keys(deps),
    `
    let bt4AvailabilityResolved = false, probingSeatOptions = false, running = false, startPending = false, seatAvailabilityDirty = false;
    ${stripTypeScriptTypes(body)}
    ${stripTypeScriptTypes(refreshStart + '}')}
    return { refresh: refreshBigNetAvailability, resolved: () => bt4AvailabilityResolved,
      start: () => { running = true; }, finish: () => { running = false; refreshSeatControls(); } };
  `,
  )(...Object.values(deps));
  return { ...load, BIG_NETS };
}

test('Arena retries a transient large-net probe without another picker interaction', async (t) => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'] });
  const api = arenaAvailability('retry');
  const urls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    urls.push(url);
    return new Response(null, { status: urls.length === 1 ? 503 : 200 });
  });
  await api.refresh();
  assert.equal(api.resolved(), false, 'a transient failure is not a resolved probe');
  assert.equal(bigNetAssetStatusSync(api.BIG_NETS.bt4), 'unknown');
  t.mock.timers.tick(1_000);
  await new Promise(setImmediate);
  assert.deepEqual(urls, [api.BIG_NETS.bt4.modelUrl, api.BIG_NETS.t3.modelUrl, api.BIG_NETS.bt4.modelUrl]);
  assert.equal(api.resolved(), true);
  assert.equal(bigNetAssetStatusSync(api.BIG_NETS.bt4), 'present');
});

test('Arena applies availability that finished during a match when the match ends', async (t) => {
  const response = Promise.withResolvers();
  const head = t.mock.method(globalThis, 'fetch', () => response.promise.then(() => new Response(null, { status: 200 })));
  const renderSeatSelectors = t.mock.fn();
  const api = arenaAvailability('match', { renderSeatSelectors });
  const probing = api.refresh();
  api.start();
  response.resolve();
  await probing;
  assert.equal(renderSeatSelectors.mock.callCount(), 0);
  api.finish();
  assert.equal(renderSeatSelectors.mock.callCount(), 1);
  assert.equal(head.mock.callCount(), 2, 'apply results without reprobing');
});

test('Arena restores focus to the same seat control after availability changes', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 200 }));
  const label = 'Engine 1 version';
  const original = { getAttribute: (name) => (name === 'aria-label' ? label : null) };
  const document = { activeElement: original };
  const replacement = {
    focus: () => {
      document.activeElement = replacement;
    },
  };
  const host = {
    contains: (element) => element === original,
    querySelector: (selector) => (selector === `[aria-label="${label}"]` ? replacement : null),
  };
  const api = arenaAvailability('focus', {
    document,
    CSS: { escape: (value) => value },
    el: () => host,
    renderSeatSelectors: () => {
      document.activeElement = null;
    },
  });
  await api.refresh();
  assert.equal(document.activeElement, replacement);
});

test('Arena unload confirmation is consumed once and resets after bfcache or aborted navigation', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
  const setup = source.slice(
    source.indexOf('let arenaUnloadConfirmed'),
    source.indexOf('  void init(controller.signal).catch', source.indexOf('export function mountArenaBrowser()')),
  );
  for (const reason of ['consumed', 'bfcache', 'timeout', 'pointerdown', 'keydown']) {
    const window = new EventTarget();
    const code = `
      let lastArenaMountHref = null, mountAbort, arenaMountedCallbackCache;
      const pageDom = { mount: () => () => {} }, document = { getElementById: () => ({}) };
      const location = {href: 'https://example.test/app/arena/?dev'};
      const resetArenaPageState = () => {}, isArenaRunActive = () => true;
      ${stripTypeScriptTypes(setup.replaceAll('export ', '') + 'return () => controller.abort(); }')}
      return {allow: allowArenaUnload, cleanup: mountArenaBrowser()};
    `;
    const { allow, cleanup } = new Function('window', 'guardArenaUnload', code)(window, guardArenaUnload);
    const unload = () => {
      const event = new Event('beforeunload', { cancelable: true });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    };
    assert.equal(unload(), true);
    allow();
    if (reason === 'consumed') assert.equal(unload(), false, 'Confirmed navigation must bypass exactly one unload');
    else if (reason === 'bfcache') {
      const event = new Event('pageshow');
      event.persisted = true;
      window.dispatchEvent(event);
    } else if (reason === 'timeout') t.mock.timers.tick(2000);
    else window.dispatchEvent(new Event(reason));
    assert.equal(unload(), true, reason + ' must restore the active-game guard');
    cleanup();
    assert.equal(unload(), false, 'Unmount removes the guard');
  }
});
