import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { arenaMoveDelay, guardArenaUnload } from '../src/lc0/arenaNavigation.ts';

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
