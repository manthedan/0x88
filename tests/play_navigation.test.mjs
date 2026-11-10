import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';

test('Play bfcache restoration retires temporary resign and new-game confirmations', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const wiring = source.slice(source.indexOf("  ctx.autosave = createWorkspaceAutosave('play'"), source.indexOf("  trackListener(ctx, el('newGame'), 'click'"));
  const disarmResign = source.slice(source.indexOf('function ctxDisarmResign('), source.indexOf('/** Icon buttons'));
  const disarmNewGame = source.slice(source.indexOf('function ctxDisarmNewGame('), source.indexOf('function ctxExportPgn('));
  const window = new EventTarget(),
    document = new EventTarget();
  const ctx = { resignArmed: true, newGameArmed: true, resignArmTimer: setTimeout(() => {}, 4000), newGameArmTimer: setTimeout(() => {}, 4000) };
  const code = `
    let completeHandoff, renders = 0;
    const elements = new Map();
    const el = key => {if (!elements.has(key)) elements.set(key, new EventTarget()); return elements.get(key);};
    const buttonEl = () => ({classList: {contains: () => true, remove() {}}}), setButtonLabel = () => {};
    const createWorkspaceAutosave = () => ({}), ctxRender = () => renders++;
    const trackListener = (ctx, target, event, callback) => target.addEventListener(event, callback);
    ${stripTypeScriptTypes(disarmResign + disarmNewGame + wiring)}
    return () => renders;
  `;
  const renders = new Function('ctx', 'window', 'document', code)(ctx, window, document);
  window.dispatchEvent(new Event('pageshow'));
  assert.equal(ctx.resignArmed, true, 'Ordinary pageshow does not reset a mounted session');
  const restored = new Event('pageshow');
  restored.persisted = true;
  window.dispatchEvent(restored);
  assert.equal(ctx.resignArmed, false, 'Returning to a cached page requires fresh resign confirmation');
  assert.equal(ctx.newGameArmed, false, 'Returning to a cached page requires fresh new-game confirmation');
  assert.equal(ctx.resignArmTimer, null);
  assert.equal(ctx.newGameArmTimer, null);
  assert.equal(renders(), 1, 'The restored controls must show their unarmed labels');
});
