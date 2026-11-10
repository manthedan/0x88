import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { evalDetailsMarkup, evalEngineTag, evalMarkerMarkup } from '../src/lc0/evalBarPresentation.ts';

const marker = (name, cp, color = '#1f5fcf') => ({ id: name, name, scoreCp: cp, color, scoreText: `${cp < 0 ? '' : '+'}${(cp / 100).toFixed(2)}` });
const desktop = { pixels: 680, mobile: false };
const mobile = { pixels: 326, mobile: true };

test('evaluation tags share the canonical Lc0 token', () => {
  assert.equal(evalEngineTag('Lc0 Small'), 'Lc0');
  assert.equal(evalEngineTag('Stockfish Lite'), 'SF');
  assert.equal(evalEngineTag('PlentyChess'), 'PC');
});
test('every engine keeps a plain tick even at exactly the same value', () => {
  const html = evalMarkerMarkup(
    ['Lc0 Small', 'Stockfish Lite', 'PlentyChess'].map((n) => marker(n, 20)),
    'Lc0 Small',
    desktop,
  );
  assert.equal((html.match(/class="eval-tick"/g) ?? []).length, 3);
  assert.doesNotMatch(html, /button|count|grouped|segment/);
  assert.equal((html.match(/class="eval-engine-label/g) ?? []).length, 1);
  assert.match(html, /class="eval-chosen-mark"[^>]*><\/span>$/);
});
test('desktop retains the selected label; mobile uses the source chip instead', () => {
  for (const layout of [desktop, mobile]) {
    const html = evalMarkerMarkup([marker('Lc0 Small', 20)], 'Lc0 Small', layout);
    if (layout.mobile) assert.doesNotMatch(html, /eval-engine-label/);
    else assert.match(html, /class="eval-engine-label selected"[^>]*>Lc0</);
    assert.match(html, /eval-chosen-mark/);
  }
});
test('only meaningful, separated range ends get secondary labels', () => {
  const ticks = [marker('Lc0 Small', 20), marker('Stockfish Lite', 21), marker('PlentyChess', 220)];
  const html = evalMarkerMarkup(ticks, 'Lc0 Small', desktop);
  assert.match(html, />PC</);
  assert.doesNotMatch(html, />SF</);
  assert.doesNotMatch(evalMarkerMarkup(ticks.slice(0, 2), 'Lc0 Small', desktop), />SF</);
});
test('value list is sorted by eval with names, values and selected identity only', () => {
  const html = evalDetailsMarkup([marker('Lc0 Small', 180), marker('Stockfish Lite', -20)], 'Lc0 Small');
  assert.ok(html.indexOf('Stockfish Lite') < html.indexOf('Lc0 Small'));
  assert.match(html, /Shown on bar/);
  assert.match(html, /-0.20/);
  assert.doesNotMatch(html, /depth|nodes|nps|nonlinear/);
  assert.match(evalDetailsMarkup([], undefined), /Waiting for this position/);
});
test('mate ticks stay pinned to true endpoints without badges or displaced labels', () => {
  const html = evalMarkerMarkup(
    [
      { ...marker('Lc0 Small', 0), mateIn: 1 },
      { ...marker('Stockfish Lite', 0), mateIn: -1 },
    ],
    'Lc0 Small',
    desktop,
  );
  assert.match(html, /--pos:100%/);
  assert.match(html, /--pos:0%/);
  assert.doesNotMatch(html, /shift|count/);
});
test('the stable whole-bar invoker owns native focus ordering and survives one-engine refreshes', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  assert.match(source, /showPopover as \(options: \{ source: HTMLElement \}\) => void\)\(\{ source: rail \}\)/);
  assert.match(source, /if \(details.dataset.fen !== tree.current.fen\) \{/);
  assert.doesNotMatch(source, /details.dataset.fen !== tree.current.fen \|\| choices.length/);
  assert.match(source, /newState === 'closed'\) \{\s*restoringEvalFocus = true;\s*restoreEvalFocus =/);
  assert.match(source, /if \(restoreEvalFocus && rail.isConnected\) rail.focus\(\{ preventScroll: true \}\);/);
  assert.match(source, /if \(restoringEvalFocus \|\| retry/);
});

test('all engine text is escaped', () => {
  const tick = marker('Stockfish <img src=x onerror="alert(1)">', 20);
  const html = evalMarkerMarkup([tick], tick.id, desktop) + evalDetailsMarkup([tick], tick.id);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img/);
});
