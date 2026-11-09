import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { evalDetailsMarkup, evalEngineTag, evalMarkerGroups, evalMarkerMarkup } from '../src/lc0/evalBarPresentation.ts';
import { evalPercent, evalTickBadgeShift, groupEvalTicks } from '../src/lc0/evalCompare.ts';

const marker = (name, cp, color = '#1f5fcf') => ({ id: name, name, scoreCp: cp, color, scoreText: `+${(cp / 100).toFixed(2)}`, detail: 'depth 12' });
const desktop = { pixels: 680, mobile: false, left: 120, viewport: 1440 };
const mobile = { pixels: 356, mobile: true, left: 16, viewport: 390 };

test('evaluation tags use Lc0 rather than the ambiguous L0', () => {
  assert.equal(evalEngineTag('Lc0 Small'), 'Lc0');
  assert.equal(evalEngineTag('Stockfish Lite'), 'SF');
  assert.equal(evalEngineTag('PlentyChess'), 'PC');
});
test('desktop range ends identify their engines independently of colour', () => {
  const html = evalMarkerMarkup([marker('Lc0 Small', 20), marker('PlentyChess', 180)], 'Lc0 Small', desktop);
  assert.match(html, /class="eval-edge-tag"[^>]*>Lc0</);
  assert.match(html, /class="eval-edge-tag"[^>]*>PC</);
  assert.match(html, /PC · PlentyChess \+1.80/);
});
test('group counts preserve every colour and draw the chosen tick separately, last', () => {
  const ticks = [marker('Lc0 Small', 20, '#7a4fd6'), marker('Stockfish Lite', 21), marker('PlentyChess', 22, '#007a87')];
  const html = evalMarkerMarkup(ticks, 'Lc0 Small', desktop);
  assert.equal((html.match(/class="eval-tick /g) ?? []).length, 1);
  assert.match(html, /class="eval-tick-count"[^>]*><b>3<\/b>/);
  for (const tick of ticks) assert.ok(html.includes(`background:${tick.color}`));
  assert.match(html, /class="eval-chosen-mark"[^>]*--pos:/);
  assert.ok(html.lastIndexOf('eval-chosen-mark') > html.lastIndexOf('</button>'));
});
test('every marker opens the same compact value list without changing source or tabs', () => {
  const html = evalMarkerMarkup([marker('Lc0 Small', 20), marker('Stockfish Lite', 180)], 'Lc0 Small', mobile);
  assert.equal((html.match(/popovertarget="evalCompareDetails"/g) ?? []).length, 2);
  assert.equal((html.match(/aria-haspopup="dialog"/g) ?? []).length, 2);
  assert.doesNotMatch(html, /data-eval-source|data-view|eval-edge-tag/);
});
test('tap details contain all exact values, names, effort and chosen-source identity', () => {
  const html = evalDetailsMarkup([marker('Lc0 Small', 20), marker('Stockfish Lite', 180)], 'Lc0 Small');
  assert.match(html, /Lc0 Small/);
  assert.match(html, /Stockfish Lite/);
  assert.match(html, /\+0.20/);
  assert.match(html, /\+1.80/);
  assert.match(html, /depth 12 · shown on bar/);
  assert.match(evalDetailsMarkup([], undefined), /Waiting for this position/);
});
test('rendered engine text and focus keys cannot inject markup', () => {
  const tick = marker('Stockfish <img src=x onerror="alert(1)">', 20);
  const html = evalMarkerMarkup([tick], tick.id, desktop) + evalDetailsMarkup([tick], tick.id);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img/);
  assert.match(html, /&quot;/);
});
test('focused engine key survives a resize that merges or splits tick groups', () => {
  const ticks = [marker('Lc0 Small', 25), marker('Stockfish Lite', 59), marker('PlentyChess', 22)];
  const html = evalMarkerMarkup(ticks, 'Lc0 Small', { ...mobile, focusedId: 'Stockfish Lite' });
  assert.match(html, /data-focus-key="Stockfish Lite"/);
  const larger = evalMarkerMarkup(ticks, 'Lc0 Small', { ...desktop, focusedId: 'Stockfish Lite' });
  assert.match(larger, /data-focus-key="Stockfish Lite"/);
});
test('new larger counters use the same collision and endpoint footprint as their renderer', () => {
  const ticks = [-10000, -620, -610, -510, 20, 21, 180, 10000].map((cp, index) => marker(String(index), cp));
  for (const [pixels, footprint] of [[356, 20], [680, 46]]) {
    const groups = groupEvalTicks(ticks, '4', pixels, 6, footprint);
    for (const group of groups.filter((group) => group.members.length > 1)) {
      const center = group.percent * pixels / 100 + evalTickBadgeShift(group.percent, pixels, group.selected, footprint);
      assert.ok(center - footprint / 2 >= -0.001);
      assert.ok(center + footprint / 2 <= pixels + 0.001);
    }
  }
});
test('clamped singleton endpoint labels cannot obscure nearby extreme ticks', () => {
  const ticks = [marker('Lc0 Small', 980), marker('Stockfish Lite', 1160), { ...marker('PlentyChess', 0), mateIn: 1, scoreText: '#1' }];
  const groups = evalMarkerGroups(ticks, 'Lc0 Small', desktop);
  assert.equal(groups.length, 1, 'the clamped mate tag must not cover the +11.60 tick');
  assert.equal(groups[0].members.length, 3);
  const negative = ticks.map((tick) => ({ ...tick, scoreCp: -tick.scoreCp, ...(tick.mateIn !== undefined ? { mateIn: -tick.mateIn } : {}) }));
  assert.equal(evalMarkerGroups(negative, 'Lc0 Small', desktop).length, 1, 'protect the negative endpoint too');
});
test('clamped endpoint tags clear their own unselected true tick on either end', () => {
  for (const sign of [1, -1]) {
    const ticks = [marker('Lc0 Small', 0), marker('Stockfish Lite', sign * 1200)];
    const groups = evalMarkerGroups(ticks, 'Lc0 Small', desktop);
    assert.equal(groups.length, 2);
    const html = evalMarkerMarkup(ticks, 'Lc0 Small', desktop);
    const position = Number(html.match(/--edge:([\d.]+)px;--edge-color:[^>]+>SF</)[1]);
    const point = evalPercent(ticks[1]) * desktop.pixels / 100;
    assert.ok(Math.abs(position - point) >= 9.5, '12px tag + 5px tick + 1px gap');
    assert.ok(position >= 7 && position <= desktop.pixels - 7);
  }
});
test('coincident mate endpoint tags clear their own count, tick and each other', () => {
  const ticks = [{ ...marker('Stockfish Lite', 0), mateIn: 1 }, { ...marker('PlentyChess', 0), mateIn: 2 }];
  const html = evalMarkerMarkup(ticks, 'Stockfish Lite', desktop);
  const positions = [...html.matchAll(/--edge:([\d.]+)px/g)].map((match) => Number(match[1]));
  assert.equal(positions.length, 2);
  const point = desktop.pixels;
  const count = point + evalTickBadgeShift(100, point, true, 46);
  for (const position of positions) {
    assert.ok(Math.abs(position - point) >= 10);
    assert.ok(Math.abs(position - count) >= 16);
  }
  assert.ok(Math.abs(positions[0] - positions[1]) >= 13);
});

test('a sole completed result among pending engines needs only one endpoint tag', () => {
  const html = evalMarkerMarkup([marker('Lc0 Small', 980)], 'Lc0 Small', desktop);
  assert.equal((html.match(/class="eval-edge-tag"/g) ?? []).length, 1);
});

test('the open value list refreshes markers while reusing its native invoker object', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const render = source.slice(source.indexOf('function renderEvalBar()'), source.indexOf('function evalSourceChoices('));
  assert.match(render, /next\.replaceWith\(evalDetailsInvoker\);\s*tickContainer\.replaceChildren\(template.content\)/);
  assert.match(render, /details\.matches\(':popover-open'\) \? evalDetailsInvoker\?\.dataset.focusKey/);
  assert.ok(render.indexOf('details.dataset.fen !== tree.current.fen') < render.indexOf('const markerMarkup'), 'position changes close the list before refreshing markers');
  assert.match(source, /addEventListener\('beforetoggle',[\s\S]*?newState === 'open'[\s\S]*?evalDetailsInvoker = active[\s\S]*?signal: controller.signal/);
  assert.match(source, /addEventListener\('toggle',[\s\S]*?newState === 'closed'[\s\S]*?evalDetailsInvoker = null;\s*renderEvalBar\(\)/);
});
test('a newly completed default source updates the chosen notch without losing the invoking engine key', () => {
  const ticks = [marker('Lc0 Small', -200, '#7a4fd6'), marker('Stockfish Lite', 20)];
  const html = evalMarkerMarkup(ticks, 'Lc0 Small', { ...mobile, focusedId: 'Stockfish Lite' });
  assert.match(html, /data-focus-key="Stockfish Lite"/);
  assert.ok(html.includes(`class="eval-chosen-mark" aria-hidden="true" style="--pos:${evalPercent(ticks[0])}%;--tick-color:#7a4fd6"`));
  assert.doesNotMatch(html, /Stockfish Lite \+0.20 \(shown on bar\)/);
});

test('mate lines stay on the true endpoint even when the labelled count shifts inward', () => {
  const ticks = [{ ...marker('Stockfish Lite', 0), mateIn: 1, scoreText: '#1' }, { ...marker('PlentyChess', 0), mateIn: 2, scoreText: '#2' }];
  const html = evalMarkerMarkup(ticks, 'Stockfish Lite', desktop);
  assert.equal(evalPercent(ticks[0]), 100);
  assert.match(html, /--pos:100%/);
  assert.match(html, /--badge-shift:-23px/);
  assert.match(html, /Stockfish Lite #1/);
  assert.match(html, /PlentyChess #2/);
});
