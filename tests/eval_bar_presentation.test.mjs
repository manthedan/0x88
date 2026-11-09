import assert from 'node:assert/strict';
import { test } from 'node:test';
import { evalDetailsMarkup, evalEngineTag, evalMarkerMarkup } from '../src/lc0/evalBarPresentation.ts';
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
test('mate lines stay on the true endpoint even when the labelled count shifts inward', () => {
  const ticks = [{ ...marker('Stockfish Lite', 0), mateIn: 1, scoreText: '#1' }, { ...marker('PlentyChess', 0), mateIn: 2, scoreText: '#2' }];
  const html = evalMarkerMarkup(ticks, 'Stockfish Lite', desktop);
  assert.equal(evalPercent(ticks[0]), 100);
  assert.match(html, /--pos:100%/);
  assert.match(html, /--badge-shift:-23px/);
  assert.match(html, /Stockfish Lite #1/);
  assert.match(html, /PlentyChess #2/);
});
