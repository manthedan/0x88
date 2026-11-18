import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const template = readFileSync(new URL('../src/routes/app/analysis/+page.svelte', import.meta.url), 'utf8');

test('the mobile toolbar wraps before the running status loses its width budget', () => {
  const wrapped = /@container \(max-width:(\d+)px\)\{\s*\.analysis-overview\{grid-template-columns:minmax\(0,1fr\) auto auto\}/.exec(template);
  assert.ok(wrapped, 'the compact toolbar has a row for Stop and a separate row for engine actions');
  const breakpoint = Number(wrapped[1]);
  // Allow up to 32px for “Stop”, 82px for “Add engine” and 84px for “Analyzing…”.
  // The other terms are the stylesheet's icons, progress slot, padding, borders and gaps.
  const stop = 32 + 16 + 60 + 18 + 2 * 5;
  const engines = 16 + 18;
  const add = 82 + 14 + 18 + 5;
  const status = 84 + 8 + 2 * 2 + 5;
  for (let width = 320; width <= 480; width++) {
    const controls = width <= breakpoint ? Math.max(stop, engines + 5 + add) + 2 * 5 : stop + engines + add + 3 * 5;
    assert.ok(width - 12 - controls >= status, `the running label fits at ${width}px`);
  }
});

test('the desktop toolbar wraps based on the untruncated status and action widths', () => {
  assert.match(template, /\.analysis-overview\{display:flex; flex-wrap:wrap;/);
  assert.match(template, /\.position-status\{[^}]*flex:1 0 max-content;[^}]*max-inline-size:100%/);
});
