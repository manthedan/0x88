import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { ARROW_GEOMETRY, arrowLayout } from '../src/lc0/arrowGeometry.ts';
import { CHESSGROUND_SVG_PATCH_MARKER, patchChessgroundSvg } from '../src/lc0/chessgroundArrowPatch.ts';

const close = (actual, expected, message) => assert.ok(Math.abs(actual - expected) < 1e-9, `${message}: ${actual} != ${expected}`);
const along = (_layout, from, x, y) => Math.hypot(x - from[0], y - from[1]);

test('arrow tip sits on the target centre with the head base 0.10 inside the target square', () => {
  const from = [4.5, 6.5], to = [4.5, 4.5]; // e2 → e4
  const a = arrowLayout(from, to, false, ARROW_GEOMETRY);
  assert.deepEqual(a.tip, to);
  close(along(a, from, a.x1, a.y1), 0.42, 'tail starts near the origin square edge');
  // The marker puts its base `overlap` behind the shaft end.
  const baseFromTarget = Math.hypot(to[0] - a.x2, to[1] - a.y2) + ARROW_GEOMETRY.overlap;
  close(baseFromTarget, 0.4, 'head base distance from the target centre');
  assert.ok(0.5 - baseFromTarget >= 0.1 - 1e-9, 'base is clearly inside the target square, not on the boundary');
});

test('arrow geometry does not depend on line width; brushes only change the shaft', () => {
  assert.equal(Object.keys(ARROW_GEOMETRY).some((key) => /width/i.test(key) && key !== 'headWidth'), false);
  const patched = patchChessgroundSvg(readFileSync(chessgroundSvg(), 'utf8'));
  const marker = patched.slice(patched.indexOf('function renderMarker'), patched.indexOf('function renderLabel'));
  assert.doesNotMatch(marker, /lineWidth/);
  assert.match(marker, /markerUnits: 'userSpaceOnUse'/);
});

test('one-square and diagonal moves keep a visible shaft', () => {
  const one = arrowLayout([3.5, 6.5], [3.5, 5.5], false, ARROW_GEOMETRY); // d2 → d3
  close(one.y1, 6.5 - 0.42, 'tail');
  close(one.y2, 6.5 - 0.64, 'shaft end');
  const shared = arrowLayout([3.5, 6.5], [3.5, 5.5], true, ARROW_GEOMETRY);
  close(shared.tip[1], 5.5 + 10 / 64, 'shared targets stop short of the centre');
  assert.ok(shared.y1 - shared.y2 >= 0.06 + ARROW_GEOMETRY.overlap - 1e-9, 'shared one-square arrows still show a shaft');
  const knight = arrowLayout([6.5, 7.5], [5.5, 5.5], false, ARROW_GEOMETRY); // g1 → f3
  close(Math.hypot(knight.ux, knight.uy), 1, 'unit direction');
});

function chessgroundSvg() {
  return fileURLToPath(import.meta.resolve('chessground/svg'));
}

test('the chessground patch applies to the installed version and injects the shared layout', () => {
  const source = readFileSync(chessgroundSvg(), 'utf8');
  const patched = patchChessgroundSvg(source);
  assert.ok(patched.startsWith(CHESSGROUND_SVG_PATCH_MARKER));
  assert.doesNotMatch(patched, /xo = Math\.cos\(angle\) \* m/);
  assert.match(patched, /'stroke-linecap': 'butt'/);
  const header = patched.slice(0, patched.indexOf('\nimport'));
  const injected = new Function(`${header}\nreturn { layout: __cgArrowLayout, geometry: __CG_ARROW_GEOMETRY, head: __CG_ARROW_HEAD };`)();
  assert.deepEqual(injected.geometry, ARROW_GEOMETRY);
  assert.equal(injected.head, 'M0,-0.275 L0.4,0 L0,0.275 Z');
  assert.deepEqual(injected.layout([6.5, 7.5], [5.5, 5.5], false, injected.geometry), arrowLayout([6.5, 7.5], [5.5, 5.5], false, ARROW_GEOMETRY));
});

test('the chessground patch fails loudly when chessground changes the patched code', () => {
  const source = readFileSync(chessgroundSvg(), 'utf8').replace("'stroke-linecap': 'round',", "'stroke-linecap': 'square',");
  assert.throws(() => patchChessgroundSvg(source), /expected 1 match, found 0/);
});

test('the shapes-layer opacity constant matches chessground.base.css', async () => {
  const { CHESSGROUND_SHAPES_LAYER_OPACITY } = await import('../src/lc0/analysisFormat.ts');
  const css = readFileSync(new URL('../public/chessground.base.css', import.meta.url), 'utf8');
  const rule = css.match(/\.cg-wrap \.cg-shapes \{([^}]*)\}/)[1];
  assert.equal(Number(rule.match(/opacity:\s*([\d.]+)/)[1]), CHESSGROUND_SHAPES_LAYER_OPACITY);
});
