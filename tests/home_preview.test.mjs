import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';

// Exercise the real component's SSR, not a second copy of its example data.
const source = readFileSync(new URL('../src/lib/components/AnalysisPreview.svelte', import.meta.url), 'utf8');
const { js } = compile(source, { generate: 'server' });
const code = js.code
  .replace(/(['"])svelte\/internal\/server\1/g, JSON.stringify(import.meta.resolve('svelte/internal/server')))
  .replace(/(['"])(\.\.\/\.\.\/lc0\/[^'"]+)\1/g, (_, _q, path) => JSON.stringify(new URL(path, new URL('../src/lib/components/', import.meta.url)).href));
const { default: Preview } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const { body: html } = render(Preview);

test('home comparison is explicitly an example with unmodified measured results', () => {
  assert.match(html, /Example engine comparison/);
  assert.match(html, /White to move · evaluations from White’s perspective/);
  assert.match(html, /0\.30 pawns/);
  assert.match(html, /Lc0 Small/);
  assert.match(html, /365 visits/);
  assert.match(html, />\+0\.25</);
  assert.match(html, /Stockfish Lite/);
  assert.match(html, /depth 12/);
  assert.match(html, />\+0\.55</);
  assert.doesNotMatch(html, /class="evalbar/, 'do not imply that one engine’s score represents both engines');
});

test('home position retains its chessground board, coordinates and FEN deep link', () => {
  assert.equal((html.match(/<cg-board\b/g) ?? []).length, 1);
  assert.equal((html.match(/<piece /g) ?? []).length, 30);
  assert.match(html, /class="white knight[^\"]*" style="transform:translate\(200%,500%\)"/);
  assert.match(html, /class="black knight[^\"]*" style="transform:translate\(500%,200%\)"/);
  const ranks = html.match(/class="ranks[^\"]*"[^>]*>(.*?)<\/div>/s)[1];
  const files = html.match(/class="files[^\"]*"[^>]*>(.*?)<\/div>/s)[1];
  assert.deepEqual(
    [...ranks.matchAll(/<span[^>]*>(\d)<\/span>/g)].map((m) => m[1]),
    [...'87654321'],
  );
  assert.deepEqual(
    [...files.matchAll(/<span[^>]*>([a-h])<\/span>/g)].map((m) => m[1]),
    [...'abcdefgh'],
  );
  const link = html.match(/href="(\/app\/analysis\/\?fen=[^\"]+)"/)[1];
  const fen = new URL(link, 'https://0x88.app').searchParams.get('fen');
  assert.equal(fen, 'r2q1rk1/3nbppp/p2pbn2/4p1P1/1p2P3/1NN1BP2/PPPQ3P/2KR1B1R w - - 0 13');
});

test('home arrows keep one best move per engine at the owner’s uniform width', () => {
  const arrows = [...html.matchAll(/<line\b[^>]+>/g)].map((m) => m[0]);
  assert.equal(arrows.length, 2);
  for (const arrow of arrows) {
    assert.match(arrow, /stroke-width="0\.171875"/, '11 / 64 on one-unit squares, as chessground draws them');
    assert.match(arrow, /stroke-linecap="butt"/);
  }
  // The shaft starts near the origin square's edge (c3 → e2): 0.42 squares along the arrow.
  const [x1, y1] = ['x1', 'y1'].map((name) => Number(arrows[0].match(new RegExp(`${name}="([^"]+)"`))[1]));
  assert.ok(Math.abs(Math.hypot(x1 - 2.5, y1 - 5.5) - 0.42) < 1e-9);
  assert.match(html, /markerUnits="userSpaceOnUse"/);
  assert.match(html, /<svg class="arrows[^"]*"[^>]*opacity="0\.45"/, 'brush 75% × chessground shapes layer 60%');
  assert.match(html, /d="M0,-0\.275 L0\.4,0 L0,0\.275 Z"/);
  assert.match(arrows[0], /stroke="#7a4fd6"/);
  assert.match(arrows[1], /stroke="#1f5fcf"/);
  assert.match(html, /Best-move arrows/);
  assert.ok(html.indexOf('class="pv-board') < html.indexOf('class="pv-results'), 'board precedes detailed results in reading order');
});

test('home PVs group complete move pairs and retain native mobile disclosures', () => {
  assert.equal((html.match(/<details /g) ?? []).length, 2);
  assert.doesNotMatch(html, /<details[^>]*\bopen\b/);
  assert.match(html, /<summary[^>]*>365 visits/);
  assert.match(html, /<summary[^>]*>depth 12/);
  assert.match(html, />13\. Ne2 Nh5<\/span> /);
  assert.match(html, />14\. Ng3 Nf4<\/span> /);
  assert.match(html, />15\. h4 a5<\/span> /);
  assert.match(html, />13\. gxf6 bxc3<\/span> /);
  assert.match(html, />14\. Qxc3 Nxf6<\/span> /);
  assert.match(html, />15\. Na5 Rc8<\/span> /);
});
