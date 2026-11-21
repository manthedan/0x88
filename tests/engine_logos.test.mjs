import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { engineLogoFamilyForEngineFamily, engineLogoFamilyForName, engineLogoHtml, engineLogoUrl } from '../src/lc0/engineLogos.ts';

test('engine logo helper maps displayed engine names to bundled logo families', () => {
  assert.equal(engineLogoFamilyForName('Lc0 BT4'), 'lc0');
  assert.equal(engineLogoFamilyForName('Leela Chess Zero'), 'lc0');
  assert.equal(engineLogoFamilyForName('Stockfish Lite'), 'sf');
  assert.equal(engineLogoFamilyForName('SF 18'), 'sf');
  assert.equal(engineLogoFamilyForName('Reckless NNUE'), 'reckless');
  assert.equal(engineLogoFamilyForName('Viridithas WASI'), 'viridithas');
  assert.equal(engineLogoFamilyForName('Berserk Emscripten'), 'berserk');
  assert.equal(engineLogoFamilyForName('Centipawn'), 'centipawn');
  assert.equal(engineLogoFamilyForName('PlentyChess'), 'generic');
});

test('engine logo helper maps selector families to bundled assets', () => {
  assert.equal(engineLogoFamilyForEngineFamily('lc0'), 'lc0');
  assert.equal(engineLogoFamilyForEngineFamily('sf'), 'sf');
  assert.equal(engineLogoFamilyForEngineFamily('reckless'), 'reckless');
  assert.equal(engineLogoFamilyForEngineFamily('viridithas'), 'viridithas');
  assert.equal(engineLogoFamilyForEngineFamily('berserk'), 'berserk');
  assert.equal(engineLogoFamilyForEngineFamily('centipawn'), 'centipawn');
  assert.equal(engineLogoFamilyForEngineFamily('plentychess'), 'generic');
});

test('engine logo URLs match the committed public asset names and extensions', () => {
  assert.equal(engineLogoUrl('lc0'), '/engine-logos/lc0.svg');
  assert.equal(engineLogoUrl('centipawn'), '/engine-logos/centipawn.svg');
  assert.equal(engineLogoUrl('sf'), '/engine-logos/stockfish.png');
  assert.equal(engineLogoUrl('reckless'), '/engine-logos/reckless.png');
  assert.equal(engineLogoUrl('viridithas'), '/engine-logos/viridithas.png');
  assert.equal(engineLogoUrl('berserk'), '/engine-logos/berserk.jpg');
  assert.equal(engineLogoUrl('generic'), '/engine-logos/generic.svg');
});

test('bundled logos render as lazy images without an availability probe', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', () => assert.fail('bundled logo must not require fetch()'));
  for (const family of ['lc0', 'centipawn', 'sf', 'reckless', 'viridithas', 'berserk', 'generic']) {
    const url = engineLogoUrl(family);
    assert.ok((await readFile(new URL(`../public${url}`, import.meta.url))).byteLength > 0);
    assert.ok(engineLogoHtml(family).includes(`src="${url}" alt="" loading="lazy"`));
  }
  assert.equal(fetch.mock.callCount(), 0);
});

test('engine comparison retains readable engine-and-variant display names', async () => {
  const { analysisEngineName } = await import('../src/lc0/analysisPresentation.ts');
  assert.equal(analysisEngineName('Lc0'), 'Lc0 Small');
  assert.equal(analysisEngineName('Lc0 BT4'), 'Lc0 BT4');
  assert.equal(analysisEngineName('SF Lite'), 'Stockfish Lite');
  assert.equal(analysisEngineName('SF 18'), 'Stockfish 18');
  const source = await readFile(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const comparisonSource = source.slice(source.indexOf('function renderEngineComparison'), source.indexOf('function renderLines'));
  assert.match(comparisonSource, /htmlEscape\(analysisEngineName\(line\.engine\)\)/);
});
