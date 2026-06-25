import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  ENGINE_FAMILY_CATALOG,
  ENGINE_FAMILY_PRIORITY,
  defaultEngineStrength,
  defaultStaticEngineVariant,
  engineFamilyOptions,
  engineStrengthMeta,
  isEngineFamily,
  lc0EngineLabel,
  lc0VariantOptions,
  stockfishEngineLabel,
  stockfishVariantOptions,
  tinyEngineLabel,
  tinyVariantOptions,
} from '../src/lc0/engineCatalog.ts';

test('engine family catalog covers the staged selector families in UI order', () => {
  assert.deepEqual(ENGINE_FAMILY_PRIORITY, ['lc0', 'sf', 'reckless', 'viridithas', 'berserk', 'plentychess', 'tiny']);
  assert.deepEqual(engineFamilyOptions().map((option) => option.value), ENGINE_FAMILY_PRIORITY);
  for (const family of ENGINE_FAMILY_PRIORITY) {
    assert.equal(ENGINE_FAMILY_CATALOG[family].id, family);
    assert.ok(ENGINE_FAMILY_CATALOG[family].label.length > 0);
    assert.ok(ENGINE_FAMILY_CATALOG[family].docHref.includes('engine_catalog.md'));
  }
});

test('engine strength metadata captures arena vs analysis defaults', () => {
  assert.equal(defaultEngineStrength('lc0', 'arena'), 100);
  assert.equal(defaultEngineStrength('lc0', 'analysis'), 400);
  assert.deepEqual(engineStrengthMeta('sf', 'arena'), { unit: 'depth', min: 1, max: 40, def: 8 });
  assert.deepEqual(engineStrengthMeta('sf', 'analysis'), { unit: 'depth', min: 1, max: 30, def: 14 });
  assert.equal(engineStrengthMeta('tiny', 'arena').def, 100);
  assert.equal(engineStrengthMeta('tiny', 'analysis').def, 400);
  assert.equal(engineStrengthMeta('viridithas', 'arena').def, 6);
  assert.equal(engineStrengthMeta('viridithas', 'analysis').def, 14);
  assert.equal(engineStrengthMeta('berserk', 'arena').def, 4);
  assert.equal(engineStrengthMeta('berserk', 'analysis').def, 14);
  assert.equal(engineStrengthMeta('plentychess', 'arena').def, 4);
  assert.equal(engineStrengthMeta('plentychess', 'analysis').def, 14);
});

test('static LC0 and Stockfish variants expose labels and gating metadata', () => {
  assert.equal(defaultStaticEngineVariant('lc0'), 'small');
  assert.equal(defaultStaticEngineVariant('tiny'), 'bt4-auto');
  assert.equal(defaultStaticEngineVariant('sf'), 'lite');
  assert.equal(defaultStaticEngineVariant('berserk'), 'emscripten');
  assert.equal(defaultStaticEngineVariant('plentychess'), 'emscripten');
  assert.equal(lc0EngineLabel('small'), 'Lc0');
  assert.equal(lc0EngineLabel('bt4'), 'Lc0 BT4-it332');
  assert.equal(stockfishEngineLabel('lite', 'arena'), 'Stockfish Lite');
  assert.equal(stockfishEngineLabel('lite', 'analysis'), 'SF Lite');
  assert.equal(tinyEngineLabel('bt4-custom'), 'Tiny Leela · custom WebGPU');
  assert.deepEqual(tinyVariantOptions().map((option) => option.value), ['bt4-auto', 'bt4-ort', 'bt4-custom']);
  assert.deepEqual(stockfishVariantOptions().map((option) => option.value), ['lite', 'full']);
  assert.equal(lc0VariantOptions(false).find((option) => option.value === 'bt4')?.disabled, true);
  assert.equal(lc0VariantOptions(true).find((option) => option.value === 'bt4')?.disabled, false);
});

test('engine family guard rejects unknown selector values', () => {
  assert.equal(isEngineFamily('lc0'), true);
  assert.equal(isEngineFamily('tiny'), true);
  assert.equal(isEngineFamily('berserk'), true);
  assert.equal(isEngineFamily('plentychess'), true);
  assert.equal(isEngineFamily('stockfish'), false);
  assert.equal(isEngineFamily(''), false);
});
