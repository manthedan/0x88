import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  ENGINE_FAMILY_CATALOG,
  ENGINE_FAMILY_DEFINITIONS,
  ENGINE_FAMILY_PRIORITY,
  V0_ENGINE_FAMILY_PRIORITY,
  canonicalEngineFamily,
  defaultEngineStrength,
  defaultStaticEngineVariant,
  engineFamilyOptions,
  enginePlayLevels,
  enginePlayOptions,
  engineResourceProfile,
  engineStrengthMeta,
  isEngineFamily,
  lc0EngineLabel,
  lc0VariantOptions,
  stockfishEngineLabel,
  stockfishVariantOptions,
  centipawnEngineLabel,
  centipawnVariantOptions,
} from '../src/lc0/engineCatalog.ts';

test('engine family catalog covers the staged selector families in UI order', () => {
  assert.deepEqual(ENGINE_FAMILY_PRIORITY, ['lc0', 'sf', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax', 'centipawn']);
  assert.deepEqual(V0_ENGINE_FAMILY_PRIORITY, ['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess', 'stormphrax', 'centipawn']);
  assert.deepEqual(engineFamilyOptions().map((option) => option.value), ENGINE_FAMILY_PRIORITY);
  assert.equal(ENGINE_FAMILY_PRIORITY.at(-1), 'centipawn');
  assert.equal(V0_ENGINE_FAMILY_PRIORITY.at(-1), 'centipawn');
  assert.equal(ENGINE_FAMILY_CATALOG.centipawn.shortLabel, 'Centi');
  for (const family of ENGINE_FAMILY_PRIORITY) {
    assert.equal(ENGINE_FAMILY_CATALOG[family].id, family);
    assert.ok(ENGINE_FAMILY_CATALOG[family].label.length > 0);
    assert.ok(ENGINE_FAMILY_CATALOG[family].docHref.includes('engine_catalog.md'));
  }
});

test('unified family definitions drive order, resources, strengths, and Play options', () => {
  assert.deepEqual(Object.keys(ENGINE_FAMILY_DEFINITIONS).sort(), [...ENGINE_FAMILY_PRIORITY].sort());
  for (const family of ENGINE_FAMILY_PRIORITY) {
    const definition = ENGINE_FAMILY_DEFINITIONS[family];
    assert.equal(definition.id, family);
    assert.deepEqual(engineResourceProfile(family), definition.resource);
    assert.deepEqual(engineStrengthMeta(family, 'arena'), definition.strength.arena);
    assert.deepEqual(engineStrengthMeta(family, 'analysis'), definition.strength.analysis);
  }
  assert.deepEqual(enginePlayOptions().map((option) => option.id), [
    'leela-queen-odds', 'sf-lite', 'sf-full', 'lc0-small', 'lc0-t3', 'lc0-bt4',
    'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax', 'centipawn',
  ]);
  assert.deepEqual(enginePlayLevels('stormphrax'), [2, 4, 6, 9, 12]);
  assert.throws(() => enginePlayLevels('sf'), /does not use the shared Play strength ladder/);
});

test('engine strength metadata captures arena vs analysis defaults', () => {
  assert.equal(defaultEngineStrength('lc0', 'arena'), 100);
  assert.equal(defaultEngineStrength('lc0', 'analysis'), 400);
  assert.deepEqual(engineStrengthMeta('sf', 'arena'), { unit: 'depth', min: 1, max: 40, def: 8 });
  assert.deepEqual(engineStrengthMeta('sf', 'analysis'), { unit: 'depth', min: 1, max: 30, def: 12 });
  assert.equal(engineStrengthMeta('centipawn', 'arena').def, 100);
  assert.equal(engineStrengthMeta('centipawn', 'analysis').def, 400);
  assert.equal(engineStrengthMeta('viridithas', 'arena').def, 6);
  assert.equal(engineStrengthMeta('viridithas', 'analysis').def, 12);
  assert.equal(engineStrengthMeta('berserk', 'arena').def, 4);
  assert.equal(engineStrengthMeta('berserk', 'analysis').def, 12);
  assert.equal(engineStrengthMeta('plentychess', 'arena').def, 4);
  assert.equal(engineStrengthMeta('plentychess', 'analysis').def, 12);
  assert.equal(engineStrengthMeta('stormphrax', 'arena').def, 4);
  assert.equal(engineStrengthMeta('stormphrax', 'analysis').def, 12);
});

test('static LC0 and Stockfish variants expose labels and gating metadata', () => {
  assert.equal(defaultStaticEngineVariant('lc0'), 'small');
  assert.equal(defaultStaticEngineVariant('centipawn'), 'bt4-ort');
  assert.equal(defaultStaticEngineVariant('sf'), 'lite');
  assert.equal(defaultStaticEngineVariant('berserk'), 'emscripten');
  assert.equal(defaultStaticEngineVariant('plentychess'), 'emscripten');
  assert.equal(defaultStaticEngineVariant('stormphrax'), 'emscripten');
  assert.deepEqual(ENGINE_FAMILY_DEFINITIONS.stormphrax.variants.v0Allowed, ['emscripten', 'emscripten-relaxed']);
  assert.equal(ENGINE_FAMILY_DEFINITIONS.stormphrax.play.options[0]?.variant, 'default');
  assert.equal(lc0EngineLabel('small'), 'Lc0');
  assert.equal(lc0EngineLabel('bt4'), 'Lc0 BT4-it332');
  assert.equal(stockfishEngineLabel('lite', 'arena'), 'Stockfish Lite');
  assert.equal(stockfishEngineLabel('lite', 'analysis'), 'SF Lite');
  assert.equal(centipawnEngineLabel('bt4-ort'), 'Centipawn');
  assert.equal(centipawnEngineLabel('bt4-custom'), 'Centipawn · custom WebGPU');
  assert.deepEqual(centipawnVariantOptions().map((option) => option.value), ['bt4-ort', 'bt4-auto', 'bt4-custom']);
  assert.deepEqual(stockfishVariantOptions().map((option) => option.value), ['lite', 'full']);
  assert.equal(lc0VariantOptions(false).find((option) => option.value === 'bt4')?.disabled, true);
  assert.equal(lc0VariantOptions(true).find((option) => option.value === 'bt4')?.disabled, false);
});

test('engine family guard rejects unknown selector values', () => {
  assert.equal(isEngineFamily('lc0'), true);
  assert.equal(isEngineFamily('centipawn'), true);
  assert.equal(isEngineFamily('berserk'), true);
  assert.equal(isEngineFamily('plentychess'), true);
  assert.equal(isEngineFamily('stormphrax'), true);
  assert.equal(isEngineFamily('tiny'), false);
  assert.equal(isEngineFamily('stockfish'), false);
  assert.equal(isEngineFamily(''), false);
});

test('legacy engine family aliases resolve to canonical families', () => {
  assert.equal(canonicalEngineFamily('tiny'), 'centipawn');
  assert.equal(canonicalEngineFamily('centipawn'), 'centipawn');
  assert.equal(canonicalEngineFamily('lc0'), 'lc0');
  assert.equal(canonicalEngineFamily('stockfish'), null);
});
