import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { engineVariantMenuLabel } from '../src/lc0/engineVariantLabels.ts';
import { ENGINE_FAMILY_DEFINITIONS } from '../src/lc0/engineCatalog.ts';
import {
  RECKLESS_FULL_VARIANT,
  RECKLESS_RELAXED_SIMD_VARIANT,
  RECKLESS_SIMD_VARIANT,
  RECKLESS_VARIANTS,
} from '../src/lc0/recklessVariants.ts';
import {
  BERSERK_EMSCRIPTEN_RELAXED_VARIANT,
  BERSERK_EMSCRIPTEN_SIMD_VARIANT,
  BERSERK_EMSCRIPTEN_VARIANT,
} from '../src/lc0/berserkVariants.ts';
import { VIRIDITHAS_DEFAULT_VARIANT } from '../src/lc0/viridithasVariants.ts';
import { PLENTYCHESS_EMSCRIPTEN_VARIANT } from '../src/lc0/plentychessVariants.ts';

test('variant menus omit the family and implementation boilerplate', () => {
  assert.equal(engineVariantMenuLabel('reckless', 'relaxed-simd', 'unused'), 'Relaxed SIMD');
  assert.equal(engineVariantMenuLabel('viridithas', 'default', 'unused'), 'Scalar');
  assert.equal(engineVariantMenuLabel('berserk', 'emscripten', 'unused'), 'Scalar');
  assert.equal(engineVariantMenuLabel('berserk', 'emscripten-simd', 'unused'), 'SIMD');
  assert.equal(engineVariantMenuLabel('plentychess', 'emscripten-relaxed', 'unused'), 'Relaxed SIMD');
  assert.equal(engineVariantMenuLabel('stormphrax', 'emscripten-relaxed', 'unused'), '8 · Relaxed SIMD');
  assert.equal(engineVariantMenuLabel('lc0', 'small', 'Small'), 'Small');
});

test('engine result labels stay concise and omit implementation status words', () => {
  assert.deepEqual([
    RECKLESS_FULL_VARIANT.label,
    RECKLESS_SIMD_VARIANT.label,
    RECKLESS_RELAXED_SIMD_VARIANT.label,
  ], ['Reckless scalar fallback', 'Reckless SIMD', 'Reckless Relaxed SIMD']);
  assert.deepEqual([
    BERSERK_EMSCRIPTEN_VARIANT.label,
    BERSERK_EMSCRIPTEN_SIMD_VARIANT.label,
    BERSERK_EMSCRIPTEN_RELAXED_VARIANT.label,
  ], ['Berserk', 'Berserk SIMD', 'Berserk Relaxed SIMD']);
  assert.equal(VIRIDITHAS_DEFAULT_VARIANT.label, 'Viridithas');
  assert.equal(PLENTYCHESS_EMSCRIPTEN_VARIANT.label, 'PlentyChess');
  for (const variant of RECKLESS_VARIANTS) assert.doesNotMatch(variant.label, /\bFull\b|experimental/i);
});

test('production catalog exposes hosted variants and prefers relaxed Viridithas', () => {
  assert.deepEqual(ENGINE_FAMILY_DEFINITIONS.lc0.variants.v0Allowed, ['small', 't3', 'bt4']);
  assert.deepEqual(ENGINE_FAMILY_DEFINITIONS.sf.variants.v0Allowed, ['lite', 'full']);
  assert.equal(ENGINE_FAMILY_DEFINITIONS.viridithas.variants.default, 'relaxed-simd');
});

test('production pages probe big-net capabilities and do not force Viridithas back to fixed SIMD', async () => {
  const [analysis, arena] = await Promise.all([
    readFile(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8'),
    readFile(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8'),
  ]);
  assert.doesNotMatch(analysis, /if \(isV0DeployProfile\(\)\) return;\s*await Promise\.all\(\[probeBt4Support/);
  assert.doesNotMatch(arena, /if \(isV0DeployProfile\(\)\) return;\s*await Promise\.all\(\[probeBt4Support/);
  assert.match(arena, /const pendingBigNet = !bt4AvailabilityResolved && family === 'lc0'/);
  assert.match(arena, /const pendingBigNet = !bt4AvailabilityResolved && row\.family === 'lc0'/);
  assert.doesNotMatch(analysis, /REQUESTED_VIRIDITHAS_VARIANT\.key === 'relaxed-simd'/);
});
