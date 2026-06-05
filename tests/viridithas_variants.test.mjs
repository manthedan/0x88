import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const source = readFileSync(new URL('../src/lc0/viridithasVariants.ts', import.meta.url), 'utf8');

test('Viridithas SIMD is the default browser variant', () => {
  assert.match(source, /VIRIDITHAS_VARIANTS: readonly ViridithasVariant\[\] = \[\s*VIRIDITHAS_SIMD_VARIANT,/);
  assert.match(source, /params\.get\('viridithas'\) \?\? params\.get\('viridithasVariant'\) \?\? 'simd'/);
});

test('Viridithas scalar remains explicit compatibility fallback', () => {
  assert.match(source, /value === 'scalar' \|\| value === 'default'/);
  assert.match(source, /VIRIDITHAS_DEFAULT_VARIANT/);
});
