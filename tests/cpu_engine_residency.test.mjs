import assert from 'node:assert/strict';
import test from 'node:test';
import { inactiveCpuEnginesToEvict } from '../src/lc0/cpuEngineResidency.ts';

test('an empty cache and a lone active engine dispose nothing', () => {
  assert.deepEqual(inactiveCpuEnginesToEvict([], 'sf-lite'), []);
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-lite'], 'sf-lite'), []);
});

test('switching between two engines keeps the one just left', () => {
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-lite', 'sf-full'], 'sf-full'), []);
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-full', 'sf-lite'], 'sf-lite'), []);
});

test('older inactive engines past the cap are disposed and the active id is not', () => {
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-lite', 'sf-full', 'berserk'], 'berserk'), ['sf-lite']);
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-lite', 'sf-full', 'viridithas', 'berserk'], 'berserk'), ['sf-lite', 'sf-full']);
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-lite', 'berserk', 'sf-full'], 'berserk'), ['sf-lite']);
  const evicted = inactiveCpuEnginesToEvict(['sf-lite', 'sf-full', 'viridithas', 'berserk'], 'sf-full');
  assert.deepEqual(evicted, ['sf-lite', 'viridithas']);
  assert.equal(evicted.includes('sf-full'), false);
});

test('a cap of zero drops every inactive engine and still spares the active one', () => {
  assert.deepEqual(inactiveCpuEnginesToEvict(['sf-lite', 'sf-full'], 'sf-full', 0), ['sf-lite']);
});
