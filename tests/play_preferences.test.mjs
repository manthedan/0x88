import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  DEFAULT_PLAY_PREFERENCES,
  PLAY_PREFERENCES_STORAGE_KEY,
  loadPlayPreferences,
  normalizePlayPreferences,
  savePlayPreferences,
} from '../src/lc0/playPreferences.ts';

const engines = new Set(['maia3', 'centipawn', 'stormphrax']);

test('play preferences normalize persisted engine, strength, color, and Maia controls', () => {
  assert.deepEqual(normalizePlayPreferences({
    engineId: 'centipawn', level: 9, color: 'black', maiaElo: 1733,
    maiaStyle: 'argmax', maiaTemperature: 9, maiaTopP: 0,
  }, engines), {
    engineId: 'centipawn', level: 4, color: 'black', maiaElo: 1700,
    maiaStyle: 'argmax', maiaTemperature: 5, maiaTopP: 0.01,
  });
});

test('play preferences reject unknown engines and malformed storage', () => {
  assert.equal(normalizePlayPreferences({ engineId: 'unknown' }, engines).engineId, 'maia3');
  assert.deepEqual(loadPlayPreferences({ getItem: () => '{bad json', setItem() {} }, engines), DEFAULT_PLAY_PREFERENCES);
});

test('play preferences round-trip through storage', () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const preferences = { ...DEFAULT_PLAY_PREFERENCES, engineId: 'stormphrax', level: 1, color: 'random' };
  savePlayPreferences(storage, preferences);
  assert.equal(values.has(PLAY_PREFERENCES_STORAGE_KEY), true);
  assert.deepEqual(loadPlayPreferences(storage, engines), preferences);
});
