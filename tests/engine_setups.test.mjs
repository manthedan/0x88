import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { ENGINE_PROFILE_BACKUP_KIND, engineSetupRuntime, normalizeEngineProfiles } from '../src/lc0/engineSetups.ts';

const row = { family: 'sf', variant: 'lite', strength: 8 };
const sanitize = (value) => value?.family === 'sf' && value.variant !== 'custom' ? { ...value } : null;
const profile = (name, extra = {}) => ({ name, rows: [row], multiPv: 3, lc0Runtime: 'onnx', ...extra });

test('saved setups and version-one backups use the same normalization', () => {
  const profiles = [profile(' Work '), profile('Fast', { multiPv: 25 })];
  const expected = [profile('Fast', { multiPv: 10 }), profile('Work')];
  assert.deepEqual(normalizeEngineProfiles(profiles, sanitize), expected);
  assert.deepEqual(normalizeEngineProfiles({ kind: ENGINE_PROFILE_BACKUP_KIND, version: 1, profiles }, sanitize), expected);
  assert.equal(profiles[0].name, ' Work ');
});

test('setup imports reject invalid envelopes, rows and reserved names', () => {
  for (const input of [null, 3, {}, { kind: 'other', version: 1, profiles: [profile('A')] }, { kind: ENGINE_PROFILE_BACKUP_KIND, version: 2, profiles: [profile('A')] }]) {
    assert.deepEqual(normalizeEngineProfiles(input, sanitize), []);
  }
  assert.deepEqual(normalizeEngineProfiles([
    null, {}, profile(''), profile('builtin:probe'), profile('Custom', { rows: [{ ...row, variant: 'custom' }] }),
    profile('Invalid', { rows: [{}] }), profile('Good', { lc0Runtime: {}, multiPv: -3 }),
  ], sanitize), [profile('Good', { multiPv: 1 })]);
});

test('duplicate names are unambiguous and valid row policy is delegated', () => {
  const seen = [];
  const results = normalizeEngineProfiles([profile('Same'), profile('Same', { multiPv: 1, rows: [row, row] })], (value, index) => {
    seen.push(index);
    return sanitize(value);
  });
  assert.deepEqual(seen, [0, 0, 1]);
  assert.equal(results.length, 1);
  assert.equal(results[0].multiPv, 1);
  assert.equal(results[0].rows.length, 2);
});

test('large-net setup confirmation reads metadata without allocating an engine session', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const guard = source.slice(source.indexOf('function profileRowsForUse('), source.indexOf('function loadEngineProfiles('));
  assert.match(guard, /const config = BIG_NETS\[/);
  assert.match(guard, /bigNetSelectableSync\(config\) && allowBt4Prompt && window.confirm/);
  assert.match(guard, /if \(!allowed\) next.variant = 'small'/);
  assert.doesNotMatch(guard, /bigNetFor\(|acquireBigNetSearcher\(/);
});

test('normal saved setups cannot activate developer-only runtimes, including on restore/import', () => {
  for (const runtime of ['onnx', 'hybrid-wgsl-heads', 'hybrid-ort-heads', 'whole-onnx-webgpu']) {
    assert.equal(engineSetupRuntime(runtime, false), 'onnx');
  }
  assert.equal(engineSetupRuntime('hybrid-wgsl-heads', true), 'hybrid-wgsl-heads');
  const [research] = normalizeEngineProfiles([profile('Research', { lc0Runtime: 'hybrid-wgsl-heads' })], sanitize);
  assert.equal(engineSetupRuntime(research.lc0Runtime, false), 'onnx');
});
