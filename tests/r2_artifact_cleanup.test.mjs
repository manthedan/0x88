import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { artifactKeyFromUrl, buildCleanupPlan, parseArgs } from '../scripts/plan_r2_artifact_cleanup.mjs';

const now = new Date('2026-06-26T00:00:00.000Z');

function object(key, size = 1, last_modified = '2026-03-01T00:00:00.000Z') {
  return { key, size, last_modified, storage_class: 'Standard' };
}

test('artifactKeyFromUrl extracts only content-addressed artifact keys', () => {
  const key = `artifacts/sha256/${'a'.repeat(64)}/model.onnx`;
  assert.equal(artifactKeyFromUrl(`/${key}`), key);
  assert.equal(artifactKeyFromUrl('/models/lc0/model.onnx'), undefined);
});

test('R2 cleanup plan protects retained release artifacts and flags safe legacy duplicates', () => {
  const retainedKey = `artifacts/sha256/${'1'.repeat(64)}/model.onnx`;
  const oldKey = `artifacts/sha256/${'2'.repeat(64)}/old-model.onnx`;
  const freshOrphan = `artifacts/sha256/${'3'.repeat(64)}/fresh.onnx`;
  const sourceOrphan = `artifacts/sha256/${'4'.repeat(64)}/engine-corresponding-source.tar.gz`;
  const releases = [{
    releaseId: 'stable-release',
    artifacts: [{ logicalUrl: '/models/lc0/model.onnx', artifactUrl: `/${retainedKey}` }],
  }];
  const plan = buildCleanupPlan({
    now,
    retentionDays: 30,
    channel: { releaseId: 'stable-release' },
    releases,
    objects: [
      object('channels/stable.json'),
      object('releases/stable-release.json'),
      object(retainedKey),
      object('models/lc0/model.onnx', 123),
      object('models/lc0/manifest.json', 5),
      object(oldKey, 50, '2026-01-01T00:00:00.000Z'),
      object(freshOrphan, 50, '2026-06-20T00:00:00.000Z'),
      object(sourceOrphan, 50, '2026-01-01T00:00:00.000Z'),
    ],
  });

  assert.deepEqual(plan.missingReferencedArtifacts, []);
  assert.equal(plan.summaryByCategory['legacy-logical-duplicate'].count, 1);
  assert.equal(plan.summaryByCategory['legacy-unreferenced-metadata'].count, 1);
  assert.equal(plan.summaryByCategory['hashed-orphan'].count, 1);
  assert.ok(plan.candidates.some((candidate) => candidate.key === oldKey && candidate.category === 'hashed-orphan'));
  assert.ok(plan.protected.some((entry) => entry.key === retainedKey && entry.reason === 'referenced by stable release'));
  assert.ok(plan.protected.some((entry) => entry.key === freshOrphan && /retention window/.test(entry.reason)));
  assert.ok(plan.protected.some((entry) => entry.key === sourceOrphan && /source archive/.test(entry.reason)));
});

test('R2 cleanup plan reports missing retained artifacts', () => {
  const missingKey = `artifacts/sha256/${'5'.repeat(64)}/missing.wasm`;
  const plan = buildCleanupPlan({
    now,
    channel: { releaseId: 'stable-release' },
    releases: [{ releaseId: 'stable-release', artifacts: [{ logicalUrl: '/missing.wasm', artifactUrl: `/${missingKey}` }] }],
    objects: [object('channels/stable.json'), object('releases/stable-release.json')],
  });

  assert.deepEqual(plan.missingReferencedArtifacts, [{ key: missingKey, releases: ['stable-release'] }]);
});

test('parseArgs requires hashed opt-in separately from execute', () => {
  const args = parseArgs(['node', 'script', '--execute', '--delete-category', 'legacy-logical-duplicate,legacy-unreferenced-metadata', '--retention-days', '7']);
  assert.equal(args.execute, true);
  assert.equal(args.allowDeleteHashed, false);
  assert.deepEqual([...args.deleteCategories].sort(), ['legacy-logical-duplicate', 'legacy-unreferenced-metadata']);
  assert.equal(args.retentionDays, 7);
});

test('public docs do not link directly to the legacy R2 dev bucket', async () => {
  const docsPage = await readFile('src/routes/docs/+page.svelte', 'utf8');
  assert.doesNotMatch(docsPage, /r2\.dev|pub-c3fb64db6e434c738bc86cb1a56d6384/);
});
