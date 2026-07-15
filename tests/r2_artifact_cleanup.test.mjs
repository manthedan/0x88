import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { artifactKeyFromUrl, buildCleanupPlan, listR2Objects, parseArgs } from '../scripts/plan_r2_artifact_cleanup.mjs';

const now = new Date('2026-06-26T00:00:00.000Z');

function object(key, size = 1, last_modified = '2026-03-01T00:00:00.000Z') {
  return { key, size, last_modified, storage_class: 'Standard' };
}

test('artifactKeyFromUrl extracts only content-addressed artifact keys', () => {
  const key = `artifacts/sha256/${'a'.repeat(64)}/model.onnx`;
  assert.equal(artifactKeyFromUrl(`/${key}`), key);
  assert.equal(artifactKeyFromUrl('/models/lc0/model.onnx'), undefined);
});

test('R2 cleanup plan protects retained release artifacts and migration-compatible legacy logical objects', () => {
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
  assert.equal(plan.summaryByCategory['legacy-logical-duplicate'], undefined);
  assert.equal(plan.summaryByCategory['legacy-unreferenced-metadata'].count, 1);
  assert.equal(plan.summaryByCategory['hashed-orphan'].count, 1);
  assert.ok(plan.candidates.some((candidate) => candidate.key === oldKey && candidate.category === 'hashed-orphan'));
  assert.ok(plan.protected.some((entry) => entry.key === retainedKey && entry.reason === 'referenced by stable release'));
  assert.ok(plan.protected.some((entry) => entry.key === 'models/lc0/model.onnx' && /active stable release/.test(entry.reason)));
  assert.ok(plan.protected.some((entry) => entry.key === freshOrphan && /retention window/.test(entry.reason)));
  assert.ok(plan.protected.some((entry) => entry.key === sourceOrphan && /source archive/.test(entry.reason)));
});

test('R2 cleanup plan preserves v1 logical objects referenced by retained non-stable releases', () => {
  const plan = buildCleanupPlan({
    now,
    retentionDays: 30,
    channel: { releaseId: 'stable-release' },
    releases: [
      { releaseId: 'stable-release', artifacts: [] },
      {
        releaseId: 'rollback-release',
        artifacts: [{
          logicalUrl: '/legacy/engine.wasm',
          artifactUrl: `/artifacts/sha256/${'8'.repeat(64)}/engine.wasm`,
        }],
      },
    ],
    objects: [object('legacy/engine.wasm', 10, '2025-01-01T00:00:00.000Z')],
  });

  assert.equal(plan.candidates.some((entry) => entry.key === 'legacy/engine.wasm'), false);
  assert.ok(plan.protected.some((entry) => entry.key === 'legacy/engine.wasm'
    && /rollback and migration client compatibility/.test(entry.reason)));
});

test('R2 cleanup plan requires manual review for unreferenced SHA-only v2 objects', () => {
  const rawSha = '9'.repeat(64);
  const brSha = 'a'.repeat(64);
  const identityKey = `artifacts/${'sha256'}/${rawSha}/identity`;
  const brKey = `artifacts/${'sha256'}/${rawSha}/br/${brSha}`;
  const plan = buildCleanupPlan({
    now,
    retentionDays: 30,
    channel: undefined,
    releases: [],
    objects: [
      object(identityKey, 10, '2025-01-01T00:00:00.000Z'),
      object(brKey, 8, '2025-01-01T00:00:00.000Z'),
    ],
  });

  assert.equal(plan.candidateCount, 0);
  assert.ok(plan.protected.every((entry) => /logical filename and artifact kind are unavailable/.test(entry.reason)));
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

test('R2 cleanup plan protects every shared v2 representation and reports catalog gaps once', () => {
  const rawSha = '6'.repeat(64);
  const brSha = '7'.repeat(64);
  const identityKey = `artifacts/${'sha256'}/${rawSha}/identity`;
  const brKey = `artifacts/sha256/${rawSha}/br/${brSha}`;
  const migratedV1Key = identityKey.replace(/\/identity$/, '/model-a.onnx');
  const representationMap = [
    { encoding: 'identity', url: `/${identityKey}`, sha256: rawSha, bytes: 3 },
    { encoding: 'br', url: `/${brKey}`, sha256: brSha, bytes: 2 },
  ];
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'stable-v2',
    artifacts: [
      {
        logicalUrl: '/models/a.onnx',
        carriedForwardFrom: 'legacy-v1',
        migrationSource: {
          schema: 'lc0_browser.artifact_migration_source.v1',
          releaseId: 'legacy-v1',
          key: migratedV1Key,
          url: `/${migratedV1Key}`,
        },
        raw: { sha256: rawSha, bytes: 3 },
        representations: representationMap,
      },
      { logicalUrl: '/models/b.onnx', raw: { sha256: rawSha, bytes: 3 }, representations: representationMap },
    ],
  };
  const plan = buildCleanupPlan({
    now,
    channel: { releaseId: 'stable-v2' },
    releases: [release],
    objects: [object('channels/stable.json'), object('releases/stable-v2.json'), object(identityKey), object(migratedV1Key)],
  });

  assert.equal(plan.catalogObjectCount, 3);
  assert.deepEqual(plan.missingReferencedArtifacts, [{ key: brKey, releases: ['stable-v2'] }]);
  assert.ok(plan.protected.some((entry) => entry.key === identityKey && entry.reason === 'referenced by stable release'));
  assert.ok(plan.protected.some((entry) => entry.key === migratedV1Key && entry.reason === 'referenced by stable release'));
  assert.equal(plan.candidates.some((entry) => entry.key === identityKey), false);
});

test('R2 cleanup plan releases legacy migration bodies only after their v2 release is no longer retained', () => {
  const rawSha = 'd'.repeat(64);
  const migrationKey = `artifacts/sha256/${rawSha}/legacy-model.onnx`;
  const retainedRelease = {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'rollback-v2',
    artifacts: [{
      logicalUrl: '/models/legacy-model.onnx',
      carriedForwardFrom: 'legacy-v1',
      migrationSource: {
        schema: 'lc0_browser.artifact_migration_source.v1',
        releaseId: 'legacy-v1',
        key: migrationKey,
        url: `/${migrationKey}`,
      },
      raw: { sha256: rawSha, bytes: 3 },
      representations: [{
        encoding: 'identity',
        url: `/artifacts/sha256/${rawSha}/identity`,
        sha256: rawSha,
        bytes: 3,
      }],
    }],
  };
  const objects = [object(migrationKey, 3, '2025-01-01T00:00:00.000Z')];

  const retained = buildCleanupPlan({ now, retentionDays: 30, releases: [retainedRelease], objects });
  assert.equal(retained.candidates.some((entry) => entry.key === migrationKey), false);
  assert.ok(retained.protected.some((entry) => entry.key === migrationKey
    && entry.reason === 'referenced by retained release'));

  const unretained = buildCleanupPlan({ now, retentionDays: 30, releases: [], objects });
  assert.ok(unretained.candidates.some((entry) => entry.key === migrationKey
    && entry.category === 'hashed-orphan'));
});

test('R2 cleanup plan fails closed on malformed v2 migration source metadata', () => {
  const rawSha = 'e'.repeat(64);
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'malformed-migration',
    artifacts: [{
      logicalUrl: '/models/model.onnx',
      carriedForwardFrom: 'legacy-v1',
      migrationSource: {
        schema: 'lc0_browser.artifact_migration_source.v1',
        releaseId: 'legacy-v1',
        key: `artifacts/sha256/${rawSha}/br/not-a-v1-body`,
        url: `/artifacts/sha256/${rawSha}/br/not-a-v1-body`,
      },
      raw: { sha256: rawSha, bytes: 3 },
      representations: [{
        encoding: 'identity',
        url: `/artifacts/sha256/${rawSha}/identity`,
        sha256: rawSha,
        bytes: 3,
      }],
    }],
  };

  assert.throws(
    () => buildCleanupPlan({ now, releases: [release], objects: [] }),
    /Invalid v1 migration source metadata.*manual review/,
  );
});

test('R2 cleanup plan carries source kind and logical metadata into protected v2 catalog objects', () => {
  const rawSha = 'b'.repeat(64);
  const identityKey = `artifacts/${'sha256'}/${rawSha}/identity`;
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'source-release',
    artifacts: [{
      logicalUrl: '/stockfish/stockfish-corresponding-source.tar.gz',
      kind: 'source',
      raw: { sha256: rawSha, bytes: 3 },
      representations: [{
        encoding: 'identity',
        url: `/${identityKey}`,
        sha256: rawSha,
        bytes: 3,
      }],
    }],
  };
  const plan = buildCleanupPlan({
    now,
    channel: { releaseId: 'source-release' },
    releases: [release],
    objects: [object(identityKey)],
  });

  const protectedSource = plan.protected.find((entry) => entry.key === identityKey);
  assert.deepEqual(protectedSource.kinds, ['source']);
  assert.deepEqual(protectedSource.logicalUrls, ['/stockfish/stockfish-corresponding-source.tar.gz']);
});

test('R2 cleanup plan rejects malformed v2 releases instead of planning around missing identity objects', () => {
  const rawSha = 'c'.repeat(64);
  const identity = {
    encoding: 'identity',
    url: `/artifacts/sha256/${rawSha}/identity`,
    sha256: rawSha,
    bytes: 3,
  };
  const release = (representations) => ({
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'malformed-v2',
    artifacts: [{
      logicalUrl: '/models/model.onnx',
      raw: { sha256: rawSha, bytes: 3 },
      representations,
    }],
  });

  assert.throws(
    () => buildCleanupPlan({ now, releases: [release([])], objects: [] }),
    /V2 artifact has no representations/,
  );
  assert.throws(
    () => buildCleanupPlan({ now, releases: [release([identity, { ...identity }])], objects: [] }),
    /must have exactly one identity representation.*found 2/,
  );
});

test('parseArgs requires hashed opt-in separately from execute', () => {
  const args = parseArgs(['node', 'script', '--execute', '--delete-category', 'legacy-logical-duplicate,legacy-unreferenced-metadata', '--retention-days', '7']);
  assert.equal(args.execute, true);
  assert.equal(args.allowDeleteHashed, false);
  assert.deepEqual([...args.deleteCategories].sort(), ['legacy-logical-duplicate', 'legacy-unreferenced-metadata']);
  assert.equal(args.retentionDays, 7);
});

test('listR2Objects follows Cloudflare result_info cursors across pages', async (t) => {
  const calls = [];
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url) => {
    calls.push(String(url));
    const isSecondPage = String(url).includes('cursor=next-page');
    return new Response(JSON.stringify({
      success: true,
      result: isSecondPage ? [object('b')] : [object('a')],
      result_info: isSecondPage ? { cursor: undefined } : { cursor: 'next-page' },
    }), { status: 200, headers: { 'content-type': 'application/json' } });
  };

  const objects = await listR2Objects({ accountId: 'account', apiToken: 'token', bucket: 'bucket' });

  assert.deepEqual(objects.map((entry) => entry.key), ['a', 'b']);
  assert.equal(calls.length, 2);
  assert.match(calls[1], /cursor=next-page/);
});

test('public docs do not link directly to the legacy R2 dev bucket', async () => {
  const docsPage = await readFile('src/routes/docs/+page.svelte', 'utf8');
  assert.doesNotMatch(docsPage, /r2\.dev|pub-c3fb64db6e434c738bc86cb1a56d6384/);
});
