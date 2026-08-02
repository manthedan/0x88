#!/usr/bin/env node
import { pathToFileURL } from 'node:url';
import { artifactKeyFromReleaseUrl, buildArtifactReleaseCatalog, isArtifactReleaseV2 } from './engine_artifact_registry.mjs';
import { parseScriptArgs } from './lib/cli.mjs';

const DEFAULT_BUCKET = 'browser-chess-models';
const DEFAULT_RETENTION_DAYS = 90;
const CONTROL_PREFIXES = ['channels/', 'releases/'];
const HASHED_PREFIX = 'artifacts/sha256/';
const SOURCE_ARCHIVE_RE = /(?:corresponding-source|source).*\.tar\.gz$/i;
const SHA_ONLY_V2_KEY_RE = /^artifacts\/sha256\/[a-f0-9]{64}\/(?:identity|br\/[a-f0-9]{64})$/;
const SAFE_DELETE_CATEGORIES = new Set(['legacy-logical-duplicate', 'legacy-unreferenced-metadata']);

const USAGE = `Usage: node scripts/plan_r2_artifact_cleanup.mjs [options]\n\nBuilds a conservative R2 cleanup plan for the artifact bucket. Dry-run is the default.\n\nOptions:\n  --bucket NAME              R2 bucket (default ${DEFAULT_BUCKET})\n  --account-id ID            Cloudflare account id (or CLOUDFLARE_ACCOUNT_ID)\n  --api-token TOKEN          Cloudflare API token (or CLOUDFLARE_API_TOKEN)\n  --retention-days N         Minimum age before hashed orphan deletion candidates (default ${DEFAULT_RETENTION_DAYS})\n  --execute                  Delete selected candidates; default is dry-run only\n  --delete-category NAME     Candidate category to delete. Repeatable or comma-separated.\n                             Safe categories: ${[...SAFE_DELETE_CATEGORIES].join(', ')}\n  --allow-delete-hashed      Allow deleting hashed-orphan candidates too; requires --execute and --delete-category hashed-orphan\n  --now ISO                  Override current time for deterministic tests\n  --json                     Emit only JSON\n  -h, --help                 Show help\n\nThe script never deletes channels/ or releases/. It never deletes any\nartifacts/sha256/* object referenced by a retained release manifest.\n`;

export function parseArgs(argv) {
  const args = parseScriptArgs(argv.slice(2), {
    options: {
      bucket: { type: 'string', default: process.env.LC0_R2_ARTIFACT_BUCKET || DEFAULT_BUCKET },
      'account-id': { type: 'string', default: process.env.CLOUDFLARE_ACCOUNT_ID },
      'api-token': { type: 'string', default: process.env.CLOUDFLARE_API_TOKEN },
      'retention-days': { type: 'string', default: String(DEFAULT_RETENTION_DAYS) },
      'delete-category': { type: 'string', multiple: true, default: [] },
      now: { type: 'string' },
      execute: { type: 'boolean', default: false },
      'allow-delete-hashed': { type: 'boolean', default: false },
      json: { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.retentionDays = Number(args.retentionDays);
  args.deleteCategories = new Set();
  for (const category of args.deleteCategory
    .flatMap((value) => value.split(','))
    .map((value) => value.trim())
    .filter(Boolean))
    args.deleteCategories.add(category);
  delete args.deleteCategory;
  args.now = args.now !== undefined ? new Date(args.now) : new Date();
  if (!Number.isFinite(args.retentionDays) || args.retentionDays < 0) throw new Error('--retention-days must be a non-negative number');
  if (Number.isNaN(args.now.getTime())) throw new Error('--now must be a valid ISO timestamp');
  return args;
}

function objectAgeDays(object, now) {
  const modified = new Date(object.last_modified ?? object.uploaded ?? object.modified ?? 0);
  if (Number.isNaN(modified.getTime())) return undefined;
  return (now.getTime() - modified.getTime()) / 86_400_000;
}

export function artifactKeyFromUrl(raw) {
  return artifactKeyFromReleaseUrl(raw);
}

function logicalKeyFromUrl(raw) {
  if (!raw) return undefined;
  const url = new URL(raw, 'https://assets.0x88.app');
  return url.pathname.replace(/^\/+/, '');
}

function addRef(map, key, releaseId) {
  if (!key) return;
  if (!map.has(key)) map.set(key, new Set());
  map.get(key).add(releaseId);
}

function validatedMigrationSourceKey(artifact) {
  const migration = artifact.migrationSource;
  if (!migration) return undefined;
  const logicalUrl = artifact.logicalUrl ?? artifact.name ?? 'unknown artifact';
  const rawSha256 = artifact.raw?.sha256?.toLowerCase();
  const match = migration.key?.match(/^artifacts\/sha256\/([a-f0-9]{64})\/([^/]+)$/);
  const sourceUrlKey = migration.url ? artifactKeyFromReleaseUrl(migration.url) : undefined;
  if (
    migration.schema !== 'lc0_browser.artifact_migration_source.v1' ||
    !migration.releaseId ||
    migration.releaseId !== artifact.carriedForwardFrom ||
    match?.[1] !== rawSha256 ||
    match?.[2] === 'identity' ||
    sourceUrlKey !== migration.key
  ) {
    throw new Error(`Invalid v1 migration source metadata for ${logicalUrl}; cleanup requires manual review`);
  }
  return migration.key;
}

function artifactReferences(releases) {
  const catalog = buildArtifactReleaseCatalog(releases);
  const refs = new Map(
    [...catalog].map(([key, entry]) => [
      key,
      {
        releases: new Set(entry.releases),
        logicalUrls: entry.logicalUrls,
        kinds: entry.kinds,
      },
    ]),
  );
  for (const release of releases) {
    if (!isArtifactReleaseV2(release)) continue;
    const releaseId = release.releaseId ?? release.id ?? 'unknown-release';
    for (const artifact of release.artifacts ?? []) {
      const key = validatedMigrationSourceKey(artifact);
      if (!key) continue;
      const existing = refs.get(key) ?? { releases: new Set(), logicalUrls: [], kinds: [] };
      existing.releases.add(releaseId);
      const logicalUrl = artifact.logicalUrl ?? artifact.name;
      if (logicalUrl && !existing.logicalUrls.includes(logicalUrl)) existing.logicalUrls.push(logicalUrl);
      if (artifact.kind && !existing.kinds.includes(artifact.kind)) existing.kinds.push(artifact.kind);
      refs.set(key, existing);
    }
  }
  return refs;
}

function isControlKey(key) {
  return CONTROL_PREFIXES.some((prefix) => key.startsWith(prefix));
}

function isLegacyMetadataKey(key) {
  return key.endsWith('.manifest.json') || key.endsWith('/manifest.json') || key.endsWith('/NOTICE.md') || key.endsWith('/README.md');
}

function publicObject(object) {
  return {
    key: object.key,
    size: Number(object.size ?? 0),
    lastModified: object.last_modified ?? object.uploaded ?? object.modified,
    storageClass: object.storage_class,
  };
}

function candidateFrom(object, category, reason, extra = {}) {
  return {
    ...publicObject(object),
    category,
    reason,
    ...extra,
  };
}

function protectedFrom(object, reason, extra = {}) {
  return {
    ...publicObject(object),
    reason,
    ...extra,
  };
}

export function buildCleanupPlan({ objects, releases, channel, now = new Date(), retentionDays = DEFAULT_RETENTION_DAYS }) {
  const artifactRefs = artifactReferences(releases);
  const logicalRefs = new Map();
  const releasesById = new Map();
  for (const release of releases) {
    const releaseId = release.releaseId ?? release.id ?? 'unknown-release';
    releasesById.set(releaseId, release);
    for (const artifact of release.artifacts ?? []) {
      addRef(logicalRefs, logicalKeyFromUrl(artifact.logicalUrl ?? artifact.name), releaseId);
    }
  }

  const stableReleaseId = channel?.releaseId;
  const stableRelease = stableReleaseId ? releasesById.get(stableReleaseId) : undefined;
  const stableArtifactRefs = new Set();
  const stableLogicalRefs = new Set();
  if (stableRelease) {
    for (const key of artifactReferences([stableRelease]).keys()) stableArtifactRefs.add(key);
    for (const artifact of stableRelease.artifacts ?? []) {
      const logicalKey = logicalKeyFromUrl(artifact.logicalUrl ?? artifact.name);
      if (logicalKey) stableLogicalRefs.add(logicalKey);
    }
  }

  const candidates = [];
  const protectedObjects = [];
  const missingReferencedArtifacts = [];
  const objectKeys = new Set(objects.map((object) => object.key));
  for (const key of artifactRefs.keys()) {
    if (!objectKeys.has(key)) missingReferencedArtifacts.push({ key, releases: [...artifactRefs.get(key).releases].sort() });
  }

  for (const object of objects) {
    const key = object.key;
    if (isControlKey(key)) {
      protectedObjects.push(protectedFrom(object, 'control manifest object'));
      continue;
    }

    if (key.startsWith(HASHED_PREFIX)) {
      const refs = artifactRefs.get(key);
      if (refs?.releases.size) {
        protectedObjects.push(
          protectedFrom(object, stableArtifactRefs.has(key) ? 'referenced by stable release' : 'referenced by retained release', {
            releases: [...refs.releases].sort(),
            logicalUrls: refs.logicalUrls,
            kinds: refs.kinds,
          }),
        );
        continue;
      }
      const ageDays = objectAgeDays(object, now);
      if (SHA_ONLY_V2_KEY_RE.test(key)) {
        protectedObjects.push(
          protectedFrom(object, 'unreferenced SHA-only v2 artifact; logical filename and artifact kind are unavailable, so manual review is required', {
            ageDays,
          }),
        );
        continue;
      }
      if (SOURCE_ARCHIVE_RE.test(key)) {
        protectedObjects.push(
          protectedFrom(object, 'unreferenced source archive; preserve for license/source obligations unless manually reviewed', { ageDays }),
        );
        continue;
      }
      if (ageDays === undefined || ageDays < retentionDays) {
        protectedObjects.push(protectedFrom(object, 'unreferenced hashed artifact inside retention window', { ageDays, retentionDays }));
        continue;
      }
      candidates.push(candidateFrom(object, 'hashed-orphan', `unreferenced by retained releases and older than ${retentionDays} days`, { ageDays }));
      continue;
    }

    const logicalRefsForObject = logicalRefs.get(key);
    if (logicalRefsForObject?.size) {
      protectedObjects.push(
        protectedFrom(
          object,
          stableLogicalRefs.has(key)
            ? 'legacy logical object referenced by active stable release; preserve for migration client compatibility'
            : 'legacy logical object referenced by retained release; preserve for rollback and migration client compatibility',
          { releases: [...logicalRefsForObject].sort() },
        ),
      );
      continue;
    }
    if (isLegacyMetadataKey(key)) {
      candidates.push(candidateFrom(object, 'legacy-unreferenced-metadata', 'legacy metadata/notice object not referenced by retained release manifests'));
      continue;
    }
    protectedObjects.push(protectedFrom(object, 'unclassified non-content-addressed object; manual review required'));
  }

  const summaryByCategory = {};
  for (const candidate of candidates) {
    summaryByCategory[candidate.category] ??= { count: 0, bytes: 0 };
    summaryByCategory[candidate.category].count += 1;
    summaryByCategory[candidate.category].bytes += candidate.size;
  }

  return {
    schema: 'lc0_browser.r2_artifact_cleanup_plan.v1',
    bucket: undefined,
    generatedAt: now.toISOString(),
    retentionDays,
    stableReleaseId,
    objectCount: objects.length,
    releaseCount: releases.length,
    catalogObjectCount: artifactRefs.size,
    candidateCount: candidates.length,
    candidateBytes: candidates.reduce((sum, candidate) => sum + candidate.size, 0),
    summaryByCategory,
    missingReferencedArtifacts,
    candidates: candidates.sort((a, b) => a.category.localeCompare(b.category) || a.key.localeCompare(b.key)),
    protectedCount: protectedObjects.length,
    protectedBytes: protectedObjects.reduce((sum, object) => sum + object.size, 0),
    protected: protectedObjects.sort((a, b) => a.key.localeCompare(b.key)),
  };
}

async function cfFetch(args, path, init = {}, options = {}) {
  if (!args.accountId) throw new Error('--account-id or CLOUDFLARE_ACCOUNT_ID is required');
  if (!args.apiToken) throw new Error('--api-token or CLOUDFLARE_API_TOKEN is required');
  const response = await fetch(`https://api.cloudflare.com/client/v4${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${args.apiToken}`,
      'User-Agent': '0x88-r2-cleanup-plan',
      ...(init.headers ?? {}),
    },
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`Cloudflare API ${response.status} for ${path}: ${text}`);
  if (!text) return undefined;
  const json = JSON.parse(text);
  if (json.success === false) throw new Error(`Cloudflare API error for ${path}: ${JSON.stringify(json.errors ?? json)}`);
  return options.envelope ? json : (json.result ?? json);
}

export async function listR2Objects(args) {
  const objects = [];
  let cursor;
  do {
    const params = new URLSearchParams({ per_page: '1000' });
    if (cursor) params.set('cursor', cursor);
    const envelope = await cfFetch(args, `/accounts/${args.accountId}/r2/buckets/${args.bucket}/objects?${params}`, {}, { envelope: true });
    const result = envelope?.result;
    const page = Array.isArray(result) ? result : (result?.objects ?? []);
    objects.push(...page);
    cursor = envelope?.result_info?.cursor ?? result?.cursor;
  } while (cursor);
  return objects;
}

async function getR2TextObject(args, key) {
  const result = await cfFetch(args, `/accounts/${args.accountId}/r2/buckets/${args.bucket}/objects/${key}`, { method: 'GET' });
  if (typeof result === 'string') return result;
  return JSON.stringify(result);
}

async function loadReleaseManifests(args, objects) {
  const releases = [];
  let channel;
  for (const object of objects.filter((entry) => entry.key.startsWith('releases/') && entry.key.endsWith('.json'))) {
    releases.push(JSON.parse(await getR2TextObject(args, object.key)));
  }
  const stableObject = objects.find((entry) => entry.key === 'channels/stable.json');
  if (stableObject) channel = JSON.parse(await getR2TextObject(args, stableObject.key));
  return { releases, channel };
}

function validateDeleteArgs(args, plan) {
  if (!args.execute) return [];
  if (!args.deleteCategories.size) throw new Error('--execute requires at least one --delete-category');
  for (const category of args.deleteCategories) {
    if (category === 'hashed-orphan') {
      if (!args.allowDeleteHashed) throw new Error('Deleting hashed-orphan requires --allow-delete-hashed');
      continue;
    }
    if (!SAFE_DELETE_CATEGORIES.has(category)) throw new Error(`Unsupported delete category: ${category}`);
  }
  const selected = plan.candidates.filter((candidate) => args.deleteCategories.has(candidate.category));
  for (const candidate of selected) {
    if (candidate.key.startsWith(HASHED_PREFIX) && !args.allowDeleteHashed)
      throw new Error(`Refusing to delete hashed object without --allow-delete-hashed: ${candidate.key}`);
    if (isControlKey(candidate.key)) throw new Error(`Refusing to delete control manifest object: ${candidate.key}`);
  }
  return selected;
}

async function deleteR2Object(args, key) {
  await cfFetch(args, `/accounts/${args.accountId}/r2/buckets/${args.bucket}/objects/${key}`, { method: 'DELETE' });
}

export async function main(argv = process.argv) {
  const args = parseArgs(argv);
  const objects = await listR2Objects(args);
  const { releases, channel } = await loadReleaseManifests(args, objects);
  const plan = buildCleanupPlan({ objects, releases, channel, now: args.now, retentionDays: args.retentionDays });
  plan.bucket = args.bucket;
  plan.execute = args.execute;
  plan.deleteCategories = [...args.deleteCategories].sort();
  const selected = validateDeleteArgs(args, plan);
  plan.selectedForDeletion = selected.map((candidate) => candidate.key);
  if (args.execute) {
    for (const candidate of selected) await deleteR2Object(args, candidate.key);
    plan.deletedCount = selected.length;
    plan.deletedBytes = selected.reduce((sum, candidate) => sum + candidate.size, 0);
  }
  if (args.json) console.log(JSON.stringify(plan, null, 2));
  else {
    console.log(
      JSON.stringify(
        {
          schema: plan.schema,
          bucket: plan.bucket,
          execute: plan.execute,
          stableReleaseId: plan.stableReleaseId,
          objectCount: plan.objectCount,
          releaseCount: plan.releaseCount,
          candidateCount: plan.candidateCount,
          candidateBytes: plan.candidateBytes,
          summaryByCategory: plan.summaryByCategory,
          missingReferencedArtifacts: plan.missingReferencedArtifacts,
          selectedForDeletion: plan.selectedForDeletion,
          deletedCount: plan.deletedCount,
          deletedBytes: plan.deletedBytes,
        },
        null,
        2,
      ),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
