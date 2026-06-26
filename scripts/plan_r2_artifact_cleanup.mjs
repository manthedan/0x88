#!/usr/bin/env node
import { pathToFileURL } from 'node:url';

const DEFAULT_BUCKET = 'browser-chess-models';
const DEFAULT_RETENTION_DAYS = 90;
const CONTROL_PREFIXES = ['channels/', 'releases/'];
const HASHED_PREFIX = 'artifacts/sha256/';
const SOURCE_ARCHIVE_RE = /(?:corresponding-source|source).*\.tar\.gz$/i;
const SAFE_DELETE_CATEGORIES = new Set([
  'legacy-logical-duplicate',
  'legacy-unreferenced-metadata',
]);

function usage() {
  console.log(`Usage: node scripts/plan_r2_artifact_cleanup.mjs [options]\n\nBuilds a conservative R2 cleanup plan for the artifact bucket. Dry-run is the default.\n\nOptions:\n  --bucket NAME              R2 bucket (default ${DEFAULT_BUCKET})\n  --account-id ID            Cloudflare account id (or CLOUDFLARE_ACCOUNT_ID)\n  --api-token TOKEN          Cloudflare API token (or CLOUDFLARE_API_TOKEN)\n  --retention-days N         Minimum age before hashed orphan deletion candidates (default ${DEFAULT_RETENTION_DAYS})\n  --execute                  Delete selected candidates; default is dry-run only\n  --delete-category NAME     Candidate category to delete. Repeatable or comma-separated.\n                             Safe categories: ${[...SAFE_DELETE_CATEGORIES].join(', ')}\n  --allow-delete-hashed      Allow deleting hashed-orphan candidates too; requires --execute and --delete-category hashed-orphan\n  --now ISO                  Override current time for deterministic tests\n  --json                     Emit only JSON\n  -h, --help                 Show help\n\nThe script never deletes channels/ or releases/. It never deletes any\nartifacts/sha256/* object referenced by a retained release manifest.\n`);
}

export function parseArgs(argv) {
  const args = {
    bucket: process.env.LC0_R2_ARTIFACT_BUCKET || DEFAULT_BUCKET,
    accountId: process.env.CLOUDFLARE_ACCOUNT_ID,
    apiToken: process.env.CLOUDFLARE_API_TOKEN,
    retentionDays: DEFAULT_RETENTION_DAYS,
    execute: false,
    deleteCategories: new Set(),
    allowDeleteHashed: false,
    now: new Date(),
    json: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--bucket' && next) { args.bucket = next; i += 1; continue; }
    if (arg === '--account-id' && next) { args.accountId = next; i += 1; continue; }
    if (arg === '--api-token' && next) { args.apiToken = next; i += 1; continue; }
    if (arg === '--retention-days' && next) { args.retentionDays = Number(next); i += 1; continue; }
    if (arg === '--delete-category' && next) {
      for (const category of next.split(',').map((value) => value.trim()).filter(Boolean)) args.deleteCategories.add(category);
      i += 1;
      continue;
    }
    if (arg === '--now' && next) { args.now = new Date(next); i += 1; continue; }
    if (arg === '--execute') { args.execute = true; continue; }
    if (arg === '--allow-delete-hashed') { args.allowDeleteHashed = true; continue; }
    if (arg === '--json') { args.json = true; continue; }
    if (arg === '-h' || arg === '--help') { args.help = true; continue; }
    throw new Error(`Unknown argument: ${arg}`);
  }
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
  if (!raw) return undefined;
  const url = new URL(raw, 'https://assets.0x88.app');
  const key = url.pathname.replace(/^\/+/, '');
  return key.startsWith(HASHED_PREFIX) ? key : undefined;
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
  const artifactRefs = new Map();
  const logicalRefs = new Map();
  const releasesById = new Map();
  for (const release of releases) {
    const releaseId = release.releaseId ?? release.id ?? 'unknown-release';
    releasesById.set(releaseId, release);
    for (const artifact of release.artifacts ?? []) {
      addRef(artifactRefs, artifactKeyFromUrl(artifact.artifactUrl), releaseId);
      addRef(logicalRefs, logicalKeyFromUrl(artifact.logicalUrl), releaseId);
    }
  }

  const stableReleaseId = channel?.releaseId;
  const stableRelease = stableReleaseId ? releasesById.get(stableReleaseId) : undefined;
  const stableArtifactRefs = new Set();
  const stableLogicalRefs = new Set();
  if (stableRelease) {
    for (const artifact of stableRelease.artifacts ?? []) {
      const artifactKey = artifactKeyFromUrl(artifact.artifactUrl);
      const logicalKey = logicalKeyFromUrl(artifact.logicalUrl);
      if (artifactKey) stableArtifactRefs.add(artifactKey);
      if (logicalKey) stableLogicalRefs.add(logicalKey);
    }
  }

  const candidates = [];
  const protectedObjects = [];
  const missingReferencedArtifacts = [];
  const objectKeys = new Set(objects.map((object) => object.key));
  for (const key of artifactRefs.keys()) {
    if (!objectKeys.has(key)) missingReferencedArtifacts.push({ key, releases: [...artifactRefs.get(key)].sort() });
  }

  for (const object of objects) {
    const key = object.key;
    if (isControlKey(key)) {
      protectedObjects.push(protectedFrom(object, 'control manifest object'));
      continue;
    }

    if (key.startsWith(HASHED_PREFIX)) {
      const refs = artifactRefs.get(key);
      if (refs?.size) {
        protectedObjects.push(protectedFrom(object, stableArtifactRefs.has(key) ? 'referenced by stable release' : 'referenced by retained release', { releases: [...refs].sort() }));
        continue;
      }
      const ageDays = objectAgeDays(object, now);
      if (SOURCE_ARCHIVE_RE.test(key)) {
        protectedObjects.push(protectedFrom(object, 'unreferenced source archive; preserve for license/source obligations unless manually reviewed', { ageDays }));
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
    if (stableLogicalRefs.has(key)) {
      candidates.push(candidateFrom(object, 'legacy-logical-duplicate', 'non-content-addressed object shadowed by current stable release manifest; Worker resolves this logical path to artifacts/sha256', { releases: [...(logicalRefsForObject ?? [])].sort() }));
      continue;
    }
    if (logicalRefsForObject?.size) {
      protectedObjects.push(protectedFrom(object, 'legacy object referenced only by retained non-stable release logicalUrl', { releases: [...logicalRefsForObject].sort() }));
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

async function cfFetch(args, path, init = {}) {
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
  return json.result ?? json;
}

async function listR2Objects(args) {
  const objects = [];
  let cursor;
  do {
    const params = new URLSearchParams({ per_page: '1000' });
    if (cursor) params.set('cursor', cursor);
    const result = await cfFetch(args, `/accounts/${args.accountId}/r2/buckets/${args.bucket}/objects?${params}`);
    const page = Array.isArray(result) ? result : (result.objects ?? []);
    objects.push(...page);
    cursor = Array.isArray(result) ? undefined : result.cursor;
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
    if (candidate.key.startsWith(HASHED_PREFIX) && !args.allowDeleteHashed) throw new Error(`Refusing to delete hashed object without --allow-delete-hashed: ${candidate.key}`);
    if (isControlKey(candidate.key)) throw new Error(`Refusing to delete control manifest object: ${candidate.key}`);
  }
  return selected;
}

async function deleteR2Object(args, key) {
  await cfFetch(args, `/accounts/${args.accountId}/r2/buckets/${args.bucket}/objects/${key}`, { method: 'DELETE' });
}

export async function main(argv = process.argv) {
  const args = parseArgs(argv);
  if (args.help) { usage(); return; }
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
    console.log(JSON.stringify({
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
    }, null, 2));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
