/**
 * Valibot schemas for the remote JSON manifests fetched by modelCache.ts:
 * the per-directory model manifest (public/models/lc0/manifest.json, written
 * by scripts/lc0_prepare_model_assets.mjs) and the artifact channel/release
 * manifests (written by scripts/write_artifact_release_manifests.mjs).
 * All schemas are loose: unknown keys pass through untouched and only the
 * fields the loader relies on are validated.
 */

import * as v from 'valibot';

export const Lc0ModelManifestEntrySchema = v.looseObject({
  file: v.string(),
  url: v.string(),
  /** Optional immutable/content-addressed URL for the bytes named by `url`. */
  artifactUrl: v.optional(v.string()),
  bytes: v.optional(v.number()),
  sha256: v.optional(v.string()),
});

export const Lc0ModelManifestSchema = v.looseObject({
  models: v.optional(v.array(Lc0ModelManifestEntrySchema)),
});

export const Lc0ArtifactChannelManifestSchema = v.looseObject({
  releaseManifestUrl: v.optional(v.string()),
  releaseUrl: v.optional(v.string()),
});

export const Lc0ArtifactRepresentationSchema = v.looseObject({
  encoding: v.picklist(['identity', 'br']),
  url: v.string(),
  bytes: v.number(),
  sha256: v.string(),
});

export const Lc0ArtifactReleaseEntrySchema = v.looseObject({
  logicalUrl: v.optional(v.string()),
  name: v.optional(v.string()),
  file: v.optional(v.string()),
  artifactUrl: v.optional(v.string()),
  bytes: v.optional(v.number()),
  sha256: v.optional(v.string()),
  raw: v.optional(v.looseObject({ bytes: v.number(), sha256: v.string() })),
  representations: v.optional(v.array(Lc0ArtifactRepresentationSchema)),
});

export const Lc0ArtifactReleaseManifestSchema = v.looseObject({
  artifacts: v.optional(v.array(Lc0ArtifactReleaseEntrySchema)),
});
