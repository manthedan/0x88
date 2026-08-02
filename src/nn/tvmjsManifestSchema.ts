/**
 * Valibot schema for the TVMJS/WebGPU bundle manifest fetched by
 * squareformerTvmjsWebgpuEvaluator.ts (staged under
 * public/runtimes/centipawn-tvmjs-webgpu/ by
 * scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs). Loose objects throughout:
 * the manifest carries provenance fields the evaluator does not consume.
 * Mirrors the TvmjsManifest type kept in the evaluator module.
 */

import * as v from 'valibot';

export const TvmjsManifestModelSchema = v.looseObject({
  batch: v.number(),
  wasm: v.string(),
  bytes: v.optional(v.number()),
  sha256: v.optional(v.string()),
});

export const TvmjsManifestFileSchema = v.looseObject({
  path: v.string(),
  bytes: v.optional(v.number()),
  sha256: v.optional(v.string()),
});

export const TvmjsManifestSchema = v.looseObject({
  schema: v.string(),
  modelFamily: v.string(),
  dtype: v.string(),
  target: v.string(),
  requiredFeatures: v.optional(v.array(v.string())),
  runtime: v.optional(
    v.looseObject({
      tvmjsBundle: v.optional(v.string()),
      tvmjsRuntimeWasm: v.optional(v.string()),
    }),
  ),
  models: v.optional(v.array(TvmjsManifestModelSchema)),
  files: v.optional(v.array(TvmjsManifestFileSchema)),
});
