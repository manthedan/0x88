/**
 * Valibot schema for the staged whole-model runtime manifest fetched by
 * wholeOnnxWebgpuEvaluator.ts (public/runtimes/lc0-tvmjs-webgpu/..., staged by
 * scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs). Loose objects throughout:
 * the manifest carries compiler provenance and tensor-cache metadata the
 * evaluator does not consume. Mirrors the RuntimeManifest interface kept in
 * the evaluator module.
 */

import * as v from 'valibot';

export const WholeOnnxRuntimeManifestModelSchema = v.looseObject({
  batch: v.number(),
  wasm: v.string(),
  bytes: v.optional(v.number()),
  sha256: v.optional(v.string()),
});

export const WholeOnnxRuntimeManifestSchema = v.looseObject({
  runtime: v.optional(v.record(v.string(), v.optional(v.string()))),
  parameterStrategy: v.optional(v.looseObject({ current: v.optional(v.string()) })),
  tensorCache: v.optional(
    v.looseObject({
      directory: v.optional(v.string()),
      manifest: v.optional(v.string()),
    }),
  ),
  models: v.array(WholeOnnxRuntimeManifestModelSchema),
  files: v.optional(
    v.array(
      v.looseObject({
        path: v.string(),
        sha256: v.optional(v.string()),
      }),
    ),
  ),
});
