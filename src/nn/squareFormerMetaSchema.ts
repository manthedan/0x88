/**
 * Valibot schema for the SquareFormer meta JSON fetched by
 * browserRuntimeEvaluator.ts (public/models/*.meta.json). Loose object: the
 * training pipeline emits many hyperparameter fields the browser does not
 * consume. Every field of the SquareFormerMeta interface is mirrored so the
 * parsed output stays assignable to it. `action_value_move_encoding` is null
 * in current meta files and is normalized to undefined.
 */

import * as v from 'valibot';

export const SquareFormerMetaSchema = v.looseObject({
  kind: v.picklist(['squareformer', 'squareformer_v2']),
  variant: v.optional(v.string()),
  input_dim: v.number(),
  token_features: v.optional(v.number()),
  input_mode: v.optional(v.string()),
  input_format: v.optional(v.string()),
  policy_size: v.number(),
  history_plies: v.number(),
  relation_bias: v.optional(v.boolean()),
  av_head_exported: v.optional(v.boolean()),
  action_value_move_encoding: v.optional(
    v.pipe(
      v.nullable(v.string()),
      v.transform((value) => value ?? undefined),
    ),
  ),
  max_legal_moves: v.optional(v.number()),
  onnx_fixed_legal_moves: v.optional(v.number()),
  outputs: v.optional(v.array(v.string())),
  onnx_dynamic_batch: v.optional(v.boolean()),
  board_normalization: v.optional(v.string()),
  input_index_dtype: v.optional(v.string()),
  attack_summary_feature_count: v.optional(v.number()),
  attack_summary_schema: v.optional(v.nullable(v.string())),
  attack_summary_scale: v.optional(v.number()),
});
