# MoveFormer sidecar cache v1

Goal: provide lightweight legal/candidate move tokens for `MoveFormer-CNN-AV` without replacing the existing supervised/residual caches.

## Format

Directory with `meta.json` plus memmap arrays.

Required arrays:

```text
policy_index.int64             [rows]
policy_legal_slot.int16         [rows]       # slot in legal move list, -1 if not found
wdl.float32                     [rows, 3]
q.float32                       [rows]
legal_policy_indices.int64      [rows, max_legal_moves]  # -1 padded
legal_action_ids.int64          [rows, max_legal_moves]  # -1 padded, TS moveToActionId mapping
legal_uci.uint8                 [rows, max_legal_moves, 5] # ASCII bytes, 0 padded
legal_features.float32          [rows, max_legal_moves, num_move_features]
legal_mask.float32              [rows, max_legal_moves]
```

Optional arrays:

```text
x.int8                          [rows, input_planes, 8, 8]
```

`x.int8` is included for smoke/prototype caches. Full-scale builds can omit it and join with existing residual caches by source shard/row order or rebuild a combined feature cache later.

## Move feature names v1

```text
moving_piece_type               # 1 pawn .. 6 king
captured_piece_type             # 0 none, 1 pawn .. 6 king
promotion_type                  # 0 none, n=1,b=2,r=3,q=4
is_capture
is_check
is_castle
is_promotion
is_en_passant
from_attacked_by_enemy_pre
from_defended_by_own_pre
to_attacked_by_enemy_after
to_defended_by_own_after
to_enemy_attackers_after_capped8
to_own_defenders_after_capped8
moving_piece_value
captured_piece_value
material_delta                  # captured + promo_gain, simple piece values
from_piece_pinned_pre
king_distance_to_enemy_after
king_distance_to_own_after
```

These are intentionally minimal attack-map features embedded in move tokens. Full square-level attack planes and ray tokens should be separate v1.5/v2 caches after the first MoveFormer baseline works.

## Policy/action mappings

- `policy_index` uses `uci_queen_knight_promo_v1` from `training.train_residual_torch.fixed_policy_moves()`.
- `legal_action_ids` match `src/chess/moveCodec.ts::moveToActionId`: `(from * 64 + to) * 5 + promo`, with promo `n=1,b=2,r=3,q=4`.

## Validation metrics

Builder emits:

```text
METRIC rows_written
METRIC policy_target_legal_rate
METRIC policy_index_found_rate
METRIC legal_truncation_rate
METRIC avg_legal_moves
```

A valid supervised cache should have policy target legal and policy-index-found rates close to 1.0.

## ONNX export and legal-length buckets

Current PyTorch/ONNX export of `nn.TransformerEncoderLayer` is reliable for dynamic batch size, but not for truly dynamic legal-move length: the exported attention graph can bake reshape constants from the traced legal length. Therefore MoveFormer runtime exports should be treated as fixed-K buckets rather than one fully dynamic-K model.

Recommended buckets:

```text
K=32
K=64
K=128
```

Runtime selection:

```text
legal_moves <= 32  -> K32 model
legal_moves <= 64  -> K64 model
else               -> K128 model
```

Inputs must be padded to the selected bucket size and padded slots must use `legal_mask=0`. Policy logits and action values for masked slots must be ignored by search.

The trainer supports bucket export with:

```bash
--onnx-legal-ks 32,64,128
```

It writes sibling files such as `model_k32.onnx`, `model_k64.onnx`, and `model_k128.onnx` plus per-bucket metadata. True dynamic legal length remains a later optimization, likely requiring a custom ONNX-friendly masked-attention implementation instead of stock PyTorch `MultiheadAttention` export.

## ChessBench AV candidate training path

The MoveFormer trainer can also train directly from compact ChessBench AV cache collections instead of the supervised sidecar:

```bash
--av-cache data/public_teacher_overlays/chessbench_full_policy_value_direct_top48_32shards_v1/collection_manifest.json
```

In this mode:

- board planes are reconstructed from compact ChessBench square tokens;
- candidate move classes are converted from ChessBench compact move ids to runtime action ids `(from * 64 + to) * 5 + promo`;
- the policy target is the highest-value candidate in the C=48 set;
- the AV loss is applied to all masked candidate values, not only the played/chosen move;
- WDL is weakly derived from the best candidate Q, so WDL metrics from this mode are secondary.

This is the first "true AV" MoveFormer path. It still uses lightweight/generated move features for the AV cache path; richer python-chess attack/check/pin features can be precomputed later if the C=48 model looks promising.
