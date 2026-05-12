---
id: task.lc0_distillation_adapter_and_scaling
type: task
title: Task - LC0 distillation adapter and scaling path
status: planned
created: 2026-05-12
updated: 2026-05-12
priority: high
depends_on:
  - "[[Design - LC0 search-distillation pipeline]]"
  - "[[Decision - LC0 architecture funnel and deployability frontier]]"
  - "[[Design - Candidate frontier cards]]"
risks:
  - "[[Risk - Move-map mismatch]]"
  - "[[Risk - LC0 data provenance and license]]"
agent_summary: >
  Implementation checklist for going from LC0 public chunk adapter proof to 10M and 100M distillation on the two promoted lanes: bt4_h2_flip_av_relbank_d256_l8 and mf80_av_top48_10m_flipped_moverel_gate. Scale is blocked until adapter legality, normalization, provenance, and tiny-overfit gates pass.
---

# Task - LC0 distillation adapter and scaling path

## Target lanes

```text
BT4/SquareFormer:
  bt4_h2_flip_av_relbank_d256_l8

MF80:
  mf80_av_top48_10m_flipped_moverel_gate
```

## Hard gates before scale

Do not run 10M/100M LC0 training until these pass:

```text
unknown_move_rate == 0
illegal positive policy mass == 0
total_drop_rate <= 1% or investigated
total_drop_rate <= 5% hard block
10M -> 100M unexplained_drop_rate <= 0.5%
board_normalization=stm_white_rankflip_v1 declared
teacher=lc0_public provenance recorded
license/provenance manifest captured
1k-10k overfit sanity passes
```

## Phase A - Source/provenance bootstrap

- [ ] Pick one modern/strong LC0 public run/test/chunk for tiny proof.
- [ ] Download only a tiny sample first.
- [ ] Capture source URL, run/test, chunk path, checksum, byte size, license text, format/version, and local path.
- [ ] Write manifest under `data/external/lc0_public/manifests/`.
- [ ] Keep raw chunks immutable under `data/external/lc0_public/raw/`.

## Phase B - Chunk inspector

- [ ] Add `scripts/lc0_chunk_inspect.py`.
- [ ] Count records without full conversion.
- [ ] Print detected field availability: planes, probabilities, root/best/played Q/D/M, result, visits, policy_kld, best_idx, played_idx.
- [ ] Emit sampled field stats and drop-reason preview.
- [ ] Fail loudly on unknown format or unsupported compression.

## Phase C - Adapter core

- [ ] Add `training/lc0_adapter.py`.
- [ ] Parse LC0 V6 record to normalized internal example.
- [ ] Decode board state and side-to-move perspective.
- [ ] Map LC0 `probabilities[1858]` to Tiny Leela MoveId/action IDs.
- [ ] Filter/renormalize legal sparse policy.
- [ ] Convert Q/D to WDL with `W=(1-D+Q)/2`, `D=D`, `L=(1-D-Q)/2`.
- [ ] Emit auditable JSONL first with sparse `policy_target`.
- [ ] Include drop reasons and sampled rejected records.

## Phase D - Correctness tests

- [ ] Unit tests for LC0 index -> UCI -> MoveId -> UCI.
- [ ] Legal mass and illegal positive mass tests.
- [ ] Castling, en-passant, promotions/underpromotions, check evasions.
- [ ] `best_idx`/`played_idx` legality tests.
- [ ] `stm_white_rankflip_v1` side-flip parity tests.
- [ ] Q/D/WDL perspective tests.
- [ ] Tiny 1k-10k overfit KL/top-k sanity.

## Phase E - Training bridge

- [ ] Weighted expanded top-k cache for the first 10M pilot, default `top_k=8`.
- [ ] Count scale by raw searched positions, not expanded rows.
- [ ] Preserve sparse LC0 policy probabilities in JSONL/cache metadata.
- [ ] Add trainer path for LC0 policy KL + WDL CE; hard-label CE is insufficient for serious 100M+.
- [ ] Before 100M+, implement sparse soft-label KL cache/training or prove weighted expansion is still acceptable.

## Phase F - Pilot and scale

- [ ] 10k-100k adapter proof on tiny/fast path.
- [ ] 10k-100k LC0 sanity on both promoted winners.
- [ ] 10M LC0 pilot on both promoted winners.
- [ ] Emit conversion report and candidate frontier-card core metrics.
- [ ] 100M LC0 for both promoted winners only after drop/correctness/training gates pass.
- [ ] 500M/1B only for one or both if still frontier-relevant.

## Tonight realistic target

Best-case tonight:

```text
source chosen
manifest/checksum/license capture implemented
chunk inspector running on a tiny sample
adapter skeleton implemented
first legality/normalization tests drafted
possibly 1k-10k JSONL conversion if format parsing is straightforward
```

Not safe tonight unless the adapter already works and passes gates:

```text
10M LC0 training
100M LC0 training
```
