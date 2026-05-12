---
created: 2026-05-12
updated: 2026-05-12
project: tiny-neural-chess
id: design.lc0_search_distillation_pipeline
type: design
title: Design - LC0 search-distillation pipeline
status: active
confidence: medium
priority: high
depends_on:
  - [[Design - SquareFormer-AV-PUCT]]
  - [[Concept - Search-improved self-play]]
supports:
  - [[Roadmap - Current Tiny Leela portfolio]]
risks:
  - [[Risk - Move-map mismatch]]
  - [[Risk - LC0 data provenance and license]]
related:
  - [[Design - Candidate frontier cards]]
  - [[Decision - AWS Batch self-play parallelism policy]]
  - [[Decision - LC0 architecture funnel and deployability frontier]]
agent_summary: >
  Make LC0 public search-generated training chunks the primary supervised/distillation lane for MF80 and promoted BT4/SquareFormer architectures, with Gumbel-Zero kept as a separate research lane and ChessBench/Stockfish-style action-value data added later for ranking/regret.
---

# Design - LC0 search-distillation pipeline

Tiny Leela should split training into two primary research lanes:

1. **LC0 search distillation**: supervised training from public LC0 training chunks that contain search-improved policy distributions and Q/D/value-style targets.
2. **Gumbel ZeroShot self-play**: clean zero/self-play research kept separate from supervised bootstrap data.

The expectation is not that Gumbel immediately beats LC0 distillation. LC0 distillation is the pragmatic strength lane; Gumbel is the clean research/control lane that may teach us about bootstrap-free improvement.

## Motivation

Earlier teacher attempts likely underused LC0 by imitating raw network outputs, one-off evaluations, or top moves. LC0 training chunks are closer to the desired target because they contain the expensive search-generated supervision: policy probabilities, root/best/played Q and draw values, result targets, visits, and move indices. That matches Tiny Leela's principle from [[Concept - Search-improved self-play]]: train on search-improved targets, not sampled one-hot moves.

The immediate model targets are the two promoted 10M winners from [[Decision - LC0 architecture funnel and deployability frontier]]:

```text
MF80:
  mf80_av_top48_10m_flipped_moverel_gate

BT4/SquareFormer:
  bt4_h2_flip_av_relbank_d256_l8
```

Do not restart broad 10M architecture ablations while LC0 adapter/proof work is still pending. Existing prototype BT4 checkpoints should not consume broad arena budget unless a materially improved checkpoint is produced or the reopen criteria are met.

## Non-goals

- Do not blend LC0 policy, ChessBench values, Stockfish labels, and Tiny-Leela self-play into one indistinguishable target.
- Do not train on `best_idx` or `played_idx` as a one-hot replacement for the full LC0 policy distribution.
- Do not make Gumbel-Zero depend on LC0 labels; keep lane provenance explicit.
- Do not treat adapter success as assumed. The LC0 move map, board orientation, history, castling, en-passant, and value perspective are core correctness risks.

## Canonical normalized example

Every adapter should emit a normalized internal record before cache/training:

```text
source: lc0_public | chessbench | tiny_selfplay | human_supervised
teacher: lc0_run/test id or dataset id
position:
  fen_or_reconstructed_state
  tokens_64
  history_valid
  board_normalization
legal_moves:
  canonical MoveId = from_square, to_square, promotion
policy_target:
  sparse map MoveId -> probability, normalized over legal moves
value_targets:
  root_q, root_d, result_q, result_d
  wdl_root, wdl_result
  q_bucket_root optional
  plies_left optional
sparse_action_values:
  best_idx move -> best_q/best_d/best_m
  played_idx move -> played_q/played_d/played_m
metadata:
  visits, policy_kld, format_version, transform bits, checksum, source file offset
```

Use `board_normalization=stm_white_rankflip_v1` by default for new cache/train/export artifacts, consistent with [[Operations constraints]]. Raw external chunks stay immutable; only derived caches declare the transform.

## LC0 adapter contract

The adapter has five mandatory stages:

1. **Fetch and manifest**: download a small public LC0 sample first. Record URL, run/test directory, chunk/tar name, byte size, checksum, license file, format version, and local path.
2. **Parse V6 records**: extract board planes/state, `probabilities[1858]`, `root_q/root_d`, `best_q/best_d`, `played_q/played_d`, `result_q/result_d`, moves-left fields, visits, `played_idx`, `best_idx`, and `policy_kld` when present.
3. **Decode/canonicalize board**: convert LC0 state into Tiny Leela's board object and 64 square-token features. Preserve enough history/rule-state to match the selected model family; mark missing/unknown history explicitly.
4. **Map policy**: for each legal Tiny Leela move, compute or look up the corresponding LC0 policy index, read its probability, map it to Tiny Leela `MoveId`, filter illegal mass, and renormalize over legal moves.
5. **Convert values**: convert Q/D fields to WDL from side-to-move perspective:

```text
W = (1 - D + Q) / 2
D = D
L = (1 - D - Q) / 2
```

Clamp/renormalize only after logging out-of-range cases. Keep root, result, best, and played targets separate in metadata until training decides which are primary.

## Required adapter tests

Permanent tests are mandatory before scaling beyond toy samples:

- Policy target mass over legal moves is near 1 after renormalization.
- No illegal move receives nonzero training mass.
- `best_idx` and `played_idx` map to legal moves when the record is usable.
- Promotions, underpromotions, castling, en-passant, and check evasions round-trip through LC0 index -> UCI -> Tiny Leela MoveId -> UCI.
- Side-to-move rank-flip normalization preserves legal moves and maps policy distributions correctly.
- Q/D/WDL perspective flips correctly under board mirroring.
- A 1k-10k record subset can be overfit: policy KL falls sharply and top-k agreement rises.
- Manual spot checks show LC0 top-policy moves are legal and plausible.

If any of these fail, do not interpret model loss or arena play as architecture signal.

## Training phases

### Phase 0: adapter proof

Use 10k-100k records from one modern LC0 source on a tiny/fast model or stable training path. Build manifests, conversion reports, and overfit tests. Output only derived tiny caches and validation summaries. This proves LC0 parsing, move mapping, WDL/Q perspective, and normalization; it is not architecture evidence.

Exit gates:

- zero policy-map validation errors on sampled records,
- no unknown legal move class above a tiny threshold,
- policy KL overfit sanity passes,
- WDL/value perspective tests pass.

### Phase 1: LC0 policy/WDL pretraining

Train both promoted architecture winners after adapter proof:

```text
MF80 winner: bt4-independent original-project lane
  mf80_av_top48_10m_flipped_moverel_gate

BT4/SquareFormer winner: compact-transformer lane
  bt4_h2_flip_av_relbank_d256_l8
```

Use 10k-100k LC0 sanity first, then a 10M LC0 pilot on both if correctness gates pass. Both can advance to 100M+ if no eval/correctness blocker appears.

Initial loss:

```text
L_lc0 =
  1.00 * KL(policy_lc0 || policy_model)
+ 1.00 * CE(wdl_root, wdl_model)
+ 0.25 * CE(wdl_result, wdl_model_aux or same head)
+ 0.25 * q_bucket/root_q loss if enabled
+ 0.10 * moves_left loss if enabled
```

Sparse action-value from LC0 best/played moves is optional in this phase and should be labeled as sparse/weak, not full AV supervision.

### Phase 2: scale LC0 distillation

Scale from 100k to 1M, then 10M+ records only after Phase 1 metrics are clean. Both promoted lanes should reach 100M+ if no blocker appears; scale both beyond 100M only while both remain frontier-relevant. Optimize cache build throughput with Rust/native preprocessing if Python becomes the bottleneck. Track raw record count, usable record count, skipped-policy mass, top-k agreement, WDL calibration, params, FLOPs/MACs, bytes, latency, and train wall time.

### Phase 3: ChessBench / Stockfish action-value lane

Add ChessBench or Stockfish-style action-value data as a complementary teacher, not as an equal policy teacher. Use it for candidate-move value buckets, ranking, regret, and hard-negative learning. Keep LC0 policy/WDL batches separate and preserve source weights in logs.

### Phase 4: on-policy hard-negative loop

Use Tiny Leela self-play and arena blunder diagnostics to mine positions where the student policy is confident but wrong. Relabel selected top-k moves with Stockfish, LC0 deeper search, tablebase where available, or Tiny Leela search. This phase feeds action-value/regret and fixed-suite tactical improvement, not raw one-hot self-play imitation.

### Phase 5: candidate cards and promotion

Every serious LC0-distilled candidate emits [[Design - Candidate frontier cards]] metrics: searchless proxy, v16/v64/v128 protocol Elo, fixed-time strength when available, evals/sec, params, FLOPs/MACs, bytes, blunder rate, value calibration, and backend/export notes. Promotion still requires strength/runtime/size frontier evidence, not loss alone. Browser adaptive export selection and visit benchmarking are required only for final deployable candidates.

## Implementation backlog

1. Add `scripts/lc0_chunk_inspect.py` to inspect chunk format, count records, summarize fields, and sample policy/value stats.
2. Add `scripts/lc0_download_manifest.py` or equivalent manifest generator with checksums and license capture.
3. Add `training/lc0_adapter.py` for record parse -> Tiny Leela normalized example.
4. Add policy-map parity tests covering LC0 1858 indices, UCI, Tiny Leela MoveId, and board normalization.
5. Add a tiny LC0 cache builder that emits trainer-compatible sparse policy rows and WDL/Q metadata.
6. Add an overfit harness for 1k/10k LC0 examples.
7. Add LC0-pretrain configs for the promoted MF80 winner and promoted BT4/SquareFormer winner.
8. Add validation reports to candidate frontier cards.
9. Only after adapter proof, decide local/mac-mini/AWS placement for larger conversion/training.

## Evaluation gates

- **Adapter gate**: policy legality, policy mass, WDL perspective, move-map round-trip, board-normalization parity.
- **Supervised gate**: LC0 policy KL, top-1/top-4/top-8 agreement, WDL CE/calibration, sparse best/played agreement.
- **Search gate**: policy-only, v16, v64, v128; compare fixed time as soon as runtime metrics are available.
- **Blunder gate**: Stockfish node/depth diagnostics on arena games and fixed tactical/failure suites.
- **Deployment gate**: params, FLOPs/MACs, ONNX bytes, browser WASM/WebGPU readiness, evals/sec, fixed-time strength, cold/warm startup, adaptive visit selection for final deploy, and quantization drift when a quantized export is considered.
- **Provenance gate**: all artifacts declare `teacher=lc0_public`, source manifests, license notes, normalization, and model lineage.

## Relationship to Gumbel ZeroShot

Gumbel ZeroShot remains valuable because it tests whether Tiny Leela can learn without LC0 teacher data. It should use its own manifests, model IDs, and eval cards. LC0-distilled models may be used as opponents or deployment baselines, but not as hidden teachers inside the zero lane.
