# Expert handoff: current tiny_leela state and where help would matter

Date: 2026-05-08/09  
Audience: outside ML/chess-engine/browser-inference expert  
Scope: project state, current bets, known risks, and concrete questions where expert review would be useful.

## One-paragraph summary

`tiny_leela` is a browser-first, small Leela-style chess engine project.  The goal is not to port full lc0 infrastructure or train a huge engine; it is to find a Pareto-good small model/search stack balancing chess strength, model bytes, parameters, latency/FLOPs, browser deployability, and search effectiveness.  The current portfolio has three lanes: a reliable residual-CNN baseline, a Tactical-MoveFormer hybrid lane for legal-move/tactical reasoning, and a TinyBT/SquareFormer lane inspired by modern lc0 BT4-style square-token transformers.  The project has working ONNX/TypeScript evaluation, PUCT/search-mode arenas, public teacher overlays, 10M/100M supervised datasets, and active queues for cache building and model evaluation.  The biggest open questions are which architecture should become the next browser champion, how to calibrate action-value/ranking heads into search without hurting play, and how to make evaluation statistically trustworthy without overbuilding infrastructure.

## Read-this packet

If you only have time for a few docs, read these in order:

```text
docs/README.md
docs/model_manifest.md
docs/model_efficiency_metrics.md
docs/release_gate_and_distributed_bench.md
docs/elo-evaluation-process.md
docs/100m-scaling-training-plan.md
docs/head_ablation_roadmap.md
docs/transformer_model_roadmap.md
docs/unified_squareformer_architecture_roadmap.md
docs/small_bt4_progression.md
docs/public_teacher_data_sources_plan.md
```

Generated/current inventory:

```text
artifacts/analysis/model_manifest.current.md
```

## Project objective and constraints

We optimize a multi-objective target:

```text
raw Elo strength
model bytes
parameter count
batch=1 and batch=16 latency
browser deployability: ONNX Runtime Web / WASM / WebGPU friendliness
search effectiveness: policy-only, classic PUCT, AV-PUCT, reranking
```

Important constraints and preferences:

- Keep the PyTorch/ONNX/browser-first path; do **not** switch to official lc0 chunk/C++/JAX infrastructure right now.
- Generated datasets/models/artifacts are not committed under `data/*` or `artifacts/`.
- Use `.venv-onnx/bin/python` for PyTorch/ONNX/numpy tooling.
- Durable queues use logs, status files, PID files, `.done`, and `failed` markers.
- Classic PUCT remains the default; AV/aux-PUCT is opt-in until calibrated.
- Tiny aux/search weights are usually safer initially, e.g. `0.0025–0.005`, not `0.02+`.
- Browser-first means dense, regular, exportable models are preferred over exotic kernels.

## Current lanes

### 1. CNN lane: reliable baseline

Role: incumbent baseline and near-term strongest search candidate.

Current assets include 10M and 100M residual CNNs, plus CNN-AV variants.  The generated manifest currently tracks canonical 100M CNNs from 32x4 through 80x5, guarded 10M CNNs, and AV variants.  A recent manifest marks:

```text
cnn_80x5_100m_e3: current-search-champion tag
cnn64x6_top8_phase2 / cnn80x5_top8_phase3: CNN-AV search candidates
```

Why this lane matters:

- CNNs are simple, robust, browser-exportable, and have known latency/size behavior.
- They provide the floor that TinyBT/MoveFormer must beat.
- They are the safest place to validate action-value heads and calibrated PUCT before adding transformer complexity.

Main questions for an expert:

1. Are the residual CNN sizes/training schedules sensible for the target browser budget?
2. Should we prioritize 100M h2_state continuation, CNN-AV calibration, or quantization first?
3. What minimal eval suite would convince you a new CNN is truly stronger, not just arena-noisy?

### 2. Tactical-MoveFormer lane: legal-move/tactical specialist

Role: original contribution / tactical reasoning bet.

This lane uses legal-move candidate features and sidecars to model move consequences more directly than a pure board-policy head.  It is currently behind the CNN and TinyBT work in execution priority, but code exists for tactical sidecar building and MoveFormer kitchen-sink training.

Current status:

```text
training/build_moveformer_tactical_sidecar_cache.py
training/train_moveformer_kitchensink_torch.py
artifacts/head_ablation_1m/tactical_moveformer64_1m/status.txt
```

The Tactical-MoveFormer64 queue is gated by the 100M h2_state cache and prior arena completion.

Main questions for an expert:

1. Are legal-move sidecars the right way to inject tactical features, or should tactical labels be expressed as AV/rank/regret targets instead?
2. What candidate features are likely to generalize instead of overfitting to tactical heuristics?
3. How should we fairly compare a move-candidate model against board-policy CNNs at equal latency?

### 3. TinyBT / SquareFormer lane: browser inheritor bet

Role: small analogue of modern lc0 BT4 square-token transformers.

Core idea:

```text
64 square tokens
+ chess topology / relation attention bias
+ policy/WDL/q heads
+ fixed-width legal-move action-value export
+ calibrated PUCT or reranking
```

Important lesson: we are **not** trying to shrink BT4 literally.  The target is a tiny, dense, browser-suitable model that borrows square-token structure and chess-aware attention.

Current best 1M SquareFormer result:

```text
artifacts/lc0_lite_squareformer/e1_h2_static_relation
kind: squareformer_v2
input_format: compact_uint8_tokens
layers: 6
d_model: 128
heads: 4
history_plies: 2
relation_bias: true
```

Observed signal from the 1M queue:

```text
E1 h2 static relation beat E0 h2 control on policy CE/top-k/AV MSE.
Quick E3 h7 control underperformed h2 on policy CE/top-k.
```

Recent E2 AV plumbing is complete:

```text
artifacts/lc0_lite_squareformer/e2_av_plumbing/done
artifacts/lc0_lite_squareformer/e2_av_plumbing/report.md
```

That validated:

```text
SquareFormer V2 ONNX export with fixed padded legal_action_ids
SquareFormerEvaluator actionValues maps
policy vs AV-PUCT vs classic PUCT tiny arena plumbing
```

Caveat: the E2 tiny arena is protocol/plumbing evidence, not a strength claim.

Main questions for an expert:

1. Is static relation bias enough for tiny scale, or should smolgen-lite be tried sooner?
2. Is fixed-width legal-action AV export the right near-term interface for browser/search?
3. What loss mix should we use for policy, WDL/q, AV regression, rank/listwise, and regret?
4. How should h7/h8 history be represented so it helps rather than increasing sample complexity?

## Data and teacher stack

Current supervised datasets:

```text
supervised_10m_elite_tcec_v1: 10,000,000 train / 500,000 dev, history_plies=2
supervised_100m_elite_tcec_v1: 100,000,000 train / 1,000,000 dev, history_plies=2
```

Current teacher/action-value overlays:

```text
ChessBench top8:  16,286,511 positions, 123,902,725 candidate rows
ChessBench top48: 16,286,511 positions, 475,972,274 candidate rows
Lichess position eval cache/overlay for broad value supervision
```

Current cache/build notes:

- A 100M 46-plane `h2_state` residual cache build is active/important.
- New 10M h7/h8 SquareFormer queue has been staged and waits for the 100M cache to finish to avoid IO contention.
- Existing `supervised_1m_v1` only has `history_plies=2`; true h7/h8 requires rebuilding from raw game rows.

Questions for an expert:

1. Is the current source mix likely to produce a good chess-playing policy, or too much human-move imitation / opening bias?
2. Should AV/ranking labels come mostly from ChessBench, Stockfish local relabeling, lc0/search-policy chunks, or model-generated hard positions?
3. What is the best minimal schema for h7/h8 history that stays ONNX/browser-friendly?
4. How would you sample curriculum rows across policy/value/AV streams?

## Evaluation stack

Current principles:

- Do not claim one global Elo without protocol context.
- Keep policy-only, classic PUCT, AV-PUCT, fixed-visit, and fixed-time results separate.
- Report model bytes, params, latency, and quantization drift alongside arena scores.
- Use protocol cards for generated arena/eval outputs.

Relevant tools/docs:

```text
docs/release_gate_and_distributed_bench.md
docs/elo-evaluation-process.md
eval/search_mode_arena.mjs
eval/onnx_round_robin_arena.mjs
eval/uci_anchor_arena.mjs
eval/model_efficiency_report.py
eval/lint_eval_protocols.py
```

Current known caveats:

- Some old ChessFormer/SquareFormer ONNX exports have a baked batch reshape bug and are excluded from batched arenas by default.
- Maia/lc0 UCI anchor confidence still needs more hardening around illegal/anomalous behavior.
- OpenBench adoption is blocked on a stable Tiny Leela UCI wrapper.

Questions for an expert:

1. What is the minimum statistically credible promotion protocol for tiny engines under our compute budget?
2. How should we combine offline metrics with small Stockfish/Maia anchor arenas?
3. Should OpenBench be introduced now, or only after the UCI wrapper and local gate mature?
4. What tactical/blunder diagnostics best predict actual search strength?

## Current notable engineering state

Working pieces:

```text
TypeScript chess/movegen/search/evaluator stack
ONNX Runtime evaluator path
SquareFormerEvaluator for squareformer_v2 compact tokens
fixed-width SquareFormer V2 AV ONNX export
PUCT consuming optional actionValues/rank/regret/risk/uncertainty maps
search_mode_arena modes: policy, puct, av, aux
model manifest + efficiency tooling
Stockfish Lite WASM wrapper
```

Important runtime files:

```text
src/nn/evaluator.ts
src/nn/onnxEvaluator.ts
src/nn/squareformerEvaluator.ts
src/search/puct.ts
eval/search_mode_arena.mjs
training/export_squareformer_v2_av_onnx.py
```

Test note:

```bash
node --experimental-strip-types --test tests/onnx_evaluator.test.mjs
```

## What we specifically want help with

### Highest-value expert review

1. **Architecture triage**
   - Should the next serious bet be CNN-AV, TinyBT-AV, TinyBT-smolgen-lite, or Tactical-MoveFormer?
   - What experiment would most cheaply falsify each lane?

2. **Action-value/search calibration**
   - How should AV/rank/regret losses be weighted and calibrated?
   - Should AV be used for root reranking, child prior adjustment, leaf value, or PUCT bonus?
   - What calibration plots/metrics should gate AV-PUCT promotion?

3. **History representation**
   - Does h7/h8 likely help at our tiny scale?
   - If yes, what representation prevents history from becoming noisy extra tokens/features?
   - Should h7/h8 be reserved for TinyBT/smolgen-lite or also applied to CNNs?

4. **Evaluation protocol**
   - How many games/anchors/visits are enough for a promotion decision?
   - Which offline metrics are trustworthy leading indicators?
   - How should we handle browser-latency as a first-class promotion axis?

5. **Browser deployment and quantization**
   - Best path for ONNX Runtime Web / WASM / WebGPU deployment?
   - PTQ vs QAT order for CNNs and SquareFormers?
   - What model shapes are likely to be fast in actual browser kernels, not just theoretically small?

### Concrete questions to answer after reading

- What would you cut from the roadmap for the next 1-2 weeks?
- Which single model family should get the next major GPU allocation?
- What is the first protocol you would trust for a real promotion claim?
- Is the TinyBT-AV fixed-legal-width design sound enough to scale to 10M h7/h8?
- Are we overvaluing action-values relative to simply improving policy/WDL?
- What failure mode would you expect from our current data mix?

## Near-term plan without expert intervention

Current intended order:

```text
1. Finish 100M h2_state cache.
2. Let dependent anchor/tactical queues proceed.
3. Validate 100M cache schema.
4. Continue CNN/AV and Tactical-MoveFormer evaluation queues.
5. Run staged h7/h8 10M SquareFormer queue after cache IO pressure clears.
6. Revisit smolgen-lite after TinyBT-AV plumbing and h7/h8 evidence are usable.
7. Implement stable UCI wrapper for OpenBench-style promotion testing.
8. Use `docs/distributed_selfplay_training_system_design.md` only as future architecture/reference for accepted-model discipline, chunk schemas, WAL/atomic upload, and teacher-reanalysis; do not launch full self-play yet.
9. First AWS cloud scaffold is CPU-only Batch/S3/ECR for distributed SquareFormer cache/reanalysis jobs; see `docs/aws_distributed_cache_setup.md`.
```

## Path quick reference

Docs:

```text
docs/README.md
docs/model_manifest.md
docs/model_efficiency_metrics.md
docs/release_gate_and_distributed_bench.md
docs/elo-evaluation-process.md
docs/100m-scaling-training-plan.md
docs/head_ablation_roadmap.md
docs/transformer_model_roadmap.md
docs/unified_squareformer_architecture_roadmap.md
docs/small_bt4_progression.md
docs/public_teacher_data_sources_plan.md
```

Generated status/results:

```text
artifacts/analysis/model_manifest.current.md
artifacts/lc0_lite_squareformer/e2_av_plumbing/report.md
artifacts/head_ablation_1m/best_tuned_newest_vs_100m_e3_arena/status.txt
artifacts/cache_build_100m_h2_state/status.txt
artifacts/lc0_lite_squareformer/h7_h8_10m/status.txt
```

Key datasets/caches:

```text
data/datasets/supervised_10m_elite_tcec_v1/manifest.json
data/datasets/supervised_100m_elite_tcec_v1/manifest.json
data/public_teacher_overlays/chessbench_full_policy_value_direct_top8_32shards_v1/collection_manifest.json
data/public_teacher_overlays/chessbench_full_policy_value_direct_top48_32shards_v1/collection_manifest.json
```
