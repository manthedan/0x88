# Tiny Leela Research Phases

This project deliberately separates product/lab construction from dovetail research.

## Phase A — Build the lab substrate

Purpose: make it possible to run chess-engine experiments safely.

Status: **complete enough for Phase B**.

Shared substrate currently includes:

- TypeScript/Node test scaffold.
- Board representation and FEN parsing.
- Stable UCI/action-id move codec.
- Baseline pseudo-legal move generation and move application.
- Feature encoder skeleton.
- Backend-neutral `Evaluator` interface.
- Search entry point that consumes only `BoardState` plus `Evaluator`.
- Move-encoding and browser-runtime documentation.

Phase A work is not research. It is foundational product/lab work.

## Phase B — Freeze evaluation metrics

Purpose: define fixed, executable metrics before comparing research ideas.

Status: **bootstrap frozen**.

The fixed benchmark spec lives in `eval/benchmark_spec.json`. The validation script is `eval/phase_b_metrics.mjs`.

Phase B metrics are grouped by domain:

1. **Policy quality**
   - `policy_top1_acc` ↑
   - `policy_top3_acc` ↑
   - `policy_cross_entropy` ↓
2. **Value/WDL quality**
   - `wdl_cross_entropy` ↓
   - `q_mse` ↓
3. **Search strength proxy**
   - `fixed_playout_suite_score` ↑
   - `mate_or_tactic_solve_rate` ↑
4. **Runtime/product constraints**
   - `median_move_latency_ms` ↓
   - `model_size_mb` ↓
   - `browser_compatibility_score` ↑

The current `tiny_leela_score` / lab-readiness score is only a Phase A/B readiness signal. It must not be used as a Phase C research-performance metric.

## Phase C — Dovetail research

Start Phase C only after a lane proposes a concrete alternative and the relevant Phase B metric is executable and fixed.

Examples:

- Architecture lane compares model families against policy/WDL metrics and size.
- Training lane compares distillation targets against policy/WDL metrics.
- Search lane compares PUCT variants against fixed-playout suite score and latency.
- Runtime lane compares inference backends against latency, size, and compatibility.

Changing the benchmark dataset, metric definitions, or workload requires a new benchmark id.
