# Autoresearch: LC0 WGSL-head readback optimization at batch 4

## Objective

Optimize the opt-in LC0 browser hybrid runtime path `hybrid-wgsl-heads` for parity-preserving fixed-suite search at:

- `lc0BatchSize=4`
- `batchPipelineDepth=1`
- JS input backend
- JS legal-prior backend unless explicitly experimenting with opt-in alternatives

The loop should reduce or hide WebGPU readback/fence cost without changing search semantics, stable defaults, or correctness baselines. The primary workload is small enough for repeated autonomous experiments, but conclusions about promotion must come from stronger alternating fixed-suite/arena evidence outside the fast loop.

## Metrics

- **Primary**: `hybrid_b4_ms_per_eval` (ms/eval, lower is better) — derived as `1000 / evals_per_second` from the fixed-suite browser run.
- **Secondary**:
  - `evals_per_second` (higher is better)
  - `visits_per_position` and `evals_per_position`
  - `total_eval_ms`
  - `readback_synced_ms`
  - `readback_map_async_ms`
  - `readback_map_wait_ms`
  - `readback_overlap_hidden_ms`
  - `legal_priors_prep_ms`
  - `readback_bytes`
  - `readback_maps`
  - `dispatch_count`

Secondary metrics are diagnostic only. Keep/discard decisions should be based on the primary metric plus correctness/check outcomes, with secondary metrics used to choose the next hypothesis.

## How to Run

```bash
./autoresearch.sh
```

The script emits `METRIC name=value` lines for pi-autoresearch. Defaults:

- FEN corpus: `artifacts/lc0_runtime_arena_20260605/batch_matrix/fixed_suite_32_fens.txt`
- `LC0_AR_MAX_POSITIONS=4`
- `LC0_AR_MOVETIME_MS=250`
- `LC0_AR_REPS=1`
- `LC0_AR_BATCH_SIZE=4`
- `LC0_AR_PIPELINE_DEPTH=1`

Useful manual overrides:

```bash
LC0_AR_REPS=2 ./autoresearch.sh
LC0_AR_MAX_POSITIONS=16 LC0_AR_MOVETIME_MS=500 ./autoresearch.sh
```

For stronger evidence after promising keeps, run alternating 16/32-position fixed-suite checks comparing ONNX b1 against hybrid WGSL b4. Do not use the fast 4-position loop as promotion evidence.

## Files in Scope

- `src/lc0/wgslMatmulAddProbe.ts` — custom WGSL encoder/head runtimes, readback handling, legal-prior paths, telemetry.
- `src/search/puct.ts` — search telemetry only; avoid semantic changes unless explicitly marked speculative and isolated.
- `scripts/lc0_browser_runtime_fixed_suite.mjs` — benchmark/report extraction if more metrics are needed.
- `scripts/lc0_browser_runtime_arena_bench.mjs` — stronger follow-up arena/fixed-suite evidence.
- `lc0-arena.html` and `src/lc0/arenaBrowser.ts` — UI/query telemetry only, if needed.
- `knowledge/04_designs/systems/design.inference_optimization.md` and `docs/lc0web_custom_inference_checkpoint.md` — update when conclusions change.
- `autoresearch.md`, `autoresearch.sh`, `autoresearch.checks.sh`, `autoresearch.ideas.md` — session control and notes.

## Off Limits

- Do not change stable defaults: ORT ONNX/WebGPU remains the arena/default baseline.
- Do not promote WGSL heads, TVM/generated kernels, shader-f16, WASM input/legal priors, GPU legal priors, or `batchPipelineDepth>1` to defaults.
- Do not use `batchPipelineDepth>1` as parity-preserving evidence; it is speculative search semantics.
- Do not change native LC0 BLAS/Eigen or f32 ONNX correctness baselines.
- Do not make broad rewrites, add dependencies, or alter chess/search semantics for a benchmark-only gain.
- Do not overclaim from one noisy Chromium/WebGPU run or microbench-only result.

## Constraints

- Every kept implementation must pass `autoresearch.checks.sh`.
- Nontrivial kept changes should also get targeted browser smoke/lifecycle validation and auto-review before finalization.
- Preserve WebGPU buffer lifecycle safety: mapped buffers must be unmapped, pending maps must be cleaned up on error paths, and deferred readback slots must not be reused while in-flight.
- Keep all experimental lanes opt-in via existing query/CLI parameters.
- Record negative evidence clearly so future iterations do not repeat failed ideas.

## Current Baseline Context

Recent branch history before this autoresearch session:

- `a40e9f0 Add LC0 arena batch timing controls`
- `ac5e53d Overlap WGSL legal-prior prep with readback`
- `25d605d Start WGSL readback mapping before CPU prep`

Known empirical state:

- Hybrid WGSL b4 is useful and roughly tied with ONNX b1 at 500ms on prior 32-position checks.
- Hybrid WGSL b4 lost to ONNX b1 on a 1000ms check.
- Current dominant cost remains WebGPU fence/readback time, not JS legal-prior candidate prep.
- Eager mapAsync-before-CPU-prep works and has clean auto-review, but only hides roughly `~0.01–0.03ms`/eval of CPU prep.

## Hypothesis Queue

Prefer small, isolated experiments in this order:

1. Reduce readback shape/bytes for WGSL heads while preserving legal-prior output quality.
2. Improve or harden opt-in GPU legal-prior/top-k readback reduction.
3. Separate WDL and policy readback timing/shape to identify unavoidable fence cost vs byte-copy cost.
4. Add better timestamp/query attribution only if it guides E2E search changes.
5. Explore WASM legal-prior prep only if telemetry shows JS legal-prior postprocess becomes meaningful.
6. Consider generated/TVM/f16 kernel tweaks only after E2E/timestamp evidence shows kernel time, not readback, dominates.

## What's Been Tried

- Physical batching at `batchPipelineDepth=1` reduces map pressure and makes b4 the current custom-WGSL target.
- Batch 8 is not clearly better and regressed on a prior 32-position corpus.
- Starting mapAsync before JS legal-prior prep improves telemetry and is lifecycle-hardened, but CPU prep is too small to materially move E2E throughput.
- Discarded: omitting `mappedPolicy` result arrays in the evaluator search path. It did not improve the fast-loop primary metric and readback wait still dominated.
- Discarded: replacing batched readback `Float32Array.slice` copies with `subarray` views. The fast-loop metric regressed; do not revisit without a focused CPU-copy microbench.
- Discarded: opt-in GPU legal priors reduced readback bytes (`7444` -> `3084`) but worsened E2E timing and added a dispatch (`159` -> `160`). Byte reduction alone is not enough if extra GPU work/fence time grows.
- Noise note: an unchanged rerun regressed by ~2.4%, so tiny single-run improvements need confirmation before keeping.

Update this section after every few experiments, especially for discarded ideas and benchmark-noise observations.
