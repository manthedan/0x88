# Autoresearch: LC0 WGSL-head WASM-input/scheduler lane at batch 4

## Objective

Optimize the opt-in LC0 browser hybrid runtime path `hybrid-wgsl-heads` for full-search fixed-suite throughput at:

- `lc0BatchSize=4`
- `batchPipelineDepth=1` unless explicitly running scheduler exploration
- WASM input backend
- JS legal-prior backend unless explicitly experimenting with opt-in alternatives

This is a separate input-backend/scheduler lane from the previous JS-input readback loop. Keep WASM input opt-in and do not change stable defaults. The primary workload is still parity-preserving at `batchPipelineDepth=1`; any `batchPipelineDepth>1` run is speculative speed/quality exploration only, not promotion evidence.

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
- `LC0_AR_MAX_POSITIONS=16`
- `LC0_AR_MOVETIME_MS=500`
- `LC0_AR_REPS=3`
- `LC0_AR_BATCH_SIZE=4`
- `LC0_AR_PIPELINE_DEPTH=1`
- `LC0_AR_INPUT_BACKEND=wasm`

Useful manual overrides:

```bash
# Fast screening only; do not keep based on this without confirmation.
LC0_AR_REPS=1 LC0_AR_MAX_POSITIONS=4 LC0_AR_MOVETIME_MS=250 ./autoresearch.sh

# Extra confirmation for promising structural changes.
LC0_AR_REPS=5 ./autoresearch.sh

# Compare against the previous JS-input lane without changing this lane's baseline.
LC0_AR_INPUT_BACKEND=js ./autoresearch.sh
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
- The previous JS-input readback loop exhausted most readback micro-optimizations; WebGPU fence/readback remains a major cost.
- Deferred-readback fixture attribution showed ~1.3–1.4x speedup with matching best moves, but fixed-suite `batchPipelineDepth=2` timed out and needs a dedicated harness.
- A non-comparable WASM-input screen showed a strong full-search signal (`~9.75 ms/eval` vs JS-input stronger-loop baseline `~11.65 ms/eval`), motivating this separate lane.

## Hypothesis Queue

The default loop uses a stronger 16-position, 500ms, 3-rep median workload before keep/discard decisions. This lane should first confirm and then optimize the opt-in WASM-input path:

1. Establish a clean WASM-input b4/depth1 baseline and compare nearby JS-input controls only as diagnostics.
2. Sweep batch size `2/4/8` at `batchPipelineDepth=1` with WASM input.
3. Add a dedicated scheduler/quality harness for deferred readback before retrying `batchPipelineDepth>1` fixed-suite runs.
4. If scheduler/fence overlap remains promising, explore `batchPipelineDepth=2/4` only as speculative speed/quality evidence.
5. If input/backend time is no longer material, return to attribution modes or start a separate TVM/mixed-kernel lane.
6. Avoid readback micro-optimizations unless new attribution changes the bottleneck picture.

## What's Been Tried

- Physical batching at `batchPipelineDepth=1` reduces map pressure and makes b4 the current custom-WGSL target.
- Batch 8 is not clearly better and regressed on a prior 32-position corpus.
- Starting mapAsync before JS legal-prior prep improves telemetry and is lifecycle-hardened, but CPU prep is too small to materially move E2E throughput.
- Discarded: omitting `mappedPolicy` result arrays in the evaluator search path. It did not improve the fast-loop primary metric and readback wait still dominated.
- Discarded: replacing batched readback `Float32Array.slice` copies with `subarray` views. The fast-loop metric regressed; do not revisit without a focused CPU-copy microbench.
- Discarded: opt-in GPU legal priors reduced readback bytes (`7444` -> `3084`) but worsened E2E timing and added a dispatch (`159` -> `160`). Byte reduction alone is not enough if extra GPU work/fence time grows.
- Discarded: restoring full-buffer copy/early unmap after the suspect no-copy keep regressed in a two-rep check (`13.97` ms/eval). Do not toggle whole-copy vs no-copy again without a more controlled A/B.
- Discarded: pre-copying per-slot slices and unmapping before legal-prior postprocess also regressed in a two-rep check (`14.03` ms/eval). Mapped lifetime/copy placement is not the next lever.
- Discarded: compact JS legal-prior readback using one `copyBufferToBuffer` per legal move reduced reported readback bytes dramatically (`7444` -> `~143`) and kept dispatch count flat, but worsened E2E (`12.85` ms/eval). Avoid per-move copy-command gather.
- Low-confidence keep: combining nonzero/variation sanity scans had a strong single-run result (`9.86` ms/eval) but failed one-rep and two-rep confirmations (`13.04`, `13.60`). Treat as suspect until stronger alternating evidence; do not build on CPU scan micro-optimizations alone.
- Noise note: an unchanged rerun regressed by ~2.4%, and several apparent single-run wins failed one-rep/two-rep confirmations. The loop now defaults to 16 positions, 500ms, 3 reps to reduce false keeps.
- Stop exploring low-ROI readback micro-toggles without new attribution evidence: Array allocation cleanup, slice/subarray, map range, unmap timing, 256-byte stride padding, tiny dispatch fusions, and per-legal-move compact copies.
- Discarded as non-comparable to the JS-input loop but promising for this lane: opt-in WASM input reported `~9.75 ms/eval` on the stronger workload. Treat this lane's first run as the new baseline rather than mixing it with JS-input results.

Update this section after every few experiments, especially for discarded ideas and benchmark-noise observations.
