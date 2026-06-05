# Autoresearch: LC0 quantized/int8 FFN/encoder lane

## Objective

Start a separate opt-in quantization lane for the LC0 browser hybrid runtime. The target is to reduce full-search fixed-suite latency for `hybrid-wgsl-heads` while preserving policy/value quality.

Recovered-state baseline and control for this lane:

- runtime: `hybrid-wgsl-heads`
- input backend: WASM
- encoder kernel: `mixed-tvm-ffn`
- legal priors: JS
- batch: `lc0BatchSize=4`
- pipeline depth: `batchPipelineDepth=1`
- stable defaults unchanged: ORT ONNX/WebGPU remains the arena/default baseline

Quantized/int8 FFN or encoder variants must remain explicit opt-ins until repeated full-suite speed and drift gates pass.

## Metrics

- **Primary**: `hybrid_b4_ms_per_eval` (ms/eval, lower is better) — derived as `1000 / evals_per_second` from the browser fixed-suite run.
- **Required guard metrics**:
  - `drift_f32_best_move_matches` must equal `drift_fixtures`
  - `drift_native_best_move_matches` must equal `drift_fixtures`
  - `drift_f32_wdl_max_abs_diff <= 0.005` by default
  - `drift_f32_top_prior_max_abs_diff <= 0.01` by default
- **Secondary diagnostics**: `evals_per_second`, `visits_per_position`, `evals_per_position`, `total_eval_ms`, `readback_synced_ms`, `readback_map_wait_ms`, `legal_priors_prep_ms`, `readback_bytes`, `readback_maps`, `dispatch_count`.

## How to Run

```bash
./autoresearch.sh
```

Defaults now run the recovered opt-in baseline plus a small drift guard:

- `LC0_AR_MAX_POSITIONS=16`
- `LC0_AR_MOVETIME_MS=500`
- `LC0_AR_REPS=3`
- `LC0_AR_BATCH_SIZE=4`
- `LC0_AR_PIPELINE_DEPTH=1`
- `LC0_AR_INPUT_BACKEND=wasm`
- `LC0_AR_ENCODER_KERNEL=mixed-tvm-ffn`
- `LC0_AR_LEGAL_PRIORS_BACKEND=js`
- `LC0_AR_DRIFT_GUARD=1`
- `LC0_AR_DRIFT_LIMIT=3`

Useful overrides:

```bash
# Faster screening only; do not keep without stronger confirmation.
LC0_AR_REPS=1 LC0_AR_MAX_POSITIONS=4 LC0_AR_MOVETIME_MS=250 LC0_AR_DRIFT_LIMIT=1 ./autoresearch.sh

# Stronger confirmation for promising quantized variants.
LC0_AR_MAX_POSITIONS=32 LC0_AR_REPS=3 LC0_AR_DRIFT_LIMIT=9 ./autoresearch.sh

# Disable drift only for diagnosing benchmark harness failures, not for keeps.
LC0_AR_DRIFT_GUARD=0 ./autoresearch.sh
```

## Files in Scope

- `src/lc0/wgslMatmulAddProbe.ts` — WGSL encoder/head runtimes, FFN kernels, quantized-kernel experiments, telemetry.
- `src/lc0/generated/tvmPackedF16Wgsl.ts` and `scripts/generate_lc0_tvm_wgsl_kernels.mjs` — generated kernel references if a generated quantized path is added.
- `src/lc0/policyOnlyBrowser.ts`, `src/lc0/searchWorker.ts` — opt-in query/worker wiring only.
- `scripts/lc0_browser_runtime_fixed_suite.mjs` — full-search throughput harness.
- `scripts/lc0_browser_hybrid_drift.mjs` — drift/top-prior/value guard harness.
- `autoresearch.md`, `autoresearch.sh`, `autoresearch.checks.sh`, `autoresearch.ideas.md` — session control and notes.
- `docs/lc0web_custom_inference_checkpoint.md`, `docs/engine_catalog.md`, `knowledge/04_designs/systems/design.inference_optimization.md` — update only when conclusions change.

## Off Limits

- Do not change stable defaults.
- Do not promote WGSL heads, TVM kernels, WASM input, GPU legal priors, quantized kernels, or `batchPipelineDepth>1` to defaults.
- Do not accept a quantized/int8 implementation on speed alone; it must pass drift gates and `autoresearch.checks.sh`.
- Do not compare cleaned/recovered runs against the earlier degraded browser/WebGPU window as runtime speedups.
- Do not use `batchPipelineDepth>1` as parity-preserving promotion evidence.

## Current Baseline

Baseline run after starting this lane with the default fixed-suite workload and drift guard:

- `hybrid_b4_ms_per_eval=6.269691075513055`
- `evals_per_second=159.4974916556266`
- fixed-suite artifact: `/tmp/lc0_autoresearch/20260605_141725_69201_rep3.json`

Drift baseline for the same recovered opt-in path (`hybrid-wgsl-heads`, WASM input, `mixed-tvm-ffn`, JS legal, 3 fixtures):

- f32 best-move matches: `3/3`
- native best-move matches: `3/3`
- f32 WDL max abs diff: `0.0009745955467224121`
- f32 top-prior max abs diff: `0.0029649437439207005`
- native top-prior max abs diff: `0.002750797804308064`
- drift artifact: `/tmp/lc0_autoresearch/20260605_141725_69201_drift.json`

## Hypothesis Queue

1. Highest ROI moved from naive int8 to generated/fused f16 GPU kernels. The current profile says smolgen project and smolgen dense1 dominate the recovered `mixed-tvm-ffn` encoder path, ahead of FFN dense2 and QKV.
2. Keep `mixed-tvm-ffn-smolgen-project` as the current best explicit encoder-kernel opt-in. The first simple tiled smolgen-dense1 (`2048 -> 256`) and combined project+dense1 attempts passed drift but failed to beat project-only, so dense1 needs a better generated/fused schedule before being reconsidered.
3. Look for structural dispatch reductions only where they preserve the optimized matmul shape. Dense2+residual+ln2 fusion is tempting but likely loses if the matmul becomes less tiled.
4. Revisit GPU legal/top-k only as a fused mask/softmax/top-k/readback shrink path, not as a standalone GPU legal-prior toggle.
5. Treat CPU/WASM int8 as a separate future fallback lane; do not spend more WebGPU time on scalar dequant-in-inner-loop int8.

## What's Been Tried

- Productized and documented the recovered opt-in baseline: WASM input + `mixed-tvm-ffn` + JS legal priors + b4/depth1.
- Added drift-harness support for `--input-backend` and `--legal-priors-backend` so this lane can validate the same WASM-input/JS-legal path it benchmarks.
- Added `autoresearch.sh` drift guard metrics and thresholds. Keeps must retain drift guard unless the experiment is explicitly a harness diagnosis.
- Discarded: explicit opt-in per-output-channel int8 FFN WGSL variants. Full FFN int8 passed 3-fixture drift but regressed the default fixed suite to `7.30 ms/eval` versus the recovered `mixed-tvm-ffn` baseline around `6.27 ms/eval` (`/tmp/lc0_autoresearch/20260605_142441_72448_rep3.json`, drift `/tmp/lc0_autoresearch/20260605_142441_72448_drift.json`). Dense2-only int8 also passed drift but regressed to `7.06 ms/eval` (`/tmp/lc0_autoresearch/20260605_142908_74450_rep3.json`). Fast screens for dense1-only/dense2-only were slower as well. The simple WGSL dequant-in-inner-loop approach adds enough ALU/fence time to lose the f16 TVM advantage; do not revisit this shape without a generated/fused int8 kernel or different packing strategy.
- Added smolgen-substage timestamp/profile breakdown. Representative mixed-TVM/WASM-input profile: `/tmp/lc0_profile_mixed_smolgen_breakdown_final.json`, `profiledStageTotalMs=32.96` for 5 profile iterations, with per-layer averages `smolgenProject=0.1717 ms`, `smolgenDense1=0.1127 ms`, `ffnDense2Residual=0.0891 ms`, `qkvProjection=0.0695 ms`, `ffnDense1=0.0485 ms`. This redirects generated/fused f16 work toward smolgen project/dense1 before more FFN-only tweaks.
- Keep candidate: `mixed-tvm-ffn-smolgen-project`, an explicit opt-in that keeps `mixed-tvm-ffn` and switches smolgen project to a tiled f16 kernel that shares each head's 256-element input across a 64-column output tile. Fast screen: `6.18 ms/eval`, drift `1/1` (`/tmp/lc0_autoresearch/20260605_144624_81223_rep1.json`). Default confirmation: `5.37 ms/eval`, drift `3/3` (`/tmp/lc0_autoresearch/20260605_144655_81565_rep3.json`, drift `/tmp/lc0_autoresearch/20260605_144655_81565_drift.json`). Stronger 32-position/9-fixture confirmation: `5.45 ms/eval`, drift `9/9`, f32 WDL max `0.000156`, f32 top-prior max `0.000403` (`/tmp/lc0_autoresearch/20260605_144754_82124_rep3.json`, drift `/tmp/lc0_autoresearch/20260605_144754_82124_drift.json`). Note: timestamp substage profile for this tiled kernel looked slower under pass-boundary profiling, but full-search fixed-suite evidence is the promotion signal.
- Discarded local smolgen micro-shader attempts: a paired-output smolgen project shader and a shared-input tiled smolgen dense1 shader were both slower in timestamp profiles. Do not use timestamp profiles alone to reject a candidate that wins full fixed-suite throughput with drift.
- Discarded simple tiled smolgen-dense1 runtime variants after full-search checks. Isolated `mixed-tvm-ffn-smolgen-dense1` fast screen regressed to `6.90 ms/eval` with drift `1/1` (`/tmp/lc0_autoresearch/20260605_145827_84800_rep1.json`). Combined `mixed-tvm-ffn-smolgen-project-dense1` fast screen was `6.07 ms/eval` (`/tmp/lc0_autoresearch/20260605_145858_85075_rep1.json`), but default confirmation regressed to `5.57 ms/eval` with drift `3/3` (`/tmp/lc0_autoresearch/20260605_145929_85355_rep3.json`) versus a same-session project-only control of `5.46 ms/eval` (`/tmp/lc0_autoresearch/20260605_150026_85866_rep3.json`). Reverted dense1 code; do not keep this shape.
