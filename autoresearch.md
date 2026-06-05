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

1. Add a scaffolded quantized FFN candidate as an explicit encoder kernel variant, initially for FFN weights only.
2. Start with per-output-channel symmetric int8 weights and f32/f16 dequant in WGSL; measure whether lower bandwidth offsets dequant overhead.
3. If FFN int8 is too slow or drifts too much, try mixed precision: int8 dense1 only or dense2 only.
4. Keep quantized path opt-in and run stronger 32-position/9-fixture confirmations only after fast-loop wins pass drift.
5. If quantized matmul is bandwidth-bound but dequant overhead dominates, pivot to f16 packed/generator improvements instead of more int8 variants.

## What's Been Tried

- Productized and documented the recovered opt-in baseline: WASM input + `mixed-tvm-ffn` + JS legal priors + b4/depth1.
- Added drift-harness support for `--input-backend` and `--legal-priors-backend` so this lane can validate the same WASM-input/JS-legal path it benchmarks.
- Added `autoresearch.sh` drift guard metrics and thresholds. Keeps must retain drift guard unless the experiment is explicitly a harness diagnosis.
