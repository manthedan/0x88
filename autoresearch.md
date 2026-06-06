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
2. Keep `mixed-tvm-ffn-smolgen-project` as the current best explicit encoder-kernel opt-in, now with the 64-column smolgen-project tile plus an 8-lane parallel-reduction smolgen-dense1 kernel whose inner row loop is safely unrolled by two. Earlier simple one-output-per-lane dense1, dense1 fusion, and row-split attempts remain discarded.
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
- Stabilized previous winner: `mixed-tvm-ffn-smolgen-project` 32-position confirmation produced `5.416 ms/eval` with drift `9/9` (`/tmp/lc0_autoresearch/20260605_150535_87471_rep3.json`, drift `/tmp/lc0_autoresearch/20260605_150535_87471_drift.json`).
- Discarded smolgen-project tile sweep alternatives. Temporary 128-column project tile fast screen was `6.21 ms/eval` with drift `1/1` (`/tmp/lc0_autoresearch/20260605_150710_88135_rep1.json`). Temporary 32-column tile fast screen was `6.06 ms/eval` with drift `1/1` (`/tmp/lc0_autoresearch/20260605_150752_88501_rep1.json`), but default confirmation was `5.459 ms/eval` (`/tmp/lc0_autoresearch/20260605_150824_88854_rep3.json`), not better than the 64-column project winner/control. Reverted tile-size changes; keep the 64-column tile.
- Discarded first smolgen-project+attention-score fusion attempt. It skipped the separate smolgen project dispatch and computed the project dot inside the attention score shader, but the WGSL-head drift route failed with `WGSL hybrid heads produced zero or uniform mapped policy` (debug route and `/tmp/lc0_autoresearch/20260605_151418_90680_rep1.log`). Reverted the fusion code. Any future fusion attempt needs isolated score/bias parity before full fixed-suite timing.
- Current-best follow-up profile artifact: `/tmp/lc0_profile_current_best_next.json`. In this timestamp profile, `mixed-tvm-ffn-smolgen-project` reduced smolgen project from `0.2136` to `0.0944 ms/layer` versus `mixed-tvm-ffn`, with remaining prominent stages `smolgenDense1=0.173`, `ffnDense2Residual=0.127`, `qkvProjection=0.093`, and `outputProjection=0.063 ms/layer`.
- Discarded adjacent toggles against current best. `mixed-tvm-ffn-smolgen-project-outproj` fast screen regressed to `7.11 ms/eval` with drift `1/1` (`/tmp/lc0_autoresearch/20260605_152850_94310_rep1.json`); a temporary QKV-TVMed `mixed-tvm-ffn-smolgen-project-qkv` regressed to `8.05 ms/eval` with drift `1/1` (`/tmp/lc0_autoresearch/20260605_153237_96055_rep1.json`). Legal-prior toggles were slower: GPU legal `6.37 ms/eval` (`/tmp/lc0_autoresearch/20260605_152929_94618_rep1.json`) and WASM legal `6.38 ms/eval` (`/tmp/lc0_autoresearch/20260605_153000_94898_rep1.json`). Input-backend toggles were slower: JS input `8.78 ms/eval` (`/tmp/lc0_autoresearch/20260605_153118_95343_rep1.json`) and WGSL input `6.48 ms/eval` (`/tmp/lc0_autoresearch/20260605_153149_95623_rep1.json`). Reverted temporary outproj/QKV code; keep WASM input + JS legal.
- Discarded temporary smolgen dense1+swish+LN1 fusion on the project winner. It reduced dispatch count from 160 to 150 and passed drift, but did not improve fixed-suite speed: fast screen `6.08 ms/eval` (`/tmp/lc0_autoresearch/20260605_153612_97194_rep1.json`) and default confirmation `5.47 ms/eval` with drift `3/3` (`/tmp/lc0_autoresearch/20260605_153645_97583_rep3.json`) versus the current-best 32-position confirmation at `5.416 ms/eval`; drift also approached the top-prior threshold (`0.00896 < 0.01`). Reverted the fusion code; dispatch-count reduction alone is not enough if the dense matmul schedule loses parallelism.
- Discarded split-FFN TVM mixes layered over the smolgen-project winner. Temporary variants with only TVM dense1 (`mixed-tvm-ffn-dense1-smolgen-project`) and only TVM dense2 (`mixed-tvm-ffn-dense2-smolgen-project`) both passed 1-fixture drift but regressed fast screens: dense1-only `7.68 ms/eval` (`/tmp/lc0_autoresearch/20260605_154412_715_rep1.json`) and dense2-only `7.84 ms/eval` (`/tmp/lc0_autoresearch/20260605_154441_1038_rep1.json`) versus a same-session current-best control at `6.13 ms/eval` (`/tmp/lc0_autoresearch/20260605_154347_282_rep1.json`). Reverted the temporary split-FFN code; keep the paired TVM FFN path.
- Discarded temporary row-split smolgen dense1 over the smolgen-project winner. The variant split the 2048-row dense1 dot into eight 256-row reductions plus a finish pass to increase parallelism, and it passed 1-fixture drift, but the extra workgroups/dispatch work regressed the fast screen to `10.07 ms/eval` (`/tmp/lc0_autoresearch/20260605_155013_2963_rep1.json`). Reverted the row-split code; dense1 needs a generated/packed schedule that reuses tiles across columns rather than one reduction workgroup per output/chunk.
- Discarded 256-column smolgen-project tile over the project winner. It passed 1-fixture drift but badly regressed the fast screen to `29.41 ms/eval` (`/tmp/lc0_autoresearch/20260605_155224_3785_rep1.json`), likely from max-size workgroups/long per-lane loops. Reverted the tile change; 64-column tile remains the kept project kernel.
- Discarded 16-column smolgen-project tile over the project winner. It passed 1-fixture drift but lost a same-session fast screen: 16-column `6.34 ms/eval` (`/tmp/lc0_autoresearch/20260605_160132_6649_rep1.json`) versus restored 64-column control `5.94 ms/eval` (`/tmp/lc0_autoresearch/20260605_160205_6933_rep1.json`). Reverted the tile change; 64-column tile is now screened against 16/32/128/256.
- Discarded manually unrolled 64-column smolgen-project inner loop. It passed 1-fixture drift but fast-screened at `6.22 ms/eval` (`/tmp/lc0_autoresearch/20260605_160348_7547_rep1.json`), slower than the restored 64-column loop control above. Reverted the unroll; keep the compact loop unless a generator produces a better schedule.
- Discarded shared-input smolgen dense2. It passed drift and had a slightly better 4-position fast screen (`6.17 ms/eval`, `/tmp/lc0_autoresearch/20260605_160716_8712_rep1.json`) than a noisy same-session control (`6.34 ms/eval`, `/tmp/lc0_autoresearch/20260605_160746_9094_rep1.json`), but the default confirmation lost to restored project-only: shared dense2 `5.44 ms/eval` (`/tmp/lc0_autoresearch/20260605_160845_9483_rep3.json`) versus control `5.40 ms/eval` (`/tmp/lc0_autoresearch/20260605_160943_9997_rep3.json`). Reverted the dense2 code.
- Discarded shared-input smolgen compress. It passed drift and fast-screened slightly ahead (`5.88 ms/eval`, `/tmp/lc0_autoresearch/20260605_161112_10621_rep1.json`) of same-session project-only (`6.00 ms/eval`, `/tmp/lc0_autoresearch/20260605_161143_10963_rep1.json`), but default confirmation regressed to `5.53 ms/eval` (`/tmp/lc0_autoresearch/20260605_161237_11406_rep3.json`), worse than current-best controls. Reverted the compress code.
- Discarded smolgen LN2 swish-cache-in-workgroup. It avoided recomputing swish for the final write and passed drift; fast screen was slightly ahead of its immediate control (`5.88 ms/eval`, `/tmp/lc0_autoresearch/20260605_161602_12537_rep1.json`, versus `6.03 ms/eval`, `/tmp/lc0_autoresearch/20260605_161633_12805_rep1.json`), but default confirmation regressed to `5.66 ms/eval` (`/tmp/lc0_autoresearch/20260605_161716_13115_rep3.json`). Reverted the LN2 code.
- Discarded smolgen-project final-barrier elision. It passed drift but fast-screened slower at `6.08 ms/eval` (`/tmp/lc0_autoresearch/20260605_161842_13742_rep1.json`); the conditional branch likely costs more than one avoided final barrier. Reverted the barrier change.
- Discarded second smolgen-project + attention-score fusion attempt. This version was parity-clean (1-fixture drift passed) and fused the 64-column project dot into a per-row score workgroup while skipping the separate project dispatch, but it regressed the fast screen to `11.39 ms/eval` (`/tmp/lc0_autoresearch/20260605_155742_5365_rep1.json`). Reverted the fusion code. The score-side fusion repeats too much project work per score row and loses the efficient project schedule; future fusion needs a different tile shape that shares smolgen input across score rows or starts from a generated schedule.
- Keep improvement: replaced the serial smolgen-dense1 dot with an 8-lane parallel reduction over 16 output columns per workgroup. A 16-lane reduction was faster (`5.05 ms/eval`) but failed the 9-fixture top-prior drift gate (`0.01108 > 0.01`), so it was not kept. The initial 8-lane version passed the 32-position/9-fixture confirmation at `5.375 ms/eval` (`/tmp/lc0_autoresearch/20260605_162519_16363_rep3.json`) with drift `9/9`, f32 WDL max `0.0001565`, and f32 top-prior max `0.000403`; a same-session restored project-only control was `5.642 ms/eval` (`/tmp/lc0_autoresearch/20260605_162645_16975_rep3.json`). The baseline `mixed-tvm-ffn` opt-in also passed a 16-position/3-fixture check at `5.89 ms/eval` (`/tmp/lc0_autoresearch/20260605_162820_17605_rep3.json`). Follow-up profile artifact: `/tmp/lc0_profile_dense1_keep.json`.
- Keep improvement over the 8-lane dense1 kernel: unrolled the dense1 row loop by two while preserving accumulation order. Re-run confirmation passed the 32-position/9-fixture gate at `4.863 ms/eval` (`/tmp/lc0_autoresearch/20260605_164835_24873_rep3.json`) with drift `9/9`, f32 WDL max `0.0001565`, and f32 top-prior max `0.000482`; a same-session restored 8-lane/no-unroll control was `5.141 ms/eval` (`/tmp/lc0_autoresearch/20260605_164958_25479_rep3.json`).
- Discarded dense1 shape/micro follow-ups over the kept 8-lane kernel. 32-column/8-reduction (`5.88 ms/eval`, `/tmp/lc0_autoresearch/20260605_163214_18822_rep1.json`), 8-column/8-reduction (`6.05 ms/eval`, `/tmp/lc0_autoresearch/20260605_163321_19234_rep1.json`), and 16-column/4-reduction (`6.54 ms/eval`, `/tmp/lc0_autoresearch/20260605_163454_19744_rep1.json`) all lost fast screens. Manual reduction unrolling (`5.98 ms/eval`, `/tmp/lc0_autoresearch/20260605_163826_21316_rep1.json`), paired-output dense1 (`5.83 ms/eval`, `/tmp/lc0_autoresearch/20260605_163949_21998_rep1.json`), and 4-step dense1 row-loop unroll (`5.53 ms/eval`, `/tmp/lc0_autoresearch/20260605_165140_26196_rep1.json`) also lost fast screens. Reverted all of these.
- Discarded smolgen dense2 follow-ups. Paired-output dense2 passed 1-fixture drift but fast-screened slower at `5.96 ms/eval` (`/tmp/lc0_autoresearch/20260605_164102_22529_rep1.json`), and dense2 row-loop unroll passed 1-fixture drift but fast-screened slower at `5.83 ms/eval` (`/tmp/lc0_autoresearch/20260605_165229_26715_rep1.json`). Reverted the dense2 code.
- Discarded post-unroll scalar micro cleanups. Smolgen-compress row-loop unroll fast-screened slower at `5.82 ms/eval` (`/tmp/lc0_autoresearch/20260605_165622_28006_rep1.json`). Dense1 `fma(...)` accumulation failed the 9-fixture top-prior drift guard (`0.0239 > 0.01`) despite a `5.11 ms/eval` run (`/tmp/lc0_autoresearch/20260605_165752_28667_rep3.json`). Dense1 weight-index hoisting passed 9-fixture drift but did not beat the kept shape (`4.90 ms/eval`, `/tmp/lc0_autoresearch/20260605_165959_29595_rep3.json`). Manual reduction unroll plus the kept row-loop unroll still lost its fast screen (`5.73 ms/eval`, `/tmp/lc0_autoresearch/20260605_170151_30293_rep1.json`). Reverted all of these.
- Discarded WGSL-head loop unrolls. Generic vector-dense row-loop unroll produced one fast/pass confirmation (`4.69 ms/eval`, `/tmp/lc0_autoresearch/20260605_170323_30961_rep3.json`) but failed 9-fixture top-prior drift on a repeated confirmation (`0.0152 > 0.01`, `/tmp/lc0_autoresearch/20260605_170746_33025_rep3.json`), so it was not kept. Token-matrix dense unroll (`6.06 ms/eval`, `/tmp/lc0_autoresearch/20260605_170929_33710_rep1.json`), policy-logits unroll (`5.65 ms/eval`, `/tmp/lc0_autoresearch/20260605_171018_34058_rep1.json`), and mapped-promotion unroll (`5.54 ms/eval`, `/tmp/lc0_autoresearch/20260605_171106_34420_rep1.json`) also lost fast screens. Reverted all of these.
- Discarded additional current-best follow-ups. GPU legal priors (`5.89 ms/eval`, `/tmp/lc0_autoresearch/20260605_173808_42314_rep1.json`) and WGSL input (`5.79 ms/eval`, `/tmp/lc0_autoresearch/20260605_173841_42596_rep1.json`) still lost quick screens. WGSL head dense workgroup-size changes did not improve: token-matrix dense 32-wide passed drift but was effectively tied/slower (`4.867 ms/eval`, `/tmp/lc0_autoresearch/20260605_173150_39390_rep3.json`), token-matrix dense 8-wide lost quick (`5.87 ms/eval`, `/tmp/lc0_autoresearch/20260605_173330_40234_rep1.json`), value-vector 32-wide lost full (`4.94 ms/eval`, `/tmp/lc0_autoresearch/20260605_173451_40925_rep3.json`), and value-vector 128-wide lost quick (`5.66 ms/eval`, `/tmp/lc0_autoresearch/20260605_173624_41602_rep1.json`). Dense1 weight-index specialization (`5.83 ms/eval`, `/tmp/lc0_autoresearch/20260605_173728_42006_rep1.json`) and split-accumulator dense1 (`5.07 ms/eval`, `/tmp/lc0_autoresearch/20260605_173954_43195_rep3.json`) lost. Fusing value dense2+softmax reduced dispatch count to 159 but lost the full run (`4.92 ms/eval`, `/tmp/lc0_autoresearch/20260605_174313_44359_rep3.json`); mapped-policy 128-wide workgroups also lost quick (`5.64 ms/eval`, `/tmp/lc0_autoresearch/20260605_174516_45110_rep1.json`). Reverted all of these.
- Discarded another follow-up batch on the kept winner. Smolgen-compress workgroup shapes 16x4 (`5.84 ms/eval`, `/tmp/lc0_autoresearch/20260605_232212_10789_rep1.json`), 4x16 (`5.50 ms/eval`, `/tmp/lc0_autoresearch/20260605_232249_11108_rep1.json`), and 32x2 (`5.62 ms/eval`, `/tmp/lc0_autoresearch/20260605_232326_11418_rep1.json`) lost quick screens. WGSL-head packed-f16 weight buffers (`5.95 ms/eval`, `/tmp/lc0_autoresearch/20260605_232602_12305_rep1.json`) and separate linear head shaders (`5.69 ms/eval`, `/tmp/lc0_autoresearch/20260605_233515_16221_rep1.json`) were slower. TVM QKV and/or out-proj over the smolgen-project winner still lost (`5.88–6.05 ms/eval`, `/tmp/lc0_autoresearch/20260605_232708_12823_rep1.json`, `/tmp/lc0_autoresearch/20260605_232747_13193_rep1.json`, `/tmp/lc0_autoresearch/20260605_232829_13540_rep1.json`). Smolgen dense2 workgroup-size changes lost: 128-wide full confirmation `5.10 ms/eval` (`/tmp/lc0_autoresearch/20260605_233047_14371_rep3.json`) and 32-wide quick `5.64 ms/eval` (`/tmp/lc0_autoresearch/20260605_233219_15007_rep1.json`). WASM legal priors still lost quick (`5.75 ms/eval`, `/tmp/lc0_autoresearch/20260605_233254_15295_rep1.json`). Reverted all code changes; a restored control passed 32-position/9-fixture drift at `4.94 ms/eval` (`/tmp/lc0_autoresearch/20260605_233552_16577_rep3.json`).
- Discarded post-closeout shader micro-screens. A 2-lane parallel-reduction smolgen dense2 shape regressed the fast screen to `5.98 ms/eval` (`/tmp/lc0_autoresearch/20260605_235606_21047_rep1.json`). Hand QKV larger K tiles were tied/slower: 32-wide tile `5.66 ms/eval` (`/tmp/lc0_autoresearch/20260605_235739_21786_rep1.json`) versus restored control `5.65 ms/eval` (`/tmp/lc0_autoresearch/20260605_235812_22152_rep1.json`), and 64-wide tile `5.69 ms/eval` (`/tmp/lc0_autoresearch/20260606_000019_22983_rep1.json`). Replacing generated TVM `pick_lane` branches with `select` regressed to `5.75 ms/eval` (`/tmp/lc0_autoresearch/20260605_235920_22587_rep1.json`). Hand attention out-projection 32-wide K tile regressed to `5.67 ms/eval` (`/tmp/lc0_autoresearch/20260606_000114_23351_rep1.json`). Smolgen-project weight-index strength reduction and attention-score 2-step channel unroll both looked plausible in one-fixture screens but lost default confirmations: project strength reduction `5.11 ms/eval` (`/tmp/lc0_autoresearch/20260606_000234_23977_rep3.json`) and score unroll `5.108 ms/eval` (`/tmp/lc0_autoresearch/20260606_000558_25488_rep3.json`) versus restored control `4.985 ms/eval` (`/tmp/lc0_autoresearch/20260606_000332_24487_rep3.json`). Reverted all code changes.
- Discarded another continuation batch. A 3-step row-loop unroll in the kept 8-lane smolgen dense1 shape fast-screened at `5.32 ms/eval` (`/tmp/lc0_autoresearch/20260606_001444_28073_rep1.json`) but failed a 3-fixture drift check (`f32/native best-move matches 2/3`, f32 WDL max about `0.0302`), so it was not kept. A parallel score+softmax fusion that reduced dispatch count from 160 to 150 failed the one-fixture drift guard with zero best-move matches and invalid prior diffs (`/tmp/lc0_autoresearch/20260606_092459_30242_rep1.json`). Encoder attention-value follow-ups were not promotion material: 2-step value loop unroll passed 9-fixture drift but only reached `5.055 ms/eval` on the 32-position suite (`/tmp/lc0_autoresearch/20260606_092839_32247_rep3.json`) versus a restored same-session control at `5.115 ms/eval` (`/tmp/lc0_autoresearch/20260606_093002_32976_rep3.json`) and did not beat the current-best `4.863 ms/eval`; 4-step value-loop unroll quick-screened at `5.58 ms/eval` (`/tmp/lc0_autoresearch/20260606_093141_33646_rep1.json`), and a 16x4 attention-value workgroup quick-screened at `5.65 ms/eval` (`/tmp/lc0_autoresearch/20260606_093241_34035_rep1.json`). Reverted all code changes.
- Discarded generated/packed smolgen transposed-weight screens. A col-major f16 smolgen-dense1 variant passed quick drift but only reached `5.454 ms/eval` fast (`/tmp/lc0_autoresearch/20260606_095704_37366_rep1.json`) and `4.928 ms/eval` on the 16-position suite (`/tmp/lc0_autoresearch/20260606_095735_37644_rep3.json`) versus same-session restored control `4.907 ms/eval` (`/tmp/lc0_autoresearch/20260606_095835_38172_rep3.json`). Extending the transposed layout to dense2 also passed one-fixture drift but quick-screened at `5.472 ms/eval` (`/tmp/lc0_autoresearch/20260606_100119_39324_rep1.json`). Reverted all code changes.
- Discarded remaining shared/packed smolgen screens. A transposed-weight smolgen-compress variant passed quick drift but regressed to `5.622 ms/eval` (`/tmp/lc0_autoresearch/20260606_100655_41117_rep1.json`). A paired-output 128-column smolgen-project tile that shared each loaded input tile across two output columns passed quick drift but regressed to `5.625 ms/eval` (`/tmp/lc0_autoresearch/20260606_100753_41540_rep1.json`). Reverted all code changes.
- Added the smolgen isolated parity/profiling harness (`?smolgenBench=1`) as the next-roadmap foundation for generated schedule exploration. The harness runs smolgen compress → dense1 → LN1 → dense2 → LN2 → 64-column tiled project against the CPU smolgen reference, reports final bias drift, and times each stage independently. Smoke command: `lc0-policy-only.html?smolgenBench=1&smolgenIters=10&smolgenWarmup=1&packVerify=0`; smoke result passed with max abs error `4.768e-6`, RMS `8.04e-7`, and stage averages roughly project `0.178 ms`, dense1 `0.101 ms`, compress `0.0545 ms`, dense2 `0.053 ms`, LN2 `0.0465 ms`, LN1 `0.039 ms`.
