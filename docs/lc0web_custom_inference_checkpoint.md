# lc0web custom WebGPU inference checkpoint

This checkpoint records the current custom-kernel path for the batch-8 f16 `lc0web` pack. Native LC0/BLAS and f32 ONNX remain the source-of-truth correctness baselines; these browser probes are staged subgraph checks for a future custom WebGPU runtime.

## Implemented probes and benchmarks

- `?kernelBench=1`: fixed-shape WGSL `MatMul + Add` benchmark with scalar/tiled/transposed variants.
- `?ortOpBench=1`: tiny generated ORT `MatMul + Add` comparison.
- `?qkvProbe=1` / `?qkvBench=1`: encoder0 Q/K/V projections.
- `?attentionScoreBench=1`: encoder0 QK score matmul including smolgen score bias.
- `?attentionScoreOrtBench=1`: tiny ORT attention-score comparison including smolgen bias input.
- `?softmaxBench=1`: row softmax over the 8×64 attention-score rows.
- `?attentionValueBench=1`: attention-probability × value matmul.
- `?attentionValueOrtBench=1`: tiny ORT batched attention-value MatMul comparison.
- `?attentionBlockBench=1`: QKV → QK(+smolgen) → softmax → value.
- `?attentionOutputBench=1`: attention block plus output projection, residual, and ln1.
- `?attentionOutputOrtBench=1`: tiny ORT comparison for attention-value output through output projection, alpha residual, and ln1.
- `?encoder0FfnBench=1`: encoder0 FFN from ln1 through dense1/bias/sqrrelu/dense2/bias/alpha residual/ln2.
- `?encoder0FfnOrtBench=1`: tiny ORT comparison for encoder0 FFN dense1/sqrrelu/dense2/alpha residual/ln2.
- `?encoder0BlockBench=1`: full encoder0 attention+FFN block through ln2.
- `?encoder0BlockOrtBench=1`: tiny ORT comparison for attention-value output through attention output projection/ln1 plus FFN/ln2.
- `?encoderStackBench=1&encoderLayers=N`: reusable WGSL encoder-block primitive loop over `/encoder0..N-1`, with per-block CPU f32 reference checks and optional per-block tiny f32 ONNX/ORT comparison for attention-output+FFN (`encoderStackOrt=1`, default).
- `?encoderStackHeadsBench=1&encoderLayers=N&encoderStackHeads=1`: hybrid end-to-end probe that feeds the custom WGSL encoder-stack output into tiny f32 ONNX/ORT policy and WDL value heads. The policy head covers the main 64×64 move-logit matmul path plus the promotion slice/add/remap path to the final 1858 LC0 policy logits.
- `?runtime=hybrid` / `?hybridEvaluator=1`: worker-owned production evaluator wiring for the lc0web pack. It builds the real `/input/planes` → `/attn_body` activation from LC0 112-plane input, runs the full custom WGSL encoder stack, then runs the tiny f32 ONNX/ORT mapped-policy + WDL heads and feeds the resulting 1858 logits through the normal legal-prior adapter. A persistent evaluator caches the loaded pack tensor views, encoder GPU buffers/bind groups/pipelines, ORT head session, and a WGSL smolgen-bias subgraph across evaluations. This remains the stable `lc0web-wgsl-encoder-ort-heads` backend.
- `?runtime=hybrid&headBackend=wgsl` / `?runtime=hybrid-wgsl-heads`: experimental evaluator wiring for `lc0web-wgsl-encoder-wgsl-heads`. It keeps the full encoder output on GPU and dispatches the WGSL mapped-policy+WDL heads before reading back only the final 1858 logits and WDL. This is still gated as experimental and does not replace `lc0web-wgsl-encoder-ort-heads`.
- `?hybridDrift=1&encoderLayers=10&hybridDriftLimit=N`: browser fixture route used by `npm run lc0:browser-hybrid-drift` to dump hybrid evaluations for native BLAS fixtures so policy/WDL drift can be compared against f32 ONNX and native LC0 BLAS.
- `?hybridSearchBench=1&runtime=hybrid&visits=N`: bounded browser route used by `npm run lc0:browser-hybrid-search-bench` to measure cached hybrid warm-eval latency plus fixed-visit PUCT latency/visits-per-second without promoting the backend.
- `npm run lc0:browser-hybrid-search-matrix`: local/remote matrix harness that reuses one Vite server and runs `lc0:browser-hybrid-search-bench` across head backends, visit counts, and search batch sizes, writing a full JSON artifact plus compact per-cell summaries.
- `npm run lc0:browser-hybrid-search-reuse-cache-matrix`: repeated-position benchmark harness for LC0 parity controls. It sweeps `reuseTree` on/off and worker-side `evalCacheEntries` counts, then records root-reuse counts, cache hits, neural misses, completed/requested visits-per-second, and batch-fill stats.
- `?mappedPolicyProbe=1`: tiny synthetic WGSL mapped-policy probe. It feeds known `policy[4096]`, `K[64×256]`, promotion weights, and a 1858-entry mapping table; then it reads only the mapped-policy output and verifies both normal move copies and promotion-bias additions against a CPU f32 reference.
- `?wgslHeadsProbe=1`: isolated experimental WGSL policy/value-head probe. It dispatches the full mapped-policy path (policy dense+mish, Q/K projections, Q×Kᵀ scale, promotion rows, final 1858-entry mapping) plus the full value-head path (value embed+mish, dense1+mish, dense2, softmax) on a deterministic encoder-shaped input, reads intermediate/final buffers back, compares them to the CPU f32 head reference, and runs the existing ORT tiny-head path for context. This is not wired into the hybrid evaluator.
- `?wgslHeadsVsOrt=1&fixtureLimit=N&encoderLayers=10`: real-encoder-output comparison probe. It builds real LC0 112-plane fixture inputs, runs the full hybrid WGSL encoder stack, feeds that actual encoder output through experimental WGSL policy/value heads, and compares the full 1858 mapped policy, WDL, nonzero/nonuniform checks, and legal best move against the stable tiny ORT heads path.
- `?encoderPrefix=/encoderN`: experimental tensor-prefix override for attention-output/FFN/full-block routes so the same plumbing can target later encoder layers.

The browser page now emits a `benchmarkReport` object with browser metadata, GPU adapter info where available, pack verification mode, and timing summaries. Full encoder0 WGSL block results also include per-stage diagnostic timings for QKV projection, attention scores, softmax, attention value, output projection + ln1, FFN dense1, FFN dense2 + residual, and ln2. Matmul-style block kernels now upload QKV/output/FFN weights transposed and use small tiled workgroups for QKV, output projection, and both FFN dense layers so each block reuses activation/weight tiles instead of every output invocation walking the full K dimension alone. The shared ln1/ln2 WGSL now uses one token workgroup with a 64-lane parallel mean/variance reduction instead of one serial invocation per token. When Chromium exposes WebGPU `timestamp-query`, the encoder0 block route also reports a GPU timestamp duration for the attention+FFN command sequence. `scripts/lc0_browser_wgsl_smokes.mjs` automates the main browser smokes, parses `maxAbsError`, and surfaces encoder-block stage/timestamp timings when present. `scripts/lc0_browser_wgsl_vs_ort_webgpu.mjs` runs fresh-session, alternating encoder0-block WGSL vs ORT WebGPU measurements and marks results as non-promotional diagnostics.

## Current evidence

Recent local Chromium/WebGPU/WASM smokes on the batch-8 f16 lc0web pack passed. These are browser smoke results, not CI guarantees:

- `npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-ffn,encoder0-block --timeout 25000`
  - `FFN_BENCH_DONE`, max absolute error about `3.34e-6`.
  - `ENCODER0_BLOCK_BENCH_DONE`, max absolute error about `3.58e-6` before tiled QKV/output/FFN matmuls, about `3.34e-6` after them, and about `2.86e-6` after the parallel ln1/ln2 reduction.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-block --timeout 25000`
  - Stage-timing smoke passed with max absolute error about `3.58e-6`.
  - One local single-iteration diagnostic sample after removing the attention→FFN queue-completion boundary and switching softmax to one workgroup per row reported stage avg timings: QKV projection `1.9 ms`, attention scores `0.4 ms`, softmax `0.2 ms`, attention value `0.3 ms`, output projection + ln1 `0.4 ms`, FFN dense1 `0.4 ms`, FFN dense2 + residual `0.4 ms`, ln2 `0.2 ms`; these include per-stage queue completion overhead and are bottleneck hints, not pure GPU timestamps.
  - The same smoke validated submitting attention-output and FFN command buffers together, without an intermediate `queue.onSubmittedWorkDone()` sync, at max absolute error about `3.58e-6`.
  - On the local Chromium/WebGPU run, `timestamp-query` was available and reported an encoder0 attention+FFN GPU timestamp duration around `0.5–0.7 ms` for one queued block; synchronized readback samples were around `1.0–2.7 ms`, showing that queue/readback overhead is material.
- `npm run lc0:browser-wgsl-smokes -- --no-server --only attention-value-ort-wasm --timeout 25000`
  - `ATTENTION_VALUE_ORT_BENCH_DONE`, max absolute error about `9.54e-7`.
- `npm run lc0:browser-wgsl-smokes -- --only attention-output-ort-wasm --timeout 25000`
  - `ATTENTION_OUTPUT_ORT_BENCH_DONE`, max absolute error about `1.67e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-ffn-ort-wasm --timeout 25000`
  - `FFN_ORT_BENCH_DONE`, max absolute error about `1.91e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-block-ort-wasm --timeout 25000`
  - `ENCODER0_BLOCK_ORT_BENCH_DONE`, max absolute error about `1.97e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder1-block --timeout 25000`
  - `ENCODER0_BLOCK_BENCH_DONE` using `encoderPrefix=/encoder1`, max absolute error about `5.72e-6`.
  - This validates the prefix-generalized tensor plumbing against a later encoder layer's weights; it still uses the staged synthetic input/reference for that layer, not true layer-to-layer activation handoff.
- `npm run lc0:browser-wgsl-smokes -- --only encoder-stack-2-wasm --timeout 25000`
  - `ENCODER_STACK_BENCH_DONE` over `/encoder0` then `/encoder1`, max absolute error about `5.25e-6` before transposed weight uploads, about `4.29e-6` after them, and about `3.34e-6` after the parallel ln1/ln2 reduction.
  - This is the first reusable encoder-block primitive loop: it feeds each WGSL block's GPU output buffer into the next block, while validating every block against a CPU f32 reference recomputed from the actual GPU handoff activation and a tiny f32 ONNX/ORT attention-output+FFN subgraph. The smolgen/QKV/softmax/attention-value portions are still checked against the CPU f32 reference, not a full block ONNX graph.
- `npm run lc0:browser-wgsl-smokes -- --no-server --only encoder-stack-10-wasm --timeout 80000`
  - `ENCODER_STACK_BENCH_DONE` over `/encoder0` through `/encoder9`, max absolute error about `2.31e-5` before the parallel ln1/ln2 reduction and about `2.29e-5` after it; per-block ORT-vs-WGSL max absolute error was about `2.46e-5` in the earlier full-stack run.
  - This validates layer-to-layer GPU-buffer handoff across the full 10-layer encoder stack for the synthetic 64×256 activation path.
- `npm run lc0:browser-wgsl-smokes -- --no-server --only encoder-stack-heads-2-wasm --timeout 120000`
  - `ENCODER_STACK_BENCH_DONE` over `/encoder0` then `/encoder1`, then tiny f32 ONNX/ORT policy/WDL heads from the WGSL stack output. Before policy remapping was included, encoder-stack max absolute error was about `4.29e-6`, policy-head diagnostic max absolute error was about `7.75e-7`, and WDL max absolute error was about `2.98e-8`. After adding the final 1858-logit mapped policy output and parallel ln1/ln2 reduction, the same smoke reported encoder-stack max absolute error about `3.34e-6`, main policy diagnostic max absolute error about `9.54e-7`, mapped-policy diagnostic max absolute error about `1.67e-6`, and WDL max absolute error about `2.98e-8`.
- Manual split-wait browser probe: `?encoderStackHeadsBench=1&encoderLayers=10&encoderStackWarmup=0&encoderStackOrt=1&encoderStackHeads=1&ep=wasm&packVerify=0`
  - `ENCODER_STACK_BENCH_DONE` over `/encoder0` through `/encoder9`, then tiny f32 ONNX/ORT policy/WDL heads from the full WGSL stack output.
  - Representative local result: encoder-stack max absolute error about `2.31e-5`, per-block ORT-vs-WGSL max absolute error about `2.46e-5`, policy-head diagnostic max absolute error about `9.54e-7`, and WDL max absolute error about `5.96e-8`.
- `npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 2 --timeout 25000 --wgsl-iters 1 --ort-iters 2`
  - Alternated fresh browser sessions in order `wgsl, ort, ort, wgsl`.
  - ORT reported `webgpu->webgpu` with WebGPU provider accepted in both ORT samples.
  - Sample medians from that run: WGSL synchronized readback/block about `3.25 ms`, ORT WebGPU average/run about `2.55 ms`; measurement-only, not promotion evidence.
- `npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 10 --timeout 25000 --wgsl-iters 3 --ort-iters 3`
  - Alternated 20 fresh browser sessions in order `wgsl, ort, ort, wgsl, ...`.
  - ORT reported `webgpu->webgpu` with WebGPU provider accepted in all ORT samples.
  - After the softmax and queue-boundary updates, sample medians were WGSL synchronized readback/block about `1.17 ms` and ORT WebGPU average/run about `2.27 ms` (`ratioWgslOverOrt ≈ 0.51`); still measurement-only, not promotion evidence.
- `npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 4 --timeout 60000 --wgsl-iters 3 --ort-iters 3`
  - Repeated fresh Chromium/WebGPU alternating run after hybrid runtime caching; ORT reported WebGPU provider present in all 4 ORT rows.
  - Representative sample medians were WGSL synchronized readback/block about `1.10 ms` and ORT WebGPU average/run about `1.30 ms` (`ratioWgslOverOrt ≈ 0.85`); still measurement-only, not promotion evidence.
- Tiny WGSL mapped-policy probe: `?mappedPolicyProbe=1`
  - `MAPPED_POLICY_PROBE_DONE` on local Chromium 149/Apple Metal with `isFallbackAdapter=false`.
  - Synthetic normal-copy and promotion-bias mapping validates in isolation: representative max abs error `2.98e-8`, normal-copy max abs error `0`, promotion max abs error `2.98e-8`, with 1792 normal outputs and 66 promotion outputs nonzero/nonuniform.
- Isolated WGSL policy/value-head probe: `?wgslHeadsProbe=1&ep=wasm&packVerify=0`
  - `WGSL_HEADS_PROBE_DONE` on local Chromium 149/Apple Metal with `isFallbackAdapter=false`.
  - The probe is intentionally not wired into the evaluator: it dispatches the full mapped-policy path plus the full value-head path on a deterministic encoder-shaped input, reads those buffers back, and compares them to the CPU f32 head reference.
  - Representative errors after adding WGSL mapped policy: policy dense max abs error about `9.83e-7`, main 64×64 policy-logit max abs error about `8.64e-7`, mapped-policy max abs error about `8.64e-7`, value embed max abs error about `3.28e-7`, and WGSL WDL max abs error about `5.96e-8`. Policy dense, policy logits, mapped policy, value embed, and WGSL WDL readbacks were all nonzero and nonuniform. The existing tiny ORT head path on the same input produced WDL `[0.2996, 0.3884, 0.3120]` with WDL-vs-reference max abs error about `2.98e-8`.
- WGSL heads vs ORT heads on real hybrid encoder output: `?wgslHeadsVsOrt=1&fixtureLimit=9&encoderLayers=10&ep=wasm&packVerify=0`
  - `WGSL_HEADS_VS_ORT_FIXTURES_DONE` on local Chromium/WebGPU for all 9 available native fixture records.
  - The probe uses the full real LC0 112-plane input path and full 10-layer hybrid WGSL encoder stack, then compares experimental WGSL heads against the stable ORT heads on the actual encoder output. Best moves matched ORT heads on `9/9`; max mapped-policy abs diff was about `3.10e-6`, max WDL abs diff about `1.19e-7`, and all WGSL mapped-policy/WDL outputs were nonzero and nonuniform under the route's gates.
- Experimental WGSL-heads evaluator smoke: `?hybridDrift=1&headBackend=wgsl&encoderLayers=10&hybridDriftLimit=1&ep=wasm&packVerify=0`
  - `HYBRID_DRIFT_DONE` with backend `lc0web-wgsl-encoder-wgsl-heads` on the startpos fixture, best move `d2d4`, and final WDL/policy readbacks nonzero/nonuniform via runtime gates. This validates the separate backend string and worker route while preserving stable `lc0web-wgsl-encoder-ort-heads` as the default.
- Post-WGSL-heads local + Mac mini matrix for experimental `lc0web-wgsl-encoder-wgsl-heads`:
  - Drift parity, local Chromium/WebGPU: `npm run lc0:browser-hybrid-drift -- --head-backend wgsl --limit 9 --timeout 300000 --baseline-mode serial` and `--limit 16` both completed. The available fixture set still contains 9 fixtures, so both evaluated 9. Best moves matched f32 ONNX and native BLAS on `9/9`; representative max drift ranges were f32 WDL `0.000156–0.000689`, native WDL `0.00189–0.00234`, f32 top-prior `0.000737–0.01665`, native top-prior `0.000824–0.01620`.
  - Drift parity, Mac mini Chromium/WebGPU: the same `--head-backend wgsl --limit 9` and `--limit 16` serial runs completed, again evaluating 9 fixtures. Best moves matched f32 ONNX and native BLAS on `9/9`; max drift was f32 WDL `0.000156`, native WDL `0.00189`, f32 top-prior `0.000737`, native top-prior `0.000824`.
  - Search latency, local Chromium/WebGPU with `requestedEp=wasm`, `packVerification=disabled`, batch `1`, 3 timed iterations per row: visits `1` => warm eval mean `39.13 ms`, search mean `84.93 ms`, `11.77 visits/s`; visits `32` => warm eval mean `40.10 ms`, search mean `1388.97 ms`, `23.04 visits/s`; visits `128` => warm eval mean `40.83 ms`, search mean `5554.80 ms`, `23.04 visits/s`. All rows returned best move `d2d4`.
  - Search latency, Mac mini Chromium/WebGPU with `requestedEp=wasm`, `packVerification=disabled`, batch `1`, 3 timed iterations per row: visits `1` => warm eval mean `36.33 ms`, search mean `72.63 ms`, `13.77 visits/s`; visits `32` => warm eval mean `36.13 ms`, search mean `1160.80 ms`, `27.57 visits/s`; visits `128` => warm eval mean `35.77 ms`, search mean `4545.73 ms`, `28.16 visits/s`. All rows returned best move `d2d4`.
  - Cleanup check after the repeated matrix found no local or remote Chrome for Testing, `agent-browser`, Vite on port 5179, or lc0 browser Node processes left.
- Latency/caching audit after WGSL-heads wiring:
  - Added per-evaluation backend phase timing to the hybrid search benchmark (`eval.phaseTimingStats` and `eval.lastBackendTiming`) and cached/predecoded the initial input-body f16 tensors (`posEncoding`, input projection weight/bias, mul/add gates) into f32 arrays at runtime construction instead of re-decoding them inside every evaluation.
  - Also removed an avoidable WGSL-head queue barrier and combines mapped-policy/WDL readback copies into the already-submitted encoder+heads command stream. The WGSL-head path now reads back only `(1858 + 3) * 4 = 7444` bytes, while the stable ORT-head path still reads back the full encoder output (`64 * 256 * 4 = 65536` bytes) before ORT heads.
  - Local Chromium/WebGPU audit run (`--eval-iters 5 --eval-warmup 2 --search-iters 1 --search-warmup 0 --visits 1`, `ep=wasm`, `packVerify=0`) showed the previous hot-path bottleneck was CPU input activation construction: before caching it averaged about `27.9 ms` of a `41 ms` warm eval. After caching, input build dropped to about `2.3–2.7 ms`; warm eval mean dropped to about `12.18 ms` for experimental WGSL heads and `14.52 ms` for stable ORT heads. Remaining synced readback/queue drain was about `9.5–9.9 ms`, making readback/synchronization the next primary bottleneck.
  - Local drift re-check after the cache/readback changes passed for both backends on 9 fixtures: WGSL-heads and ORT-heads each matched f32 ONNX and native BLAS best moves on `9/9`.
- Post-merge LC0 search parity matrix on the custom hybrid path:
  - Added `npm run lc0:browser-hybrid-search-matrix` and ran it locally and on the Mac mini after merging `main` search parity into `lc0-webgpu-pivot`. Matrix artifacts: `/tmp/lc0_postmerge_search_matrix_local.json` and `/tmp/lc0_postmerge_search_matrix_remote.json`.
  - Matrix shape: backends `ort,wgsl`, visits `1,32,128`, batches `1,2,4,8`, 3 warm eval iterations and 3 fixed-visit searches per cell, `ep=wasm`, pack verification disabled, 10 encoder layers. All cells completed with `stopReason="visit-budget"`, `completedVisits=requestedVisits`, and `rootReused=false` because this fixed-root benchmark intentionally measures fresh searches rather than reused subtrees.
  - Local Chromium/WebGPU, best batch per backend/visit: ORT heads `1v b4 34.36 visits/s`, `32v b8 66.12 visits/s`, `128v b8 59.50 visits/s`; WGSL heads `1v b1 35.70 visits/s`, `32v b8 62.65 visits/s`, `128v b2 85.28 visits/s`. Mean warm eval by group stayed about `15.8–17.5 ms` for ORT heads and `11.9–12.2 ms` for WGSL heads. One initial local WGSL-heads `128v/b1` cell returned `g1f3`; an immediate same-cell rerun returned `d2d4` at `84.43 visits/s`, and the other local/remote cells returned `d2d4`.
  - Mac mini Chromium/WebGPU, best batch per backend/visit: ORT heads `1v b2 40.42 visits/s`, `32v b4 90.88 visits/s`, `128v b4 93.81 visits/s`; WGSL heads `1v b8 56.26 visits/s`, `32v b1 112.07 visits/s`, `128v b8 114.47 visits/s`. Mean warm eval by group stayed about `14.0–15.2 ms` for ORT heads and `10.5–10.8 ms` for WGSL heads. All Mac mini cells returned best move `d2d4`.
  - Batch-size effect is workload/host dependent. Local ORT and local WGSL improved most at `32v` with `batch=8`; local WGSL `128v` preferred `batch=2/4`; Mac mini results were mostly flat across batches for `32v/128v`, especially with WGSL heads. Search stats expose `evalCalls=visits+1` and batch aggregation counts (`batchEvalCalls`, `maxEvalBatch`), but this fixed fresh-search matrix intentionally leaves `rootReused=false`.
  - Cleanup after local and remote matrix runs required killing one leftover standalone `agent-browser`/Chrome-for-Testing session from the local loop; the final leak check then showed no matching local or remote Chrome, `agent-browser`, Vite on port 5179, or lc0 browser Node processes.
- Tree reuse + eval-cache repeated-position benchmark:
  - Added worker-side `CachedLc0Evaluator` support for the hybrid evaluator, `reuseTree`/`resetBetweenSearches`/`evalCacheEntries` query plumbing in `?hybridSearchBench=1`, and `npm run lc0:browser-hybrid-search-reuse-cache-matrix` for repeated same-position cells.
  - Local Chromium/WebGPU artifact: `/tmp/lc0_reuse_cache_task1_local.json`. Matrix shape: backends `ort,wgsl`, visits `32`, batch `4`, `reuseTree=0/1`, `evalCacheEntries=0/2048`, 3 timed repeated searches per cell, no eval/search warmup, `ep=wasm`, pack verification disabled, 10 encoder layers. All cells returned `d2d4` and stopped by `visit-budget`.
  - ORT-heads results: no reuse/no cache `36.64 requested visits/s`, `99` neural misses, `0` cache hits; no reuse/cache `83.91 visits/s`, `66` cache hits, `33` neural misses, cache hit rate `0.667`; reuse/no cache `84.14 requested visits/s`, `32` completed visits across 3 searches, root reused in `2/3` timed searches; reuse/cache was similar (`82.88 requested visits/s`) because the reused root avoids re-evaluating the repeated position rather than hitting the eval cache.
  - WGSL-heads results: no reuse/no cache `42.58 requested visits/s`, `99` neural misses; no reuse/cache `104.28 visits/s`, `66` cache hits, `33` neural misses; reuse/no cache `104.66 requested visits/s`, `32` completed visits across 3 searches, root reused in `2/3`; reuse/cache `109.60 requested visits/s`. This confirms the merged LC0 controls are visible through the hybrid custom backend and separates eval-cache wins from tree-reuse wins on a repeated root.
- Batch-fill audit:
  - Added `evalBatchSizeHistogram` and derived average batch-size summaries to search stats/matrix outputs so the browser harness can distinguish full leaf batches from singleton-heavy collection.
  - Local Chromium/WebGPU artifact: `/tmp/lc0_batch_fill_audit_local.json`. Matrix shape: backends `ort,wgsl`, visits `128`, batches `1,2,4,8`, `reuseTree=0`, `evalCacheEntries=0`, one timed search per cell, no warmup, `ep=wasm`, pack verification disabled. All cells returned `d2d4`, completed `128/128` visits, and stopped by `visit-budget`.
  - Batch collection is not mostly singleton-limited in this fixed-root workload: batch `2` produced histogram `{2:64}`, batch `4` produced `{4:32}`, and batch `8` produced `{8:16}` for both ORT-heads and WGSL-heads. That means the current leaf collection/virtual-visit retry path fills requested physical batches cleanly; the observed batch-size throughput differences are more likely backend/runtime synchronization or per-eval overhead tradeoffs than failure to fill leaf batches, so no batching-threshold tuning was applied in this pass.
- Pre-WGSL-heads local + Mac mini baseline matrix for `lc0web-wgsl-encoder-ort-heads`:
  - Drift parity, local Chromium 149: `npm run lc0:browser-hybrid-drift -- --limit 9 --timeout 300000 --baseline-mode serial` and `--limit 16` both completed. The available fixture set currently contains 9 fixtures, so the `--limit 16` run also evaluated 9. Best moves matched f32 ONNX and native BLAS on `9/9`; local max drift ranges across the two runs were f32 WDL `0.00113–0.00205`, native WDL `0.00239–0.00283`, f32 top-prior `0.00172–0.00349`, native top-prior `0.00128–0.00324`.
  - Drift parity, Mac mini Chromium 148: the same `--limit 9` and `--limit 16` serial runs completed, again evaluating the 9 available fixtures. Best moves matched f32 ONNX and native BLAS on `9/9`; max drift ranges were f32 WDL `0.000156–0.000315`, native WDL `0.00189`, f32 top-prior `0.000737–0.00271`, native top-prior `0.000824–0.00238`.
  - Search latency, local Chromium 149 with `requestedEp=wasm`, `packVerification=disabled`, batch `1`, 3 timed iterations per row: visits `1` => warm eval mean `41.43 ms`, search mean `86.93 ms`, `11.50 visits/s`; visits `32` => warm eval mean `42.77 ms`, search mean `1351.37 ms`, `23.68 visits/s`; visits `128` => warm eval mean `42.30 ms`, search mean `5314.93 ms`, `24.08 visits/s`. All rows returned best move `d2d4`; search stats showed no cache hits within a fresh search and neural misses of visits+1.
  - Search latency, Mac mini Chromium 148 with `requestedEp=wasm`, `packVerification=disabled`, batch `1`, 3 timed iterations per row: visits `1` => warm eval mean `37.23 ms`, search mean `73.13 ms`, `13.67 visits/s`; visits `32` => warm eval mean `40.17 ms`, search mean `1441.93 ms`, `22.19 visits/s`; visits `128` => warm eval mean `37.23 ms`, search mean `4607.00 ms`, `27.78 visits/s`. All rows returned best move `d2d4`; search stats showed no cache hits within a fresh search and neural misses of visits+1.
  - Cleanup check after the repeated matrix found no local or remote Chrome for Testing, `agent-browser`, Vite, or lc0 browser Node processes left. This remains a baseline/bottleneck measurement, not promotion evidence.

Validation commands used during this checkpoint:

```sh
npm run typecheck
TINY_LEELA_ORT_EP=wasm node --experimental-strip-types --test tests/lc0_wgsl_kernel_probe.test.mjs
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-ffn,encoder0-block --timeout 25000
npm run lc0:browser-wgsl-smokes -- --no-server --only attention-value-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only attention-output-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder0-ffn-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder0-block-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder1-block --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder-stack-2-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --no-server --only attention-output,encoder0-ffn --timeout 50000
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-block,encoder-stack-2-wasm --timeout 50000
npm run lc0:browser-wgsl-smokes -- --no-server --only attention-output,encoder0-block --timeout 50000
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder-stack-2-wasm,encoder-stack-heads-2-wasm --timeout 80000
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder-stack-10-wasm --timeout 80000
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder-stack-heads-2-wasm --timeout 120000
# Manual browser routes:
#   /lc0-policy-only.html?mappedPolicyProbe=1
#   /lc0-policy-only.html?wgslHeadsProbe=1&ep=wasm&packVerify=0
#   /lc0-policy-only.html?wgslHeadsVsOrt=1&fixtureLimit=9&encoderLayers=10&ep=wasm&packVerify=0
npm run lc0:browser-hybrid-drift -- --no-server --limit 3 --timeout 180000
npm run lc0:browser-hybrid-drift -- --limit 9 --timeout 300000
npm run lc0:browser-hybrid-drift -- --limit 9 --timeout 300000 --baseline-mode serial
npm run lc0:browser-hybrid-drift -- --limit 16 --timeout 300000 --baseline-mode serial
npm run lc0:browser-hybrid-drift -- --head-backend wgsl --limit 9 --timeout 300000 --baseline-mode serial
npm run lc0:browser-hybrid-drift -- --head-backend wgsl --limit 16 --timeout 300000 --baseline-mode serial
npm run lc0:browser-hybrid-search-bench -- --dry-run --visits 8 --eval-iters 1 --search-iters 1
npm run lc0:browser-hybrid-search-bench -- --visits 1 --eval-iters 3 --eval-warmup 1 --search-iters 3 --search-warmup 1 --timeout 300000
npm run lc0:browser-hybrid-search-bench -- --visits 32 --eval-iters 3 --eval-warmup 1 --search-iters 3 --search-warmup 1 --timeout 300000
npm run lc0:browser-hybrid-search-bench -- --visits 128 --eval-iters 3 --eval-warmup 1 --search-iters 3 --search-warmup 1 --timeout 300000
npm run lc0:browser-hybrid-search-bench -- --head-backend wgsl --visits 1 --eval-iters 3 --eval-warmup 1 --search-iters 3 --search-warmup 1 --timeout 300000
npm run lc0:browser-hybrid-search-bench -- --head-backend wgsl --visits 32 --eval-iters 3 --eval-warmup 1 --search-iters 3 --search-warmup 1 --timeout 300000
npm run lc0:browser-hybrid-search-bench -- --head-backend wgsl --visits 128 --eval-iters 3 --eval-warmup 1 --search-iters 3 --search-warmup 1 --timeout 300000
npm run lc0:browser-hybrid-search-matrix -- --out /tmp/lc0_postmerge_search_matrix_local.json --timeout 240000
# Mac mini: rsync current leelaweb checkout, then:
#   npm run lc0:browser-hybrid-search-matrix -- --out /tmp/lc0_postmerge_search_matrix_remote.json --timeout 240000
npm run lc0:browser-wgsl-vs-ort-webgpu -- --dry-run --samples 2
npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 2 --timeout 25000 --wgsl-iters 1 --ort-iters 2
npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 10 --timeout 25000 --wgsl-iters 3 --ort-iters 3
npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 4 --timeout 60000 --wgsl-iters 3 --ort-iters 3
```

## Current interpretation

The custom path now validates a complete encoder0 block in staged WGSL form, including smolgen score bias and FFN. This is a stronger milestone than the earlier attention-core-only checkpoint, but it is still not an end-to-end LC0 evaluator:

- The reusable encoder-block loop now covers `/encoder0` → `/encoder9` with GPU-buffer handoff. The browser worker can now run a hybrid evaluator path that builds real LC0 112-plane inputs, feeds the full WGSL encoder stack, and uses tiny f32 ONNX/ORT mapped-policy + WDL heads. Persistent evaluator instances now reuse loaded pack tensor views, encoder GPU buffers/bind groups/pipelines, and the ORT head session across evaluations. The hybrid evaluator now computes smolgen bias on GPU and removes the per-layer CPU readback; it performs one final encoder-stack readback for the ORT heads. A 9-fixture browser drift sweep matched f32 ONNX and native BLAS best moves on all fixtures, with max WDL/top-prior drift vs f32 ONNX of about `0.00387`/`0.01094` before GPU smolgen and about `0.00161`/`0.00128` in a representative post-GPU-smolgen repeat.
- Timing reports both command submission/readback synchronization and, when Chromium exposes WebGPU `timestamp-query`, a GPU timestamp duration for the encoder0 attention+FFN command sequence.
- The full encoder0 benchmark no longer forces an explicit queue-completion boundary between attention-output and FFN; both command buffers are submitted together and rely on WebGPU queue ordering for the ln1-output → FFN-dense1 dependency.
- The per-stage encoder0 timing breakdown now covers tiled projection/FFN kernels and the parallel ln1/ln2 reduction; stage timings still include queue-completion overhead and remain bottleneck hints rather than promotion evidence.
- ORT tiny comparisons are same-value subgraph checks, not full deployment performance proof.

## Decision on full custom inference

Do **not** promote full custom LC0 inference yet.

Next gates before an end-to-end custom runtime:

1. Add ORT comparisons for the remaining full attention block if practical.
2. Repeat and broaden alternating browser runs against ORT WebGPU after validating the full encoder stack.
3. Remove the remaining correctness-first hybrid bottlenecks before considering promotion: the scalar WGSL smolgen matmuls, final encoder-stack readback into ORT heads, and the ORT head session itself.
4. Broaden `lc0:browser-hybrid-drift` beyond the current smoke limit and repeat Chromium/WebGPU runs before comparing promotion candidates.
5. Continue replacing remaining correctness-first kernels (for example attention score/value and any profitable fusion points) with tiled/fused variants without loosening the parity gate.
5. Preserve f32 ONNX/native parity as the correctness ladder while using f16/WebGPU as deployment target.
