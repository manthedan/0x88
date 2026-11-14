# 0x88 docs map

This directory favors current browser-product/runtime docs over inherited
research notebooks. Historical training, cloud, agent, and knowledge-graph
material was removed in the Centipawn-origin cleanup branch.

Tracked docs are durable product, runtime, provenance, or license
documentation, plus dated decision records: investigations and upgrade audits
whose verdict, evidence, or rejected alternative the code or a later decision
relies on. A dated record keeps its original findings; later changes go in a new
record or a short correction note rather than a rewrite. One-off cleanup
inventories, agent handoff notes, and scratch working notes stay under
`.local-dev-docs/` (ignored).

Every tracked Markdown file under `docs/` is linked from this index;
`tests/docs_index.test.mjs` fails otherwise.

Descriptions distinguish current guides, dated records and unimplemented
proposals. A historical record is not an instruction to restore its old defaults.

## Product and runtime

- [`engine_catalog.md`](engine_catalog.md) — browser engine families and variant cards.
- [`engine_integration_architecture.md`](engine_integration_architecture.md) — end-to-end engine onboarding architecture.
- [`browser_runtime.md`](browser_runtime.md) — current runtime/context guide, with separately scoped smoke notes.
- [`browser_runtime_configuration_and_benchmark_schema.md`](browser_runtime_configuration_and_benchmark_schema.md) — runtime benchmark/config schema.
- [`perf_canary.md`](perf_canary.md) — local CPU fixed-suite canary. Not a CI gate.
- [`browser_uci_adapter_contract.md`](browser_uci_adapter_contract.md) — browser UCI adapter contract.
- [`neural_browser_runtimes.md`](neural_browser_runtimes.md) — ONNX WebGPU, TVMJS, custom WGSL, WebNN, and ONNX QDQ runtime lanes.
- [`cpu_wasm_runtimes.md`](cpu_wasm_runtimes.md) — CPU WebAssembly runtime choices and Emscripten/Rust WASM build targets.
- [`browser_c_engine_porting.md`](browser_c_engine_porting.md) — C/C++ browser engine porting recipe.
- [`move_encoding.md`](move_encoding.md) — Centipawn v1 move encoding reference (deferred).
- [`human_vs_computer_play.md`](human_vs_computer_play.md) — Maia3 neural human modeling vs LQO/Monty-style contempt search for practical human play.
- [`engine_upgrade_audit.md`](engine_upgrade_audit.md) — `npm run engine:upgrade-audit`: the browser-engine upgrade correctness audit tool.
- [`search_contempt_design.md`](search_contempt_design.md) — current Play contempt settings plus dated algorithm/experiment provenance; referenced from `src/lc0/playBrowser.ts`.
- [`engine_resource_broker_design.md`](engine_resource_broker_design.md) — current CPU budgets and resource-class leases; GPU evaluation arbitration deferred. Referenced from `src/lc0/resourceBroker.ts`.
- [`arena_analysis_roadmap.md`](arena_analysis_roadmap.md) — current capability/backlog map, with historical design appendix; referenced from `src/lc0/charts.ts`, `gameReview.ts`, and `tournament.ts`.

## Artifact distribution, hosting, and models

- [`engine_artifact_distribution.md`](engine_artifact_distribution.md) — artifact/source distribution policy.
- [`netlify_engine_artifacts.md`](netlify_engine_artifacts.md) — Netlify engine artifact handling.
- [`hosted_artifacts.md`](hosted_artifacts.md) — deployed/hosted artifact inventory.
- [`cdn_artifact_caching.md`](cdn_artifact_caching.md) — CDN architecture, compression pipeline, and operational playbook.
- [`artifact_hosting_cache_strategy.md`](artifact_hosting_cache_strategy.md) — artifact hosting cache strategy.
- [`artifact_retention_rollback_policy.md`](artifact_retention_rollback_policy.md) — retention/rollback policy; enforced by `scripts/check_artifact_retention_policy.mjs`.
- [`cloudflare_r2_artifact_validation.md`](cloudflare_r2_artifact_validation.md) — R2 artifact validation.
- [`asset_telemetry_plan.md`](asset_telemetry_plan.md) — unimplemented privacy-preserving asset telemetry proposal.
- [`model_manifest.md`](model_manifest.md) — historical: the removed project-level model manifest builder; points to the current per-artifact manifests.
- [`model_efficiency_metrics.md`](model_efficiency_metrics.md) — Centipawn model efficiency metrics (deferred).
- [`model_provenance/`](model_provenance/) — per-model provenance and derivation recipes (license load-bearing):
  - [`model_provenance/s4fix_v_ema.md`](model_provenance/s4fix_v_ema.md) — served Centipawn model, contract, search settings and strength limits.
  - [`model_provenance/s4fix_v_ema_verification.md`](model_provenance/s4fix_v_ema_verification.md) — local parity/search results, option comparison and publication cautions.
  - [`model_provenance/bt4_soap_rem_c19000_final.md`](model_provenance/bt4_soap_rem_c19000_final.md) — historical Centipawn model.
  - [`model_provenance/maia3.md`](model_provenance/maia3.md) — Maia3 upstream file, QDQ8 and graph-rewrite derivation recipes and source offer.

## Audits, parity records, and decision logs

- [`runtime_efficiency_and_release_readiness_audit_2026-07-25.md`](runtime_efficiency_and_release_readiness_audit_2026-07-25.md) — runtime efficiency and release-readiness ledger, including its own corrections.
- [`inference_caching_optimization_gap_audit_2026-07-14.md`](inference_caching_optimization_gap_audit_2026-07-14.md) — inference/caching gap audit; canonical record of measured dead ends (generic int4, speculative pipeline depth, standalone GPU legal priors).
- [`cpu_engines_simd_audit.md`](cpu_engines_simd_audit.md) — SIMD ladder parity audit (40/40 across Berserk, Viridithas).
- [`plentychess_simd_audit.md`](plentychess_simd_audit.md) — PlentyChess SIMD audit (40/40, plus the f32 tail that was silently scalar).
- [`reckless_simd_kernel_fixes.md`](reckless_simd_kernel_fixes.md) — Reckless SIMD kernel fixes (60/60 exact parity).
- [`threaded_emscripten_smp_prototype_2026-07-25.md`](threaded_emscripten_smp_prototype_2026-07-25.md) — dated Stormphrax/Node SMP experiment: ~4.4x raw NPS, ~zero extra fixed-time plies; current Lazy-SMP promotion rejected on this measured workload, not a universal threading verdict.
- [`lc0-search-parity-status.md`](lc0-search-parity-status.md) — browser-facing parity milestone record and current native-only boundaries.
- [`lc0_search_parity_strictness.md`](lc0_search_parity_strictness.md) — what the search parity fixtures do and do not assert.
- [`lc0_b4_depth1_safe_benchmark_protocol.md`](lc0_b4_depth1_safe_benchmark_protocol.md) — canonical local protocol for measured fixed-suite work.

## LC0/WebGPU research lane

- [`lc0web_custom_inference_checkpoint.md`](lc0web_custom_inference_checkpoint.md) — dated custom-WGSL research ledger, including rejected schedules; not a production-default guide.
- [`lc0_tvmjs_research_runbook.md`](lc0_tvmjs_research_runbook.md) — current LC0 self-hosted non-default TVMJS runbook, with dated probe/campaign evidence; v0 forces ONNX.
- [`lc0_tvmjs_bt4it332_onboarding_2026-06-09.md`](lc0_tvmjs_bt4it332_onboarding_2026-06-09.md) — BT4 int8 332 onboarding, including the int4 verdict.
- [`lc0_post_upload_pack_release_audit.md`](lc0_post_upload_pack_release_audit.md) — pack release audit.
- [`lc0_pack_serving_compression_audit.md`](lc0_pack_serving_compression_audit.md) — pack serving/compression audit.
- [`ort_webgpu_readback_diagnostics_plan.md`](ort_webgpu_readback_diagnostics_plan.md) — diagnostic guide and dated readback experiments; native-EP kernel profiling is unavailable, not zero GPU time.
- [`lc0_t3_qdq_webnn_2026-06-10.md`](lc0_t3_qdq_webnn_2026-06-10.md) — t3 QDQ/WebNN probe notes.
- [`lc0_wasm_relaxed_simd_inference_design.md`](lc0_wasm_relaxed_simd_inference_design.md) — historical WASM/Relaxed SIMD proposal, linked to measured MMI verdicts and current ORT artifact/thread controls.
- [`lc0_fused_legal_topk_readback_plan.md`](lc0_fused_legal_topk_readback_plan.md) — unimplemented fused legal top-k readback proposal with parity/throughput gates.

## CPU engine port notes, experiments, and benchmarks

- [`reckless_browser_benchmarks.md`](reckless_browser_benchmarks.md) — dated browser/API/reset/NNUE measurements and current production ladder.
- [`reckless_relaxed_simd_probe.md`](reckless_relaxed_simd_probe.md) — dated old f32-tail relaxed-SIMD probe; superseded artifact, preserved negative result.
- [`reckless_wasm_optimization_notes.md`](reckless_wasm_optimization_notes.md) — current Reckless build/runtime guide, with dated external-NNUE evidence, unmeasured swizzle sketch, and historical threading alternatives.
- [`reckless_wasm_opt_experiment.md`](reckless_wasm_opt_experiment.md) — wasm-opt experiment (negative result).
- [`reckless_wasm_simd_inspection.md`](reckless_wasm_simd_inspection.md) — dated SIMD NNUE inspection, before the later kernel fixes.
- [`reckless_wasm_nnue_activate_probe.md`](reckless_wasm_nnue_activate_probe.md) — dated standalone NNUE activation probe; backend integration followed.
- [`reckless_wasm_nnue_kernel_probe.md`](reckless_wasm_nnue_kernel_probe.md) — NNUE kernel probe.
- [`reckless_hot_path_profile.md`](reckless_hot_path_profile.md) — dated native proxy profile, not browser codegen evidence.
- [`reckless_browser_native_api_plan.md`](reckless_browser_native_api_plan.md) — implemented experimental API design, native probe history, and negative SIMD performance result.
- [`reckless_lite.md`](reckless_lite.md) — Reckless Lite variant notes (registered non-default variant).
- [`reckless_lite_standalone.md`](reckless_lite_standalone.md) — unimplemented standalone Reckless Lite packaging proposal (kept in place).
- [`stockfish_relaxed_simd_experiment.md`](stockfish_relaxed_simd_experiment.md) — Stockfish relaxed-SIMD experiment; shipped in the Stockfish corresponding-source archive.
- [`stormphrax_browser_port.md`](stormphrax_browser_port.md) — Stormphrax Emscripten port; shipped in the Stormphrax corresponding-source archive.
- [`plentychess_browser_port.md`](plentychess_browser_port.md) — PlentyChess Emscripten port notes.
- [`viridithas_wasi.md`](viridithas_wasi.md) — Viridithas WASI port notes.
- [`viridithas_browser_benchmarks.md`](viridithas_browser_benchmarks.md) — Viridithas browser benchmarks.
- [`berserk_browser_benchmarks.md`](berserk_browser_benchmarks.md) — Berserk browser benchmarks.

## Upgrade records

- [`stockfish_19_upgrade.md`](stockfish_19_upgrade.md) — Stockfish.js 18.0.7 → 19.0.1: pins, networks, parity and release requirements.
- [`plentychess_upgrade_v8.md`](plentychess_upgrade_v8.md) — PlentyChess 8.0.0 upgrade audit (2026-10-10).
- [`viridithas_v20_upgrade.md`](viridithas_v20_upgrade.md) — Viridithas move from the pre-release commit to the v20.0.0 tag.
- [`reckless_upgrade_2026-10-10.md`](reckless_upgrade_2026-10-10.md) — Reckless upgrade audit: NO-GO, production stays on `00106174`.

## Dated investigation records

- [`ort_1_30_ab_2026-10.md`](ort_1_30_ab_2026-10.md) — ORT Web 1.27 → 1.30 native-WebGPU A/B: do not upgrade.
- [`ort_batching_jspi_2026-10.md`](ort_batching_jspi_2026-10.md) — ORT 1.27 WebGPU batching and JSPI experiments.
- [`ort_native_webgpu_profiler_2026-10.md`](ort_native_webgpu_profiler_2026-10.md) — what kernel profiling ORT 1.27's native WebGPU EP actually offers.
- [`batching_strength_2026-10.md`](batching_strength_2026-10.md) — batching strength re-gate; no product default changes.
- [`kernel_hillclimb_roofline_2026-11.md`](kernel_hillclimb_roofline_2026-11.md) — WebGPU kernel roofline: per-dispatch GPU time vs measured peak for the shipped nets, and where hand-written kernels could pay off.
- [`maia3_graph_rewrite_2026-11.md`](maia3_graph_rewrite_2026-11.md) — Maia3 ONNX graph rewrite (Einsum→MatMul, vectorized promotion head): bit-identical, ~0.76× b1 / ~0.68× grid wall on the M4, ~0.87× b1 on an RTX 3090; the default Maia3 model since release 2026-11-12.kernel-releases.1.
- [`lc0_small_splitk_2026-11.md`](lc0_small_splitk_2026-11.md) — Lc0 Small split-K ONNX rewrite: ~1.5× wall / ~1.95× GPU per WebGPU invoke at batch 1 on the Apple M4, parity gate passes; ~1.46× slower on an RTX 3090 (paired); served to Apple WebGPU sessions only since release 2026-11-12.kernel-releases.1.
- [`ort_invoke_overhead_2026-11.md`](ort_invoke_overhead_2026-11.md) — ORT WebGPU per-invoke CPU overhead for Lc0 Small: graph capture + GPU-resident IO, bit-identical, ~0.58× search time on an RTX 3090 and ~0.45-0.62× on a shared M4 (paired); the ORT 1.27 capture output-handle leak is avoided with ORT-owned outputs; search-worker default, released 2026-11-13 (`2026-11-13.wave3.1`).
- [`bignet_smolgen_splitk_2026-11.md`](bignet_smolgen_splitk_2026-11.md) — big-net smolgen split-K rewrite: t3 ~0.93× per invoke and 0.90-0.94× per move on the Apple M4 (paired), served to Apple WebGPU sessions only, released 2026-11-13 (`2026-11-13.wave3.1`); BT4 within noise and LQO not shipped; RTX 3090 slower on wall time.
- [`lc0_small_splitk_nvidia_2026-11.md`](lc0_small_splitk_nvidia_2026-11.md) — Lc0 Small split-K on NVIDIA (RTX 3090, Vulkan proxy): per-kernel GPU win/loss, why the full rewrite loses (CPU-bound dispatch cost), targeted variants; no gate-passing variant beats the shipped model, Apple-only selection kept.
- [`bignet_fused_attention_2026-11.md`](bignet_fused_attention_2026-11.md) — Fused 64-token attention for BT4/t3 (lab feasibility): f32-accumulate WGSL kernel 2.4-2.5× faster than ORT's attention subgraph on an RTX 3090 and more accurate than shipped; MultiHeadAttention graph rewrite measured −4 to −8% GPU per invoke, but its parity gate flips on near-tie fixtures (BT4 on M4, t3 on 3090); projected −6 to −12% for the kernel via a small ORT patch (recommended route).
- [`centipawn_f16_2026-10.md`](centipawn_f16_2026-10.md) — Centipawn mixed-f16 TVMJS module: rejected at the numerical gate.
- [`centipawn_padded_encoding.md`](centipawn_padded_encoding.md) — Centipawn TVMJS evaluator encodes real rows once and copies padding.
- [`centipawn_tree_reuse_2026-10.md`](centipawn_tree_reuse_2026-10.md) — Centipawn retained-tree / Play yielding pilot: parked.
- [`centipawn_tvmjs_subgroups_2026-10.md`](centipawn_tvmjs_subgroups_2026-10.md) — Centipawn TVMJS subgroup rebuild: staged experimental, not promoted.
- [`analysis_continuity_2026-10.md`](analysis_continuity_2026-10.md) — Analysis continuity: model storage, evaluation cache and subtree reuse.
- [`play_early_stop_2026-10.md`](play_early_stop_2026-10.md) — Play neural early stop: ship root dominance only.
- [`hidden_tab_residency_2026-10.md`](hidden_tab_residency_2026-10.md) — hidden-tab pause and Play CPU-engine residency.
- [`wave2_integration_smoke_2026-10.md`](wave2_integration_smoke_2026-10.md) — wave-2 integration smoke and the Analysis pause fix.
- [`cpu_coop_cancel_2026-10.md`](cpu_coop_cancel_2026-10.md) — warm CPU cooperative cancellation: persistent Viridithas in Analysis.
- [`cpu_engine_lifecycle_2026-10.md`](cpu_engine_lifecycle_2026-10.md) — CPU-engine lifecycle waste: Hash/Threads command deduplication.
- [`cpu_engine_build_flags_matrix_2026-10.md`](cpu_engine_build_flags_matrix_2026-10.md) — C/C++ engine compiler-flag matrix: `-O2 -flto` for Berserk.
- [`arena_parallel_games_2026-10.md`](arena_parallel_games_2026-10.md) — parallel CPU Arena games: K=2 as the desktop Auto setting.
- [`arena_simul_gate_2026-11.md`](arena_simul_gate_2026-11.md) — Arena simul feasibility gate: bounded experiment only.
- [`lowend_memory_harness_2026-10.md`](lowend_memory_harness_2026-10.md) — memory harness rerun on a desktop baseline (not a phone).

## Upstream contributions

- [`upstream/`](upstream/) — PR drafts, patches, and repro for apache/tvm and apache/tvm-ffi fixes (`repro_constant_pool_dedup.py`):
  - [`upstream/PR_tvm_webgpu_readback.md`](upstream/PR_tvm_webgpu_readback.md) — apache/tvm: pad unaligned WebGPU GPU→CPU readback copies.
  - [`upstream/PR_tvm_ffi_structural_equal.md`](upstream/PR_tvm_ffi_structural_equal.md) — apache/tvm-ffi: StructuralEqual compares tensor content, matching StructuralHash.
