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

## Product and runtime

- [`engine_catalog.md`](engine_catalog.md) — browser engine families and variant cards.
- [`engine_integration_architecture.md`](engine_integration_architecture.md) — end-to-end engine onboarding architecture.
- [`browser_runtime.md`](browser_runtime.md) — browser runtime notes.
- [`browser_runtime_configuration_and_benchmark_schema.md`](browser_runtime_configuration_and_benchmark_schema.md) — runtime benchmark/config schema.
- [`perf_canary.md`](perf_canary.md) — local CPU fixed-suite canary. Not a CI gate.
- [`browser_uci_adapter_contract.md`](browser_uci_adapter_contract.md) — browser UCI adapter contract.
- [`neural_browser_runtimes.md`](neural_browser_runtimes.md) — ONNX WebGPU, TVMJS, custom WGSL, WebNN, and ONNX QDQ runtime lanes.
- [`cpu_wasm_runtimes.md`](cpu_wasm_runtimes.md) — CPU WebAssembly runtime choices and Emscripten/Rust WASM build targets.
- [`browser_c_engine_porting.md`](browser_c_engine_porting.md) — C/C++ browser engine porting recipe.
- [`move_encoding.md`](move_encoding.md) — move encoding reference.
- [`human_vs_computer_play.md`](human_vs_computer_play.md) — Maia3 neural human modeling vs LQO/Monty-style contempt search for practical human play.
- [`engine_upgrade_audit.md`](engine_upgrade_audit.md) — `npm run engine:upgrade-audit`: the browser-engine upgrade correctness audit tool.
- [`search_contempt_design.md`](search_contempt_design.md) — search-contempt design; referenced from `src/lc0/playBrowser.ts`.
- [`engine_resource_broker_design.md`](engine_resource_broker_design.md) — CPU/GPU budget arbitration; referenced from `src/lc0/resourceBroker.ts`.
- [`arena_analysis_roadmap.md`](arena_analysis_roadmap.md) — staged arena/analysis design; referenced from `src/lc0/charts.ts`, `gameReview.ts`, and `tournament.ts`.

## Artifact distribution, hosting, and models

- [`engine_artifact_distribution.md`](engine_artifact_distribution.md) — artifact/source distribution policy.
- [`netlify_engine_artifacts.md`](netlify_engine_artifacts.md) — Netlify engine artifact handling.
- [`hosted_artifacts.md`](hosted_artifacts.md) — deployed/hosted artifact inventory.
- [`cdn_artifact_caching.md`](cdn_artifact_caching.md) — CDN architecture, compression pipeline, and operational playbook.
- [`artifact_hosting_cache_strategy.md`](artifact_hosting_cache_strategy.md) — artifact hosting cache strategy.
- [`artifact_retention_rollback_policy.md`](artifact_retention_rollback_policy.md) — retention/rollback policy; enforced by `scripts/check_artifact_retention_policy.mjs`.
- [`cloudflare_r2_artifact_validation.md`](cloudflare_r2_artifact_validation.md) — R2 artifact validation.
- [`r2_model_hosting.md`](r2_model_hosting.md) — external model hosting policy.
- [`asset_telemetry_plan.md`](asset_telemetry_plan.md) — privacy-preserving asset telemetry spec.
- [`engine_packaging_reorg.md`](engine_packaging_reorg.md) — where patched engine sources live and why.
- [`model_manifest.md`](model_manifest.md) — historical: the removed project-level model manifest builder; points to the current per-artifact manifests.
- [`model_efficiency_metrics.md`](model_efficiency_metrics.md) — model efficiency comparison metrics.
- [`model_provenance/`](model_provenance/) — per-model provenance and derivation recipes (license load-bearing):
  - [`model_provenance/bt4_soap_rem_c19000_final.md`](model_provenance/bt4_soap_rem_c19000_final.md) — Centipawn `bt4_soap_rem_c19000_final` model.
  - [`model_provenance/maia3.md`](model_provenance/maia3.md) — Maia3 upstream file, QDQ8 derivation recipe and source offer.

## Audits, parity records, and decision logs

- [`runtime_efficiency_and_release_readiness_audit_2026-07-25.md`](runtime_efficiency_and_release_readiness_audit_2026-07-25.md) — runtime efficiency and release-readiness ledger, including its own corrections.
- [`inference_caching_optimization_gap_audit_2026-07-14.md`](inference_caching_optimization_gap_audit_2026-07-14.md) — inference/caching gap audit; canonical record of measured dead ends (generic int4, speculative pipeline depth, standalone GPU legal priors).
- [`cpu_engines_simd_audit.md`](cpu_engines_simd_audit.md) — SIMD ladder parity audit (40/40 across Berserk, Viridithas).
- [`plentychess_simd_audit.md`](plentychess_simd_audit.md) — PlentyChess SIMD audit (40/40, plus the f32 tail that was silently scalar).
- [`reckless_simd_kernel_fixes.md`](reckless_simd_kernel_fixes.md) — Reckless SIMD kernel fixes (60/60 exact parity).
- [`threaded_emscripten_smp_prototype_2026-07-25.md`](threaded_emscripten_smp_prototype_2026-07-25.md) — threaded Emscripten SMP prototype: 4.4x raw NPS, zero extra plies at fixed movetime; the threading question is closed.
- [`native-lc0-search-gap.md`](native-lc0-search-gap.md) — native LC0 search feature gap.
- [`lc0-search-parity-status.md`](lc0-search-parity-status.md) — LC0 search parity status.
- [`lc0_search_parity_strictness.md`](lc0_search_parity_strictness.md) — what the search parity fixtures do and do not assert.
- [`lc0_b4_depth1_safe_benchmark_protocol.md`](lc0_b4_depth1_safe_benchmark_protocol.md) — canonical local protocol for measured fixed-suite work.
- [`enginebattle-feature-gap.md`](enginebattle-feature-gap.md) — feature gap against EngineBattle.
- [`ui_reference_delta.md`](ui_reference_delta.md) — UI delta against reference sites.
- [`arena-time-parity.md`](arena-time-parity.md) — arena time-control parity notes.

## LC0/WebGPU research lane

- [`lc0web_custom_inference_checkpoint.md`](lc0web_custom_inference_checkpoint.md) — custom WGSL inference lane checkpoint.
- [`lc0_tvmjs_research_runbook.md`](lc0_tvmjs_research_runbook.md) — TVMJS research runbook.
- [`lc0_tvm_whole_onnx_webgpu_probe.md`](lc0_tvm_whole_onnx_webgpu_probe.md) — whole-ONNX TVM WebGPU probe.
- [`lc0_tvmjs_bt4it332_onboarding_2026-06-09.md`](lc0_tvmjs_bt4it332_onboarding_2026-06-09.md) — BT4 int8 332 onboarding, including the int4 verdict.
- [`lc0_tvmjs_campaign_2026-06-09_and_bt4_plan.md`](lc0_tvmjs_campaign_2026-06-09_and_bt4_plan.md) — TVMJS campaign and BT4 plan.
- [`lc0_post_upload_pack_release_audit.md`](lc0_post_upload_pack_release_audit.md) — pack release audit.
- [`lc0_pack_serving_compression_audit.md`](lc0_pack_serving_compression_audit.md) — pack serving/compression audit.
- [`ort_webgpu_readback_diagnostics_plan.md`](ort_webgpu_readback_diagnostics_plan.md) — ORT readback diagnostics lane.
- [`lc0_t3_qdq_webnn_2026-06-10.md`](lc0_t3_qdq_webnn_2026-06-10.md) — t3 QDQ/WebNN probe notes.
- [`lc0_wasm_relaxed_simd_inference_design.md`](lc0_wasm_relaxed_simd_inference_design.md) — relaxed-SIMD WASM inference design.
- [`lc0_fused_legal_topk_readback_plan.md`](lc0_fused_legal_topk_readback_plan.md) — fused legal top-k readback plan.
- [`onnx_deploy_workflow.md`](onnx_deploy_workflow.md) — ONNX export/deploy workflow.

## CPU engine port notes, experiments, and benchmarks

- [`reckless_wasi.md`](reckless_wasi.md) — Reckless WASI port notes.
- [`reckless_browser_benchmarks.md`](reckless_browser_benchmarks.md) — Reckless browser benchmarks.
- [`reckless_relaxed_simd_probe.md`](reckless_relaxed_simd_probe.md) — Reckless relaxed-SIMD probe.
- [`reckless_wasm_optimization_notes.md`](reckless_wasm_optimization_notes.md) — running Reckless optimization notes; index into the experiment/probe docs below.
- [`reckless_wasm_next_exploration_notes.md`](reckless_wasm_next_exploration_notes.md) — next-exploration notes (threats-accumulator swizzle, threading route).
- [`reckless_wasm_opt_experiment.md`](reckless_wasm_opt_experiment.md) — wasm-opt experiment (negative result).
- [`reckless_wasm_simd_inspection.md`](reckless_wasm_simd_inspection.md) — SIMD NNUE module inspection.
- [`reckless_wasm_nnue_activate_probe.md`](reckless_wasm_nnue_activate_probe.md) — NNUE activate kernel probe.
- [`reckless_wasm_nnue_kernel_probe.md`](reckless_wasm_nnue_kernel_probe.md) — NNUE kernel probe.
- [`reckless_hot_path_profile.md`](reckless_hot_path_profile.md) — native hot-path profile.
- [`reckless_threaded_wasm_feasibility.md`](reckless_threaded_wasm_feasibility.md) — WASI threading feasibility: root-split workers over shared-memory threading.
- [`reckless_browser_native_api_plan.md`](reckless_browser_native_api_plan.md) — browser-native API facade plan.
- [`reckless_browser_api_probe.md`](reckless_browser_api_probe.md) — browser API probe.
- [`reckless_nnue_asset_size_plan.md`](reckless_nnue_asset_size_plan.md) — external NNUE asset split.
- [`reckless_lite.md`](reckless_lite.md) — Reckless Lite variant notes (registered non-default variant).
- [`reckless_lite_standalone.md`](reckless_lite_standalone.md) — standalone Reckless Lite packaging sketch.
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
- [`maia3_graph_rewrite_2026-11.md`](maia3_graph_rewrite_2026-11.md) — Maia3 ONNX graph rewrite (Einsum→MatMul, vectorized promotion head): bit-identical, ~0.76× b1 / ~0.68× grid wall on the M4; not shipped yet.
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
