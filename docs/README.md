# 0x88 docs map

This directory favors current browser-product/runtime docs over inherited
research notebooks. Historical training, cloud, agent, and knowledge-graph
material was removed in the Centipawn-origin cleanup branch.

Tracked docs should be durable product, runtime, provenance, or license
documentation. One-off cleanup inventories, agent handoff notes, and dated
working documents are local-dev artifacts: keep them under `.local-dev-docs/`
(ignored) instead of publishing them in the repository.

## Product and runtime

- [`engine_catalog.md`](engine_catalog.md) — browser engine families and variant cards.
- [`engine_integration_architecture.md`](engine_integration_architecture.md) — end-to-end engine onboarding architecture.
- [`browser_runtime.md`](browser_runtime.md) — browser runtime notes.
- [`browser_runtime_configuration_and_benchmark_schema.md`](browser_runtime_configuration_and_benchmark_schema.md) — runtime benchmark/config schema.
- [`browser_uci_adapter_contract.md`](browser_uci_adapter_contract.md) — browser UCI adapter contract.
- [`neural_browser_runtimes.md`](neural_browser_runtimes.md) — ONNX WebGPU, TVMJS, custom WGSL, WebNN, and ONNX QDQ runtime lanes.
- [`cpu_wasm_runtimes.md`](cpu_wasm_runtimes.md) — CPU WebAssembly runtime choices and Emscripten/Rust WASM build targets.
- [`browser_c_engine_porting.md`](browser_c_engine_porting.md) — C/C++ browser engine porting recipe.
- [`move_encoding.md`](move_encoding.md) — move encoding reference.
- [`human_vs_computer_play.md`](human_vs_computer_play.md) — Maia3 neural human modeling vs LQO/Monty-style contempt search for practical human play.
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
- [`model_manifest.md`](model_manifest.md) — LC0 model manifest notes.
- [`model_efficiency_metrics.md`](model_efficiency_metrics.md) — model efficiency comparison metrics.
- [`model_provenance/`](model_provenance/) — per-model provenance and derivation recipes (license load-bearing).

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

## Upstream contributions

- [`upstream/`](upstream/) — PR drafts, patches, and repro for apache/tvm and apache/tvm-ffi fixes (`PR_tvm_webgpu_readback.md`, `PR_tvm_ffi_structural_equal.md`, `repro_constant_pool_dedup.py`).
