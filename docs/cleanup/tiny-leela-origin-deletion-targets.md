# Tiny Leela-origin deletion target inventory

Branch/worktree: `audit/tiny-origin-deletion-targets` at `/Users/macthedan/projects/lc0_browser/leelaweb-tiny-cleanup-audit`.

This file is an evaluation list only: no deletions and no renames are made on this branch yet. The first cleanup pass should remove files that are clearly inherited from the old Tiny Leela repo or its old agent/research/operator workflow. Renames of active product concepts can come later under a better name.

## Guardrails

- Keep ORT/WebGPU LC0 stable default unchanged.
- Keep TVMJS/WebGPU research-only.
- Do not stage generated engine/model artifacts.
- Do not rename active runtime keys, manifest schemas, storage keys, env vars, or UI labels in this deletion pass.
- After approved deletions, run `npm run typecheck`, `npm run lc0:tvmjs-research-only-check`, targeted tests, and `git diff --check`.

## A. High-confidence delete: old agent and scheduler scaffolding

These are repository-local agent/automation artifacts from the old Tiny Leela workflow.

```text
.pi/skills/tiny-leela-artifact-hygiene/SKILL.md
.pi/skills/tiny-leela-cloud-pipeline/SKILL.md
.pi/skills/tiny-leela-eval-promotion/SKILL.md
.pi/skills/tiny-leela-gpu-priority/SKILL.md
.pi/skills/tiny-leela-knowledge-graph/SKILL.md
.pi/skills/tiny-leela-ops/SKILL.md
.ralph/analysis-roi.md
.ralph/lc0-wgsl-readback-next-roi.md
.ralph/lc0-wgsl-readback-next2-roi.md
dovetail.checks.sh
dovetail.context.jsonl
dovetail.ideas.jsonl
dovetail.md
dovetail.runs.jsonl
dovetail.scheduler.json
dovetail.sh
dovetail.tree.json
dovetail.tree.jsonl
```

## B. High-confidence delete: Tiny Leela ops package and cloud pipeline

These are not part of the current browser product/runtime path.

```text
tiny_leela_ops/__init__.py
tiny_leela_ops/__main__.py
tiny_leela_ops/artifacts.py
tiny_leela_ops/aws_cli.py
tiny_leela_ops/cli.py
tiny_leela_ops/paths.py
tiny_leela_ops/registry.py
tiny_leela_ops/state.py
cloud/aws/Dockerfile.cache-worker
cloud/aws/Dockerfile.dataset-worker
cloud/aws/batch_cpu_spot_cloudformation.yaml
cloud/aws/bootstrap_cache_batch.sh
cloud/aws/build_push_dataset_worker.sh
cloud/aws/create_budget_guardrail.sh
cloud/aws/create_operator_user.sh
cloud/aws/download_squareformer_cache.sh
cloud/aws/finalize_squareformer_cache_s3.py
cloud/aws/make_squareformer_shard_lists.py
cloud/aws/merge_squareformer_cache_manifest.py
cloud/aws/prepare_100m_bt4_after_10m_cache.sh
cloud/aws/submit_100m_h7_h8_cache_after_h8_dataset.sh
cloud/aws/submit_100m_h8_dataset_job.sh
cloud/aws/submit_h7_h8_cache_after_h8_dataset.sh
cloud/aws/submit_h8_dataset_job.sh
cloud/aws/submit_squareformer_cache_jobs.sh
cloud/aws/tiny_leela_bootstrap_policy.json
cloud/aws/validate_h8_dataset_s3.py
cloud/aws/watch_100m_h8_dataset_job.sh
cloud/aws/watch_100m_squareformer_cache_jobs.sh
cloud/aws/watch_h8_dataset_job.sh
cloud/aws/watch_squareformer_cache_jobs.sh
cloud/aws/worker_build_supervised_dataset.sh
cloud/aws/worker_squareformer_cache.sh
```

## C. High-confidence delete: old contracts/config/spec formalism

These were coupled to the old training/cache/selfplay flow, not current LC0 browser product paths.

```text
configs/gumbel_zero/phase0_zero_lane_param_card_20260510.md
configs/gumbel_zero/phase1_review.json
contracts/README.md
contracts/schemas/cache_manifest_v1.schema.json
contracts/schemas/export_target_card_v1.schema.json
contracts/schemas/failure_packet_v1.schema.json
contracts/schemas/puct_trace_v1.schema.json
contracts/schemas/selfplay_annotation_v1.schema.json
contracts/schemas/selfplay_chunk_v1.schema.json
specs/tla/ModelPromotionLanes.cfg
specs/tla/ModelPromotionLanes.tla
specs/tla/README.md
specs/tla/ShardLifecycle.cfg
specs/tla/ShardLifecycle.tla
```

Follow-up references if deleted:

```text
tests/fixtures/contracts/move_encoding_cases.jsonl
tests/fixtures/contracts/positions.edge_cases.jsonl
scripts/failure_packet_validate.mjs
```

## D. High-confidence delete: old Tiny Leela docs/roadmaps

```text
docs/tiny_leela_lc0_browser_model_roadmap.md
docs/tiny_leela_ops_toolset.md
docs/tiny_leela_research.md
knowledge/07_roadmaps/active/roadmap.current_tiny_leela_portfolio.md
tiny_leela_research.md
```

## E. High-confidence delete: legacy browser demo pages and entrypoints

These are old lab/demo pages and TS entrypoints that predate the current `lc0-*`, Maia, Monty, Reckless, Berserk, PlentyChess pages.

```text
browser-benchmark.html
browser-eval-broker-prototype.html
browser-multimodel-arena.html
browser-ort-bridge-benchmark.html
browser-rust-wasm-webgpu-benchmark.html
browser-two-model-arena.html
browser-wasm-selfplay-broker.html
client-demo.html
tiny-ort-webnn-probe.html
tiny-tvmjs-webgpu-smoke.html
src/browser/backendPlan.ts
src/browserBenchmark.ts
src/browserEvalBrokerPrototype.ts
src/browserMultiModelArena.ts
src/browserOrtBridgeMicrobench.ts
src/browserRustWasmWebgpuBench.ts
src/browserTwoModelArena.ts
src/browserWasmSelfplayBroker.ts
src/web/appState.ts
src/web/clientState.ts
src/webClient.ts
```

Follow-up edits if approved:

```text
vite.config.ts: remove those pages from `labPages`.
package.json: remove scripts that only call deleted entrypoints/scripts.
tests/web_app_state.test.mjs: delete with `src/web/appState.ts`.
```

## F. High-confidence delete: old Rust/TS parity and UCI Tiny Leela scripts

These call removed Rust bridge/Tiny UCI paths or old parity harnesses.

```text
scripts/arena_rust_ts.mjs
scripts/bench_rust_ts_board.mjs
scripts/bench_search.mjs
scripts/compare_puct_trace.mjs
scripts/compare_rust_ts_board_encoding.mjs
scripts/compare_rust_ts_onnx_eval.mjs
scripts/compare_rust_ts_search.mjs
scripts/elo_arena_parallel.mjs
scripts/merge_rust_arena_shards.py
scripts/rust_engine.mjs
scripts/smoke_uci_tiny_leela.sh
scripts/trace_ts_puct.mjs
scripts/uci_tiny_leela.mjs
public/rust_bridge/tl_wasm_selfplay_bridge.wasm
src/nn/evaluatorWorker.ts
src/nn/studentEvaluator.ts
src/nn/workerEvaluator.ts
```

Follow-up edits if approved:

```text
package.json scripts: test:rust-ts-perft, bench:search, bench:rust-ts-board, compare:rust-ts-search, compare:rust-ts-board, compare:rust-ts-onnx-eval, arena:rust-ts, trace:puct, elo:arena.
tests/perft_parity.test.mjs: remove optional Rust parity subprocess path.
```

## G. High-confidence delete: old training/cache/cloud/offload scripts

These are old Tiny Leela distributed training/data/cache workflows. They are separate from current browser runtime/productization.

```text
scripts/build_aux_cache_shards.sh
scripts/build_chessbench_av_caches_parallel.py
scripts/build_residual_cache_from_dataset.py
scripts/build_supervised_dataset_shards.py
scripts/build_supervised_dataset_streaming.py
scripts/build_winrate_augmented_dataset.py
scripts/hourly_pi_queue_doctor.sh
scripts/overnight_100m_pipeline.sh
scripts/playable_100m_model_matrix.mjs
scripts/preflight_supervised_dataset_capacity.py
scripts/queue_moveformer_tiny_smoke_after_chain.sh
scripts/remote_cpu_offload_puct_bayes_by_visit.sh
scripts/remote_cpu_offload_puct_sweep.sh
scripts/remote_cpu_offload_puct_tune.sh
scripts/remote_cpu_offload_puct_visit_curve.sh
scripts/remote_cpu_offload_quantized_eval.sh
scripts/remote_cpu_offload_rust_super_arena.sh
scripts/remote_cpu_offload_tuned_puct_anchor_sweep.sh
scripts/report_dataset_shards.py
scripts/run_100m_cnn_bench_sweeps.sh
scripts/run_100m_h2_cache_final_validation.sh
scripts/run_stockfish_full_aux.sh
scripts/submit_onnx_inference_smoke_aws.sh
scripts/tiny_leela_chill.sh
scripts/tiny_leela_hourly_checker.sh
scripts/tiny_leela_resume_after_chill.sh
scripts/tlops
scripts/update_current_model_manifest.sh
scripts/validate_cache_schema.py
```

## H. Medium-confidence delete: legacy training/data/research docs

These look old-repo/research-lane rather than current product docs. Review before deleting because some may still be useful as historical context.

```text
docs/100m-scaling-training-plan.md
docs/10m-100m-dataset-build-todo.md
docs/aws_distributed_cache_setup.md
docs/deepseek_methods_roadmap.md
docs/distributed_selfplay_training_system_design.md
docs/gumbel_zero_selfplay_track.md
docs/head_ablation_roadmap.md
docs/lc0-inspired-roadmap.md
docs/lc0_maia_gap_closure_roadmap.md
docs/maia-inspired-roadmap.md
docs/moveformer_data_schema.md
docs/protein_carbs_puct_tuning.md
docs/puffer_selfplay_infra_plan.md
docs/self_play_scaling_roadmap.md
docs/selfplay_data_pipeline.md
docs/squareformer_v2_v3_implementation_plan.md
docs/stockfish-augmentation-and-training-plan.md
docs/training_setup_observations.md
docs/transformer_model_roadmap.md
docs/turboquant_memory_roadmap.md
docs/unified_squareformer_architecture_roadmap.md
```

## I. Medium-confidence delete: legacy knowledge graph entries

The `knowledge/` tree appears mostly inherited research scaffolding. Candidate deletion can be either whole-tree deletion or a narrower prune. Review if any of this is still used by current planning.

```text
knowledge/02_concepts/architecture/concept.squareformer.md
knowledge/02_concepts/search/concept.puct.md
knowledge/02_concepts/training/concept.action_value_head.md
knowledge/02_concepts/training/concept.search_improved_self_play.md
knowledge/03_findings/architecture/finding.squareformer_training_objective.md
knowledge/03_findings/evaluation/finding.adjudication_policy.md
knowledge/03_findings/search/finding.aux_puct_routing.md
knowledge/03_findings/systems/finding.lc0_runtime_footprint_opportunities.md
knowledge/03_findings/systems/risk.browser_path_for_offline_work.md
knowledge/03_findings/systems/risk.deprecated_roadmap_active_context.md
knowledge/03_findings/training/finding.self_play_needs_search_improved_targets.md
knowledge/04_designs/evaluation/design.candidate_frontier_cards.md
knowledge/04_designs/models/design.squareformer_av_puct.md
knowledge/04_designs/systems/design.inference_optimization.md
knowledge/04_designs/systems/design.runtime_target_matrix_and_workflow_delegation.md
knowledge/04_designs/training_system/design.h7_h8_bt4_training_pipeline.md
knowledge/04_designs/training_system/design.lc0_search_distillation_pipeline.md
knowledge/05_experiments/completed/experiment.bt4_arch_roadmap_next_ablations.md
knowledge/06_decisions/decision.aws_batch_selfplay_parallelism.md
knowledge/06_decisions/decision.lc0_architecture_funnel_and_deployability_frontier.md
knowledge/08_tasks/planned/task.backlog.md
knowledge/09_agent_context/active_roadmap.md
knowledge/09_agent_context/current_architecture.md
knowledge/graph/index.json
```

Related generator to delete if `knowledge/` goes away:

```text
scripts/knowledge_graph_overlay.py
```

## J. Medium-confidence delete: old data-labeling/training scripts

These are not browser product scripts. Keep only if you still want the repo to host data/training utilities.

```text
scripts/apply_stockfish_aux.py
scripts/apply_stockfish_aux_shard.py
scripts/apply_stockfish_winrate_loss.py
scripts/audit_chessbench_winprob_perspective.py
scripts/build_position_registry.py
scripts/contempt_vs_maia.mjs
scripts/convert_stockfish_root_to_action_values.py
scripts/convert_tiny_onnx_fixed_i32.py
scripts/download_lichess_months.sh
scripts/ingest_chessbench_action_values.py
scripts/ingest_chessbench_msgpack_zst.py
scripts/ingest_lichess_elite_months.sh
scripts/ingest_lichess_months.sh
scripts/ingest_lichess_position_evals.py
scripts/lc0_chunk_inspect.py
scripts/lc0_jsonl_sanity.py
scripts/lc0_stream_v6_to_normalized.py
scripts/lc0_weighted_jsonl_sanity.py
scripts/maia3_vs_elo_contempt.mjs
scripts/maia_elo_probe.mjs
scripts/queue_cnn80_after_cnn64_av.sh
scripts/queue_extra_cnn_av_night_chain.sh
scripts/queue_more_phase2_phase3_sf64.sh
scripts/root_stockfish_label.py
scripts/root_stockfish_label_parallel.py
scripts/run_anchor_bench_latest_cnns.sh
scripts/run_distributed_visit_curve_local.sh
scripts/run_model_release_gate.sh
scripts/stockfish_cp_loss_label.py
scripts/stockfish_cp_loss_parallel.py
scripts/write_v2_public_stream_config.py
scripts/write_v2_stream_config.py
```

## K. Keep for now, possible rename later

These still appear in active product/runtime paths or tests. Rename later after choosing the new name and adding compatibility shims if needed.

```text
src/lc0/engineCatalog.ts: active `tiny` engine family and Tiny Leela labels for BT4/SquareFormer.
src/lc0/analysisBrowser.ts and src/lc0/arenaBrowser.ts: active tiny-family rows and evaluator cache labels.
src/lc0/analysisFormat.ts: `tinyPuctAnalysisLines` naming; active test coverage.
src/nn/ortRuntime.ts: `TINY_LEELA_*` env compatibility and helper names.
src/nn/squareformerTvmHybridEvaluator.ts and public/runtimes/squareformer-tvm-hybrid/*/manifest.json: manifest schema `tiny-leela.squareformer-tvm-hybrid.v1`.
scripts/lc0_tiny_strict_custom_webgpu_smoke.mjs and package script `lc0:tiny-strict-custom-webgpu-smoke`: rename later, not deletion.
docs/engine_catalog.md: contains active Tiny-family catalog row; rename later.
README.md: one stale inherited-name note can be removed after deletion cleanup.
```

## L. Initial package/vite/test cleanup expected after approved deletions

- Remove package scripts for Rust/TS parity, old search bench, arena, trace, cache schema validation.
- Remove legacy lab pages from `vite.config.ts`.
- Delete or update:
  - `tests/web_app_state.test.mjs`
  - `tests/substrate.test.mjs` references to `dovetail.ideas.jsonl`
  - `tests/perft_parity.test.mjs` optional Rust executable branch
  - `tests/engine_catalog.test.mjs` only if labels are renamed later, not during deletion pass

## M. Suggested pass order

1. Delete A-D first: old agent/ops/docs/direct Tiny artifacts.
2. Delete E-F with package/vite/test edits.
3. Delete G after confirming no cloud/training workflows are still desired in this repo.
4. Review H-J separately: these are bigger research-history/data-pipeline cuts.
5. Plan a later naming migration for K after selecting the replacement name.
