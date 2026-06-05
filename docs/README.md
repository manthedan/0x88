# lc0_webgpu docs map

This directory is intentionally treated as a **living research notebook plus stable project docs**. The repository is now `lc0_webgpu`; many older notes still use the inherited `tiny_leela` name and remain useful historical/reference material until they are retired or renamed. The current source of truth for planning is now the Markdown knowledge graph under `knowledge/`, especially `knowledge/09_agent_context/active_roadmap.md` and `knowledge/07_roadmaps/active/roadmap.current_tiny_leela_portfolio.md`. This index keeps existing file paths stable while making it clear what to read first, what is active, and what is historical/reference material.

## Status labels

- **Current**: canonical or near-canonical for today's work.
- **Active**: still being iterated; may contain open TODOs or experimental branches.
- **Reference**: useful background/spec/design material, but not the execution source of truth.
- **Historical**: captures earlier thinking/results; do not treat as current without checking newer docs/artifacts.
- **Superseded**: mostly replaced by newer docs, but retained for provenance.
- **Seed**: small note/idea stub.

## Start here

For current planning, start in `knowledge/` first. Use `docs/` as stable references and source material after checking the KG.

| Doc | Status | Why |
| --- | --- | --- |
| [`expert_handoff_2026-05-current.md`](expert_handoff_2026-05-current.md) | Current | Best single packet for an outside expert: project state, open questions, and where help matters. |
| [`process_and_architecture_flows.md`](process_and_architecture_flows.md) | Current | Diagram-first atlas for model lanes, architecture breakdowns, training loops, cloud pipelines, self-play, eval, and deployment. |
| [`model_manifest.md`](model_manifest.md) | Current | Human-readable model inventory and promotion context. |
| [`engine_catalog.md`](engine_catalog.md) | Current | Browser engine-family and variant cards: source/version anchors, adapters, feature parity, size/speed, onboarding process. |
| [`browser_uci_adapter_contract.md`](browser_uci_adapter_contract.md) | Current | Standard TypeScript/behavioral contract for browser UCI engines before selector promotion. |
| [`browser_c_engine_porting.md`](browser_c_engine_porting.md) | Current | Standard low-effort Emscripten-first intake recipe for C/C++ browser UCI engines. |
| [`berserk_browser_benchmarks.md`](berserk_browser_benchmarks.md) | Active | Berserk Emscripten worker lifecycle and first rotated-FEN benchmark captures. |
| [`model_efficiency_metrics.md`](model_efficiency_metrics.md) | Current | How we compare strength/bytes/params/latency/search usefulness. |
| [`release_gate_and_distributed_bench.md`](release_gate_and_distributed_bench.md) | Current | Promotion gates, protocol cards, distributed benchmark shape. |
| [`mac_mini_cpu_offload_plan.md`](mac_mini_cpu_offload_plan.md) | Current | Compute split and Mac-mini workflow for CPU-bound post-training evals, arenas, tuning, and self-play limits. |
| [`100m-scaling-training-plan.md`](100m-scaling-training-plan.md) | Active | Current CNN/100M supervised scaling plan and anti-overfit notes. |
| [`transformer_model_roadmap.md`](transformer_model_roadmap.md) | Reference / source | SquareFormer/TinyBT roadmap source material; current promotion/freeze decision lives in the KG. |
| [`unified_squareformer_architecture_roadmap.md`](unified_squareformer_architecture_roadmap.md) | Reference / source | Unified SquareFormer architecture plan; current promotion/freeze decision lives in the KG. |
| [`squareformer_v2_v3_implementation_plan.md`](squareformer_v2_v3_implementation_plan.md) | Reference / source | Implementation-level SquareFormer V2/V3 source material. |
| [`small_bt4_progression.md`](small_bt4_progression.md) | Reference / source | TinyBT-static → TinyBT-AV → smolgen-lite → MiniBT progression source material; not a live ablation queue. |
| [`deepresearch_architecture_triage_2026-05.md`](deepresearch_architecture_triage_2026-05.md) | Active | Triage of external DeepResearch architecture analysis: accepted useful ideas only. |
| [`unsloth_rl_economics_triage_2026-05.md`](unsloth_rl_economics_triage_2026-05.md) | Active | Triage of Unsloth RL/GRPO economics: candidate-regret ranking, chunked aux losses, actor/trainer memory lifecycle, PTQ→QAT. |
| [`puffer_selfplay_infra_plan.md`](puffer_selfplay_infra_plan.md) | Active | PufferLib/MiniZero-inspired local self-play worker plan: batched actors, WAL/chunks, ChessOcean env tests, resign/Gumbel experiments. |
| [`head_ablation_roadmap.md`](head_ablation_roadmap.md) | Active | CNN/MoveFormer/aux-head ablation matrix and calibration plan. |

## Current execution docs

### Evaluation, release, deployment

| Doc | Status | Notes |
| --- | --- | --- |
| [`elo-evaluation-process.md`](elo-evaluation-process.md) | Current | Arena/protocol methodology and Elo evaluation pitfalls. |
| [`release_gate_and_distributed_bench.md`](release_gate_and_distributed_bench.md) | Current | Release gates, protocol cards, distributed jobs. |
| [`mac_mini_cpu_offload_plan.md`](mac_mini_cpu_offload_plan.md) | Current | Mac-mini CPU offload process for post-training evals, PUCT tuning, anchor arenas, and two-machine self-play limits. |
| [`model_efficiency_metrics.md`](model_efficiency_metrics.md) | Current | Size/latency/FLOPs/browser deployment metrics. |
| [`model_manifest.md`](model_manifest.md) | Current | Curated model inventory. |
| [`engine_catalog.md`](engine_catalog.md) | Current | Standard cards and onboarding process for browser engine families/variants. |
| [`browser_uci_adapter_contract.md`](browser_uci_adapter_contract.md) | Current | Required adapter methods, lifecycle rules, diagnostics, asset-check responsibilities, and promotion checklist for browser UCI engines. |
| [`browser_c_engine_porting.md`](browser_c_engine_porting.md) | Current | Emscripten-first C/C++ engine intake recipe, Stockfish.js reference observations, and deferred optimization ladder. |
| [`berserk_browser_benchmarks.md`](berserk_browser_benchmarks.md) | Active | Berserk Emscripten worker lifecycle smoke, arena/analysis checks, and first rotated-FEN benchmark snapshot. |
| [`onnx_deploy_workflow.md`](onnx_deploy_workflow.md) | Current | ONNX simplification/parity/deployment workflow. |
| [`browser_runtime.md`](browser_runtime.md) | Reference | Browser runtime plan; short and older, still useful. |
| [`search_aux_head_calibration.md`](search_aux_head_calibration.md) | Active | AV/aux-PUCT calibration notes. |

### Data, teachers, self-play, and training substrate

| Doc | Status | Notes |
| --- | --- | --- |
| [`process_and_architecture_flows.md`](process_and_architecture_flows.md) | Current | Visual process map covering supervised training, cloud Batch/S3, Gumbel SUP-SP, clean Zero, Spark-style curation, eval, and deployment. |
| [`puffer_selfplay_infra_plan.md`](puffer_selfplay_infra_plan.md) | Active | Near-term local self-play infrastructure plan with MiniZero/Puffer systems lessons, ChessOcean validation, and chunk schema. |
| [`self_play_scaling_roadmap.md`](self_play_scaling_roadmap.md) | Reference / Active | Longer self-play roadmap; now includes MiniZero-style batched actor loop, config separation, Gumbel-root and resign calibration TODOs. |
| [`distributed_selfplay_training_system_design.md`](distributed_selfplay_training_system_design.md) | Reference / future architecture | Distributed design drawing from MiniZero/KataGo/lczero-training; use now for schemas, WAL, accepted-model discipline, and validation. |
| [`public_teacher_data_sources_plan.md`](public_teacher_data_sources_plan.md) | Active | Public teacher overlays and SquareFormer-AV-PUCT data source plan. |
| [`aws_distributed_cache_setup.md`](aws_distributed_cache_setup.md) | Active | AWS CPU Spot/S3/ECR/Batch scaffold for first distributed cache/reanalysis jobs. |
| [`training_setup_observations.md`](training_setup_observations.md) | Current | Current 100M CNN training setup observations. |
| [`stockfish-augmentation-and-training-plan.md`](stockfish-augmentation-and-training-plan.md) | Reference | Stockfish augmentation plan; useful but not always latest queue source. |
| [`teacher_setup.md`](teacher_setup.md) | Reference | Teacher engine setup; some environment details may be historical. |
| [`lichess-ingest.md`](lichess-ingest.md) | Reference | Lichess PGN supervised pretraining ingest notes. |
| [`10m-100m-dataset-build-todo.md`](10m-100m-dataset-build-todo.md) | Historical | Earlier dataset-build TODO; check active queue scripts/manifests for latest. |
| [`opening-bias-and-finetuning.md`](opening-bias-and-finetuning.md) | Historical | Earlier opening-bias diagnosis; still useful as a caution. |

### Architecture and model roadmaps

| Doc | Status | Notes |
| --- | --- | --- |
| [`100m-scaling-training-plan.md`](100m-scaling-training-plan.md) | Active | CNN/residual scaling baseline. |
| [`head_ablation_roadmap.md`](head_ablation_roadmap.md) | Active | Aux-head and search calibration experiments. |
| [`transformer_model_roadmap.md`](transformer_model_roadmap.md) | Reference / source | SquareFormer/TinyBT roadmap source; KG decision controls current promotion/freeze state. |
| [`unified_squareformer_architecture_roadmap.md`](unified_squareformer_architecture_roadmap.md) | Reference / source | Unifies SquareFormer/TinyBT direction; use KG for current execution. |
| [`squareformer_v2_v3_implementation_plan.md`](squareformer_v2_v3_implementation_plan.md) | Reference / source | Concrete SquareFormer V2/V3 implementation plan; not a live queue. |
| [`small_bt4_progression.md`](small_bt4_progression.md) | Reference / source | BT4-inspired tiny progression source; not a live ablation queue. |
| [`deepresearch_architecture_triage_2026-05.md`](deepresearch_architecture_triage_2026-05.md) | Active | External research triage; preserves only useful additions: search-light framing, geometry/history bias, move-query decoder, regret metrics. |
| [`unsloth_rl_economics_triage_2026-05.md`](unsloth_rl_economics_triage_2026-05.md) | Active | Unsloth-inspired training-loop economics: candidate groups, chunked AV/regret losses, actor throughput, memory lifecycle, QAT ladder. |
| [`channelformer_architecture.md`](channelformer_architecture.md) | Active | ChannelFormer CNN architecture note. |
| [`moveformer_data_schema.md`](moveformer_data_schema.md) | Current | MoveFormer sidecar cache schema. |
| [`move_encoding.md`](move_encoding.md) | Current | Policy/move encoding reference. |

## Research notes and idea banks

These are useful for inspiration and provenance.  Treat them as **reference**, not current execution plans, unless a newer roadmap points back to them.

| Doc | Status | Notes |
| --- | --- | --- |
| [`BT4_response.md`](BT4_response.md) | Reference | Detailed small-BT analogue recommendation. |
| [`BT4_diagrammed.md`](BT4_diagrammed.md) | Reference | BT4/lczero-training explanation and diagrams. |
| [`chessformer_ideas.md`](chessformer_ideas.md) | Reference | ChessFormer notes and transferable ideas. |
| [`chessformer_handoff.md`](chessformer_handoff.md) | Reference | ChessFormer handoff notes. |
| [`Deepseek_Chess_ideas.md`](Deepseek_Chess_ideas.md) | Reference | DeepSeek-inspired teacher/distillation ideas. |
| [`deepseek_methods_roadmap.md`](deepseek_methods_roadmap.md) | Reference | DeepSeek methods adapted to tiny Leela. |
| [`deepseek_searchlight_architectures.md`](deepseek_searchlight_architectures.md) | Reference | Large search-light architecture brainstorm. |
| [`searchless_chess_transformer_designs.md`](searchless_chess_transformer_designs.md) | Reference | Large searchless/search-light brainstorm. |
| [`transformer_token_brainstorming.md`](transformer_token_brainstorming.md) | Reference | Tokenization and hybrid architecture brainstorm. |
| [`Turbo_quant_chess_ideas.md`](Turbo_quant_chess_ideas.md) | Reference | TurboQuant-inspired compression thoughts. |
| [`turboquant_memory_roadmap.md`](turboquant_memory_roadmap.md) | Reference | Compressed memory roadmap; later-stage work. |
| [`queen_safety_tactical_verification_design.md`](queen_safety_tactical_verification_design.md) | Reference | Tactical/queen-safety verification design bank. |
| [`self_play_scaling.md`](self_play_scaling.md) | Reference | Self-play compatibility and long-form notes. |
| [`self_play_scaling_roadmap.md`](self_play_scaling_roadmap.md) | Reference | Self-play roadmap; not the immediate supervised/BT4 queue. |
| [`distributed_selfplay_training_system_design.md`](distributed_selfplay_training_system_design.md) | Reference / future architecture | MiniZero/KataGo/lczero-training-inspired design. Use now for accepted-model discipline, schemas, WAL/atomic uploads, and teacher-reanalysis patterns; full distributed self-play is later. |

## Historical / superseded docs

Keep these for provenance, but prefer the current docs above for decisions.

| Doc | Status | Replaced or contextualized by |
| --- | --- | --- |
| [`tiny_leela_research.md`](tiny_leela_research.md) | Historical | Original project framing; current work is now split into CNN / Tactical-MoveFormer / TinyBT lanes. |
| [`scaling-and-architecture-todo.md`](scaling-and-architecture-todo.md) | Superseded | `100m-scaling-training-plan.md`, `head_ablation_roadmap.md`, `transformer_model_roadmap.md`. |
| [`expert-handoff-current-state.md`](expert-handoff-current-state.md) | Historical | Useful snapshot, but current status is in artifacts/status files and newer roadmap docs. |
| [`lc0-inspired-roadmap.md`](lc0-inspired-roadmap.md) | Superseded | `lc0_maia_gap_closure_roadmap.md`, `unified_squareformer_architecture_roadmap.md`, `small_bt4_progression.md`. |
| [`maia-inspired-roadmap.md`](maia-inspired-roadmap.md) | Superseded | `lc0_maia_gap_closure_roadmap.md` and current anchor/eval docs. |
| [`maia_lc0_infra_gap_analysis.md`](maia_lc0_infra_gap_analysis.md) | Reference | Background for `lc0_maia_gap_closure_roadmap.md`. |
| [`lc0_maia_gap_closure_roadmap.md`](lc0_maia_gap_closure_roadmap.md) | Active/Reference | Still relevant for infra gap closure, but not the immediate model-training source of truth. |
| [`architecture-ladder-1m-64x6-smoke.md`](architecture-ladder-1m-64x6-smoke.md) | Historical | Earlier 1M smoke result. |
| [`research_phases.md`](research_phases.md) | Historical/Reference | Phase framing from early dovetail/autoresearch work. |
| [`fun_goals.md`](fun_goals.md) | Seed | Small idea stub. |

## Cleanup policy for future passes

A KG-backed archive process has started at `knowledge/08_tasks/planned/task.legacy_docs_archive_after_kg_extraction.md`. Do not move/delete old docs in bulk; classify, extract current decisions into KG, then move only clearly superseded docs with stubs or updated references.

1. **Do not delete useful notes.** Mark as historical/superseded first.
2. **Avoid moving paths used by scripts/tests.** If a file is moved, leave a short stub at the old path or update references in the same change.
3. **Promote one canonical doc per topic.** Older brainstorms should point to the canonical roadmap instead of accumulating new TODOs.
4. **Separate status from artifacts.** Durable queue state remains under `artifacts/*/status.txt`, `done`, and `failed`; docs should summarize decisions and links, not become live dashboards.
5. **Archive only after two passes.** First pass marks status here; second pass can move clear historical files into an archive folder with stubs if desired.

## Likely second-pass consolidation targets

- Merge `scaling-and-architecture-todo.md` into the current CNN/TinyBT/Tactical-MoveFormer roadmap docs, then turn it into a stub.
- Collapse older lc0/Maia docs into `lc0_maia_gap_closure_roadmap.md` plus a historical appendix.
- Split large brainstorms into `research-notes/` only after adding stubs or updating links.
- Keep `move_encoding.md`, `moveformer_data_schema.md`, `model_efficiency_metrics.md`, and `release_gate_and_distributed_bench.md` at stable paths because they are practical references.
