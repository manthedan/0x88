# Rust + Infra + Refactor TODO

Date: 2026-05-11

This list tracks missing Rust implementations, production infra gaps, and cross-language refactors needed to keep Tiny Leela maintainable as more deterministic chess/search/cache/eval logic moves toward Rust.

## P0 — Current Evaluation/Arena Path

- [x] Finish Rust arena sharding/offload polish.
  - [x] Keep `--shard-count` / `--shard-index` in `rust/tiny_leela_core/src/bin/arena.rs` covered by tests or repeatable smoke commands.
  - [x] Promote `scripts/merge_rust_arena_shards.py` into the standard eval workflow.
  - [x] Add validation that merged shard outputs cover exactly the expected global game ids with no duplicates.
  - [x] Add a reusable Mac mini launcher for Rust arena shards instead of one-off shell commands.
  - [x] Add durable status, pid, log, and pullback handling, ideally via `scripts/tlops`.
    - Implemented in `scripts/remote_cpu_offload_rust_super_arena.sh`; full `tlops` registry integration remains optional future polish.

- [x] Keep Rust/native ORT as the canonical backend for the 100M super arena.
  - [x] Do not restart the TS/ORT-Web super arena for this workload.
  - [x] Merge Mac mini shard results when complete.
  - [x] Generate a compact standings/summary packet from merged output.

## P0 — Rust Core Refactor

- [x] Split `rust/tiny_leela_core/src/lib.rs` monolith into modules.
  - [x] `board.rs`: board representation, square indexing, piece/color types.
  - [x] `fen.rs`: FEN parse/serialize, start position constants.
  - [x] `movegen.rs`: legal move generation, checks, pins, castling/en-passant legality.
  - [x] `move_codec.rs`: UCI conversion, move/action id mapping.
  - [x] `encoding.rs`: CNN/residual board tensors and history encodings.
  - [x] `squareformer.rs`: compact token construction and SquareFormer-specific preprocessing.
  - [x] `eval.rs`: `Evaluation`, `PositionEvaluator`, evaluator adapters.
  - [x] `onnx.rs`: native ORT evaluator implementation behind `native-ort`.
  - [x] `search.rs`: PUCT, aux/AV PUCT, search options/results.
  - [x] `arena.rs` or `matchplay.rs`: reusable game/round-robin logic shared by bin tools.

- [x] Add module-level tests while splitting.
  - [x] FEN roundtrip/property tests.
  - [x] Movegen perft/parity tests.
  - [x] Move/action-id contract tests.
  - [x] Encoding parity tests vs fixtures.
  - [x] Search trace parity tests for classic and aux/AV PUCT.

## P1 — Production Cache Building in Rust

- [x] Replace the current toy/legacy `rust/tiny_leela_core/src/bin/feature_cache.rs` with production cache builders.
  - [x] Residual/CNN H2/H8 cache builder.
  - [x] SquareFormer compact-token cache builder.
  - [x] BT4/SquareFormer h7/h8 cache builder.
    - Covered by `tiny-leela-rust-squareformer-cache --history-plies <N>` for compact SquareFormer/BT4 token caches.
  - [x] MoveFormer sidecar cache builder.
  - [x] Tactical MoveFormer sidecar cache builder, but do not resume tactical training without explicit approval.
  - [x] Action-value / Stockfish / ChessBench overlay cache ingestion where deterministic preprocessing matters.
    - Covered for canonical `teacher.action_value.v1` overlays by `tiny-leela-rust-action-value-cache`.

- [x] Make Rust cache outputs contract-first.
  - [x] Emit `cache_manifest_v1`.
    - Implemented for `tiny-leela-rust-feature-cache`, `tiny-leela-rust-residual-cache`, `tiny-leela-rust-squareformer-cache`, `tiny-leela-rust-moveformer-cache`, `tiny-leela-rust-moveformer-tactical-cache`, and `tiny-leela-rust-action-value-cache`.
  - [x] Atomic shard writes.
    - Feature-cache writes use temp-file/rename; SquareFormer cache writes to a temp output directory before publishing.
  - [x] Resume-safe shard completion markers.
  - [x] `.jsonl.zst` or binary/zstd streaming for large outputs.
    - Rust cache builders now share streaming plain JSONL / `.jsonl.zst` input handling via `for_each_jsonl_line`.
  - [x] Deterministic row ordering and checksums.
    - Rust cache metadata/manifests now declare row ordering and include SHA-256 hashes for emitted cache artifacts.
  - [x] Python compatibility readers for training.
    - Added `training/_lib/rust_cache_readers.py`; MoveFormer training now accepts Rust sidecar slot naming.

- [x] Keep Python as the training/export owner.
  - [x] Rust should build deterministic tensors/tokens/caches.
  - [x] Python should train models and export ONNX.
  - Rust cache builders now stop at deterministic preprocessing; Python readers/trainers consume those artifacts without moving training/export into Rust.

## P1 — Batched Rust Evaluator / GPU Readiness

- [x] Add a batched evaluator abstraction.
  - [x] Queue leaf evaluations across games/searches.
  - [x] Support configurable max batch size and max wait time.
  - [x] Preserve deterministic result ordering.
  - [x] Track eval latency, batch size, throughput, cache hit rate.
  - Implemented as `BatchedEvaluator` / `BatchedEvaluatorOptions` / `BatchedEvaluatorMetrics` with ticket-order tests.

- [ ] Add CUDA/native provider support after batching exists.
  - [x] Enable/configure Rust `ort` CUDA EP.
    - `OnnxEvaluator` now accepts/env-parses `ORT_EXECUTION_PROVIDERS`, `ORT_ENABLE_CUDA`, and `ORT_REQUIRE_CUDA` for native ORT provider setup.
  - [x] Require CUDA provider in CUDA smoke tests.
    - `tiny-leela-rust-onnx-matrix` emits `--require-provider CUDAExecutionProvider` for `local_cuda` smoke commands.
  - [x] Add provider fallback reporting.
    - Native Python smoke reports missing/inactive required providers; Rust native ORT exposes requested provider config.
  - [x] Benchmark CPU vs CUDA only with enough concurrent/batched evals to feed the GPU.
    - Inference matrix smoke commands carry explicit `--batches` and CUDA remains opt-in after batching support.

- [x] Add reusable eval cache layers.
  - [x] Per-search transposition/eval cache.
  - [x] Optional cross-game cache for repeated openings/positions.
  - [x] History-aware cache key for models that consume history.
  - Implemented as `CachedEvaluator` with bounded cross-game mode, history-aware keys, hit-rate metrics, and eviction tests.

## P1 — Rust Self-Play

- [x] Upgrade `rust/tiny_leela_core/src/bin/selfplay.rs` from StudentEvaluator bootstrap to production ONNX self-play.
  - [x] Native ORT model loading.
    - `tiny-leela-rust-selfplay` now accepts `--model model.onnx --meta model.meta.json` behind the `native-ort` feature while keeping JSON StudentEvaluator fallback.
  - [x] CNN/MF80/SquareFormer/ChessFormer support.
    - Self-play now uses the shared native `OnnxEvaluator`, covering board/CNN, MoveFormer/MF80 legal-move inputs, and SquareFormer/ChessFormer compact-token metadata paths.
  - [x] Classic PUCT and experimental aux/AV modes.
    - `tiny-leela-rust-selfplay` now exposes `--policy-mode classic|av|aux` plus aux/AV weights while defaulting to classic.
  - [x] `.jsonl.zst` chunk output.
  - [x] `selfplay_chunk_v1` conformance.
  - [x] Chunk manifests and checksums.
  - [x] Atomic writes and resume/retry handling.
  - Rust self-play now emits atomic plain or `.jsonl.zst` chunks, schema-tagged rows, and optional chunk manifests with SHA-256 checksums. Resume/retry safety is via temp-file publish semantics.

- [x] Keep lane separation explicit.
  - [x] Gumbel-Zero must remain rules-only/random-init/no supervised contamination.
  - [x] SUP-SP can use supervised/distilled initialization.
  - [x] Tooling should make contamination hard, not just documented.
  - `tiny-leela-rust-selfplay` refuses `--lane gumbel_zero` and only emits model-guided `sup_sp` / `eval_demo` / `other` chunks.

## P1 — Promotion / Anchor Eval Infra

- [ ] Add Rust-native promotion gates where useful.
  - [x] Stockfish anchor arena integration.
    - Rust arena supports `--adjudicate stockfish`, `--stockfish`, `--stockfish-depth`, and draw-CP adjudication.
  - [x] Maia/lc0 wrapper integration where available.
    - Rust arena supports `--baseline-uci <engine-or-wrapper> --baseline-uci-depth N`, covering Stockfish and local Maia/lc0 UCI wrappers when installed.
  - [x] PGN export from Rust arenas.
  - [x] Elo/SPRT/confidence summaries.
    - Rust arena emits PGN via `--pgn-out` and reports WDL/score-rate/Elo estimate metrics in JSON/stdout.
  - [x] Release-gate packet generation.
    - Rust arena can emit `rust_release_gate_packet_v1` via `--release-gate-out`.
  - [ ] Model manifest update helper.

- [ ] Keep promotion policy conservative.
  - [ ] Classic PUCT remains default.
  - [ ] Aux/AV PUCT remains experimental until stronger gates support it.

## P1 — TypeScript Web Client Refactor

- [ ] Replace global mutable state in `src/webClient.ts` with an explicit app state model.
  - [ ] Define `AppState` for board, ply, mode, selected square, legal moves, history, analysis state, engine state.
  - [ ] Introduce a reducer or command/event model for state transitions.
  - [ ] Separate rendering from state mutation.
  - [ ] Separate live gameplay state from analysis-board state.
  - [ ] Add a single source of truth for `historyFens` / move history.

- [ ] Isolate async engine/search interactions.
  - [ ] Add request ids / cancellation tokens to avoid stale async updates.
  - [ ] Make UI mode switches cancel or quarantine old analysis/search results.
  - [ ] Add tests for mode switching and undo/redo/history consistency.

- [ ] Keep TS as browser/runtime glue.
  - [ ] Do not duplicate deterministic chess/search semantics in TS long-term unless contract-tested against Rust.

## P1 — Python Training Script Deduplication

- [ ] Extract common training infrastructure from large scripts.
  - [ ] Shared JSONL/zstd dataset readers.
  - [ ] Shared argument parser fragments.
  - [ ] Shared training loop utilities.
  - [ ] Shared metrics/loss logging.
  - [ ] Shared checkpoint/EMA/export helpers.
  - [ ] Shared manifest/model-card writing.

- [ ] Suggested package layout.
  - [ ] `training/_lib/data.py`
  - [ ] `training/_lib/args.py`
  - [ ] `training/_lib/loops.py`
  - [ ] `training/_lib/checkpoints.py`
  - [ ] `training/_lib/export.py`
  - [ ] `training/_lib/metrics.py` already exists; expand instead of duplicating.
  - [ ] `training/_lib/encoding.py` already exists; keep Python/Rust parity explicit.

- [ ] Refactor scripts incrementally.
  - [ ] Start with `train_residual_torch.py`.
  - [ ] Then `train_squareformer_torch.py`.
  - [ ] Then `train_squareformer_v2_torch.py`.
  - [ ] Then `train_moveformer_kitchensink_torch.py`.
  - [ ] Preserve CLI compatibility during migration.

## P2 — Python Import Hygiene

- [ ] Decide policy for heavy imports.
  - [ ] Prefer top-level imports for linting/refactorability in core modules.
  - [ ] Allow lazy imports only in thin CLI wrappers when startup/help latency matters.

- [ ] Move training implementations behind importable modules.
  - [ ] CLI files should parse args and call library functions.
  - [ ] Library modules should import `torch`, `torch.nn`, etc. normally at top level.
  - [ ] Add lint/type tooling that can inspect the real implementation paths.

- [ ] Clean known inline imports.
  - [ ] `training/train_squareformer_v2_torch.py`
  - [ ] `training/train_squareformer_torch.py`
  - [ ] `training/train_residual_torch.py`
  - [ ] `training/train_moveformer_kitchensink_torch.py`

## P2 — Contracts and Fixtures

- [ ] Expand contract coverage.
  - [ ] `policy_map_v1`
  - [ ] `move_action_id_v1`
  - [ ] `board_encoding_vN`
  - [ ] `squareformer_token_cache_v1`
  - [ ] `selfplay_chunk_v1`
  - [ ] `cache_manifest_v1`
  - [ ] `puct_trace_v1` with aux/AV fields.

- [ ] Add cross-language conformance tests.
  - [ ] Rust vs TS move/action ids.
  - [ ] Rust vs Python tensor/token construction.
  - [ ] Rust vs TS search traces.
  - [ ] Rust cache output vs Python training reader.

## P2 — Runtime/Operations Cleanup

- [ ] Promote ad hoc remote offload scripts into maintained ops commands.
  - [ ] Mac mini Rust arena launch.
  - [ ] Mac mini status/watch.
  - [ ] Pull/merge shard outputs.
  - [ ] Failure packet capture.

- [ ] Fix git hygiene around ignored cloud files.
  - [ ] Resolve `.gitignore` issue where `aws/` ignores `cloud/aws/*` unintentionally.
  - [ ] Decide which cloud scripts are source and should be tracked.

## P2 — Additional Rust Hardening Gaps

- [ ] Replace hand-rolled CLI parsing in Rust bins.
  - [ ] Use a real parser such as `clap` for consistent help, validation, defaults, and error messages.
  - [ ] Add shared config structs for arena/search/eval/cache/self-play tools.
  - [ ] Emit machine-readable run config sidecars for reproducibility.

- [ ] Improve Rust error handling and diagnostics.
  - [ ] Replace `expect`/`unwrap` in production paths with contextual errors.
  - [ ] Distinguish bad input, missing artifact, ORT load failure, invalid metadata, and search/runtime failure.
  - [ ] Emit failure packets compatible with ops tooling.

- [ ] Add Rust model metadata validation.
  - [ ] Validate ONNX input/output names against meta JSON before arena/self-play starts.
  - [ ] Validate aux/AV head availability before allowing aux/AV search modes.
  - [ ] Validate external-data ONNX sidecars are present.
  - [ ] Add clear model capability cards from Rust.

- [ ] Strengthen game-state bookkeeping.
  - [ ] Track repetition, fifty-move counter, insufficient material, and claimable draws explicitly.
  - [ ] Add tests for castling-rights transitions, en-passant edge cases, promotions, and history-sensitive encodings.
  - [ ] Make draw/adjudication policy configurable and recorded in protocol JSON.

- [ ] Add Rust opening-suite utilities.
  - [ ] Validate opening FEN files.
  - [ ] Balance colors deterministically.
  - [ ] De-duplicate and classify openings.
  - [ ] Record opening id/source in arena and self-play outputs.

- [ ] Add observability and profiling hooks.
  - [ ] Per-game/search/eval timing breakdowns.
  - [ ] Node/sec, eval/sec, cache hit rate, legal-move generation time.
  - [ ] Optional tracing spans for PUCT phases.
  - [ ] Benchmarks that compare Rust/TS/Python hot paths without relying on ad hoc scripts.

- [ ] Add fuzz/property testing for Rust chess core.
  - [ ] Random legal-game generation invariants.
  - [ ] Make/unmake or FEN roundtrip invariants if unmake is added.
  - [ ] Differential tests against TS movegen and/or python-chess for sampled positions.
  - [ ] `cargo fuzz` or proptest harnesses for FEN and move parsing.

- [ ] Decide Rust packaging boundaries.
  - [ ] Keep `tiny_leela_core` as a library with thin bins.
  - [ ] Consider separate crates later: `tiny-leela-chess`, `tiny-leela-search`, `tiny-leela-onnx`, `tiny-leela-tools`.
  - [ ] Keep native ORT behind features so non-native/WASM builds remain possible.

- [ ] Plan Rust/Python interop for training readers.
  - [ ] Either expose Rust cache/token builders through CLI-only contracts, or provide Python bindings later.
  - [ ] Avoid duplicating tokenization rules in Python once Rust owns them.

- [ ] Add reproducibility controls.
  - [ ] Deterministic RNG seeds for arenas/self-play.
  - [ ] Stable tie-break policies documented and tested.
  - [ ] Protocol JSON should include git SHA, binary version, model hashes, ORT provider, thread settings, and shard info.

## P3 — Later / Nice to Have

- [ ] Rust UCI engine wrapper for Tiny Leela models.
- [ ] Rust PGN parser/position stream for dataset ingestion.
- [ ] Rust binary cache format if JSONL/zstd becomes too slow.
- [ ] Rust cloud/AWS Batch worker binaries for cache/self-play once local tools are stable.
- [ ] Rust model artifact verifier/downloader for S3/local manifests.
- [ ] TLA+/property model checks for shard lifecycle and promotion lanes.
- [ ] Browser Rust/WASM search adapter, with JS/TS ONNX callback boundary.
