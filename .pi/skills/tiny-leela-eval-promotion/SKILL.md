---
name: tiny-leela-eval-promotion
description: Tiny Leela model evaluation and promotion policy. Use when comparing models, planning arenas, interpreting Stockfish/Maia results, tuning PUCT/AV settings, or deciding whether a model should be promoted.
---

# Tiny Leela Eval Promotion

Use this skill for evaluation, arena, and promotion decisions.

## Current portfolio

Keep lanes distinct:

```text
CNN: incumbent/simple baseline
Tactical-MoveFormer Hybrid: tactical/search specialist
Tiny BT4 / SquareFormer: browser-deploy architecture bet
```

Do not collapse lanes based on one benchmark. Compare within lane first, then cross-lane with shared protocols.

## Defaults

- Classic PUCT remains the default search mode.
- AV / aux-PUCT is opt-in until calibrated.
- PTQ before QAT.
- OpenBench is a second-layer promotion lane, not the first gate.
- Full OpenBench adoption is blocked until `scripts/uci_tiny_leela.mjs` is stable.
- Treat Rust deterministic chess/search/evaluator paths as first-class for native eval work; TypeScript remains browser/runtime glue and legacy arena coverage.

## Evaluation hygiene

Before trusting an arena result, check:

```text
model provenance / checkpoint path
ONNX/export parity where applicable
encoding parity: Python policy/action IDs ↔ TypeScript policy/action IDs
opening suite and seed
games/pairs count
search params: visits, cpuct, AV/aux flags
engine wrapper version
protocol JSON if present
```

Prefer existing summarizers and protocol files over ad-hoc parsing.

## Compute placement for evals

Default split:

```text
local: export/parity/tiny smoke
Mac mini: bounded CPU post-training evals, PUCT visit curves, aux-PUCT tuning, UCI/Stockfish anchor sweeps
cloud: massive arenas, large self-play, large reanalysis
```

Use `docs/mac_mini_cpu_offload_plan.md` for the current Mac-mini process. The Mac mini measured about 3x faster than local CPU for CNN96 ONNX/PUCT benchmarks, so prefer `scripts/remote_cpu_offload_*.sh` wrappers for bounded CPU eval packets after local correctness gates pass.

## Rust native rewrite status

Rust now owns/validates major deterministic hot paths for native evaluation:

```text
board/FEN/legal movegen/action IDs
classic PUCT search parity plumbing
ONNX input plane construction for CNN/residual models
MoveFormer legal-action tensor construction for MF80-style models
SquareFormer/BT4 compact token + legal-action construction
native ONNX Runtime arena/eval for CNN, MF80, and SquareFormer/BT4 ONNX
```

Use these commands when a model/evaluator/search change may affect promotion confidence:

```bash
npm run compare:rust-ts-board -- --meta MODEL.meta.json
npm run compare:rust-ts-onnx-eval -- --model MODEL.onnx --meta MODEL.meta.json
npm run compare:rust-ts-search -- artifacts/student_distill_benchmark.json 1,2,4,8,16
npm run rust:arena -- --candidate-onnx C.onnx --candidate-meta C.meta.json --baseline-onnx B.onnx --baseline-meta B.meta.json
```

`eval/search_mode_arena.mjs --backend rust` is available for two-player classic-PUCT native ONNX smoke/arena runs. Treat Rust-specific results as native-backend evidence until the promotion protocol explicitly says Rust and TS/Web arenas are interchangeable.

## Pre-promotion code gates

Before using new search/evaluator/training plumbing for a promotion decision, run the focused correctness gates first:

```bash
npm run typecheck
node --experimental-strip-types eval/puct_core_tests.mjs
node --experimental-strip-types --test tests/encoding_parity.test.mjs tests/policy_map.test.mjs
npm run compare:rust-ts-board -- --meta MODEL.meta.json
```

If native ONNX, Rust search, or SquareFormer/BT4 preprocessing changed, also run the relevant Rust parity/eval gates:

```bash
cargo check --features native-ort --manifest-path rust/tiny_leela_core/Cargo.toml
cargo test --manifest-path rust/tiny_leela_core/Cargo.toml
npm run compare:rust-ts-onnx-eval -- --model MODEL.onnx --meta MODEL.meta.json --generated 4
```

If a change touches Python/TS/Rust encoding, add or update parity coverage before refactoring training scripts.

## Promotion ladder

A model should generally pass:

1. Supervised dev/loss sanity.
2. Fast smoke arena against a known nearby baseline.
3. Stockfish/Maia or varied anchor arena.
4. Search/latency sanity for browser deploy.
5. Optional OpenBench-style gate when UCI wrapper is stable.

## Known caution areas

- A model that wins at one visit count may regress elsewhere; inspect visit curves when relevant.
- AV improvements can be search-parameter sensitive; compare with classic PUCT baseline.
- BT4/SquareFormer h8 policy-only checkpoints do not have AV/action-value heads; skip aux-PUCT tuning unless using an exported AV-capable SquareFormer model.
- Tactical specialists should be tested on tactical suites and normal play.
- Browser deployment requires latency and bundle-size awareness, not just Elo.
- External-data ONNX may work in native ORT while TS/ORT Web tooling still needs special handling; do not infer browser compatibility from native Rust alone.
- Do not expand the architecture matrix without a written kill criterion and a planned anchor protocol.

## Useful files/scripts to inspect

```text
eval/arena_suite.mjs
eval/onnx_round_robin_arena.mjs
eval/search_mode_arena.mjs
eval/uci_anchor_arena.mjs
eval/summarize_anchor_summary_tsv.py
scripts/compare_rust_ts_board_encoding.mjs
scripts/compare_rust_ts_onnx_eval.mjs
scripts/compare_rust_ts_search.mjs
scripts/bench_rust_ts_board.mjs
scripts/uci_tiny_leela.mjs
scripts/smoke_uci_tiny_leela.sh
docs/elo-evaluation-process.md
docs/model_manifest.md
```
