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

## Evaluation hygiene

Before trusting an arena result, check:

```text
model provenance / checkpoint path
ONNX/export parity where applicable
opening suite and seed
games/pairs count
search params: visits, cpuct, AV/aux flags
engine wrapper version
protocol JSON if present
```

Prefer existing summarizers and protocol files over ad-hoc parsing.

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
- Tactical specialists should be tested on tactical suites and normal play.
- Browser deployment requires latency and bundle-size awareness, not just Elo.

## Useful files/scripts to inspect

```text
eval/arena_suite.mjs
eval/onnx_round_robin_arena.mjs
eval/search_mode_arena.mjs
eval/uci_anchor_arena.mjs
eval/summarize_anchor_summary_tsv.py
scripts/uci_tiny_leela.mjs
scripts/smoke_uci_tiny_leela.sh
docs/elo-evaluation-process.md
docs/model_manifest.md
```
