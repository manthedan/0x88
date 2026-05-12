---
id: design.candidate_frontier_cards
type: design
title: Design - Candidate frontier cards
status: active
created: 2026-05-12
updated: 2026-05-12
priority: high
depends_on:
  - [[Design - Inference optimization]]
  - [[Design - Runtime target matrix and workflow delegation]]
  - [[Finding - Self-play needs search-improved targets]]
related:
  - [[Roadmap - Current Tiny Leela portfolio]]
  - [[Decision - LC0 architecture funnel and deployability frontier]]
risks:
  - [[Risk - Move-map mismatch]]
agent_summary: >
  Every serious Tiny Leela model candidate should get one compact frontier card with required core fields and optional final-deployment fields. The card combines calibrated protocol Elo at multiple search budgets, fixed-time strength, params/FLOPs/bytes/runtime, blunder rates, and value calibration. Quantization and adaptive browser export selection are deployment polish, not routine ablation overhead.
---

# Design - Candidate frontier cards

Tiny Leela should evaluate serious candidates by whether they improve a deployment/research **frontier**, not by one isolated arena score or supervised loss. The main objective is a tiny, fast, strong engine; training compute can be large if it produces a better deployable model. "Tiny" is multi-axis: params, FLOPs/MACs, bytes, memory, latency, and strength-per-wall-clock all matter separately.

## Standard candidate frontier card

For each serious candidate, produce a compact report with at least:

```text
model
searchless Elo/proxy
v16 Elo
v64 Elo
v128 Elo
fixed-time strength, when available
params
FLOPs/MACs estimate, when available
evals/sec
ONNX/export bytes
blunder rate
value calibration
```

Expanded machine-readable fields should include:

```text
identity:
  model_id, architecture, artifact paths, git sha, data source, trainer config

strength:
  searchless proxy / policy-only arena result
  calibrated protocol Elo at v16, v64, v128
  score vs current champion
  score vs historical anchor pool

model_efficiency:
  parameter count
  estimated inference FLOPs/MACs where available
  activation/memory notes where material
  legal bucket / top-k candidate cost

runtime:
  evals/sec by target/backend where available
  PUCT positions/sec at v16/v64/v128 or fixed-time equivalent
  fixed-time strength at browser-relevant latency budgets where available
  ONNX/model bytes and export bundle bytes
  cold/warm load time when deployment-relevant

diagnostics:
  illegal move rate
  arena blunders per 100 moves
  fixed-suite tactical blunder rate
  catastrophic-regret rate
  value calibration: WDL CE, Brier/calibration bins, Q MSE where available
  policy CE/top-k on frozen dev sets
  backend parity/drift status

optional_deployment:
  deployable flag
  export variants: wasm-fp32, webgpu-fp32/fp16, int8/qat variants when quality-neutral
  browser capability detection results
  lazy-load/export selection rule
  browser warmup benchmark and selected visit budget
  quantization quality delta and drift report, if applicable
```

## Historical protocol Elo

Historical-model and anchor-engine games should become part of the standard eval, because they give a stable ruler for candidate strength. Ratings from these pools are **Tiny Leela protocol Elo**, not universal FIDE/human Elo.

Use `eval/fit_historical_ratings.py` or successor tooling to fit candidates into an anchor pool containing:

- weak historical Tiny Leela models,
- mid-strength 10M models,
- best 100M CNN/MF/BT4/SquareFormer candidates,
- current champion/anchor,
- optional Maia/Stockfish anchors when available and protocol-compatible.

Keep separate frontiers for searchless, v16, v64, v128, and fixed-time strength. A model can be best at one search budget and not another.

## Blunder-rate policy

Blunder rate should be a first-class frontier-card metric. Use three sources with distinct meanings:

1. **Arena blunder rate** from the same standard games used for historical/anchor Elo calibration. This is realistic but distribution-dependent.
2. **Fixed-suite blunder rate** from frozen tactical/regret/failure-packet positions. This is stable across candidates and should be used to detect real tactical improvement.
3. **Self-play/production blunder discovery** from self-play diagnostics and failure packets. This discovers new failure modes; representative cases should be promoted into the fixed suite.

Suggested Stockfish-derived thresholds for post-hoc annotation:

```text
mistake:      eval drop >= 100 cp or winprob drop >= 0.07
blunder:      eval drop >= 200 cp or winprob drop >= 0.15
catastrophic: eval drop >= 400 cp or winprob drop >= 0.30
```

Depth-8 Stockfish annotations are acceptable for pipeline triage. Official frontier cards should prefer deeper/node-limited annotations, for example depth 12-16 or a stable node budget.

## Standard eval ladder

A serious candidate should normally pass through:

1. Contract/parity smoke.
2. Searchless policy/value proxy.
3. Historical calibrated arena at v16.
4. Historical calibrated arena at v64.
5. Historical calibrated arena at v128.
6. Runtime benchmark on relevant targets when the candidate is serious enough to compare frontiers.
7. Arena blunder attribution from the same games.
8. Frozen tactical/failure-suite blunder check.
9. Value calibration summary.
10. Candidate frontier card emission.

Final deployable candidates add:

11. Browser WASM/WebGPU capability and export-compatibility check.
12. Lazy-loaded export variant selection.
13. Browser warmup/benchmark and adaptive visit selection.
14. Quantization report only if a quantized export is being considered; quality loss must be effectively zero.

## Promotion interpretation

Promote or keep a candidate when it improves a frontier:

- same speed/bytes/complexity and stronger,
- same strength and much faster/smaller/simpler,
- weaker at fixed visits but stronger at the relevant fixed-time deployment budget,
- strong searchless result that becomes useful after search/calibration fixes.

Do not promote solely because dev loss improves. Do not reject solely because raw ONNX MB is larger before quantization is studied. Keep as research if it improves one diagnostic but loses on the relevant deployment frontier.

Quantization is a deployment-polish lane, not an architecture-selection gate. PTQ/QAT variants should be accepted only if strength, calibration, and top-k/policy drift are effectively clean.
