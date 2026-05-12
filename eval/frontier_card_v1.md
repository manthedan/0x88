# Candidate frontier card v1

Every serious Tiny Leela model candidate should emit one compact frontier card. The goal is not training efficiency by itself; spending large training/self-play compute is acceptable if it yields a tiny, fast, strong deployable engine. Keep core candidate fields lightweight; deployment fields are optional and required only for final deployable candidates.

## Minimum table

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

## Standard sources

- Historical/protocol Elo: fit games against the standard historical/anchor pool with `eval/fit_historical_ratings.py` or successor tooling. Report as Tiny Leela protocol Elo, not FIDE Elo.
- Runtime: native/browser benchmarks such as `eval/onnx_native_inference_benchmark.py`, `eval/puct_batch_benchmark.mjs`, export target cards, and browser smoke data.
- Model efficiency: params, FLOPs/MACs estimates, activation/memory notes, legal bucket/top-k cost, raw ONNX bytes, and quantized bytes when available.
- Blunders: post-hoc Stockfish attribution from the same arena games, plus frozen tactical/failure-suite checks.
- Value calibration: WDL CE, Brier/calibration bins, Q MSE, or available trainer/dev metrics.

## Blunder-rate sources

Use three metrics, not one:

1. Arena blunder rate from the standard historical/anchor bench games.
2. Fixed-suite blunder rate from frozen tactical/regret/failure-packet positions.
3. Self-play/production blunder discovery from sidecar diagnostics; promote representative failures into the fixed suite.

Suggested thresholds:

```text
mistake:      eval drop >= 100 cp or winprob drop >= 0.07
blunder:      eval drop >= 200 cp or winprob drop >= 0.15
catastrophic: eval drop >= 400 cp or winprob drop >= 0.30
```

Depth-8 Stockfish is fine for triage. Official cards should prefer depth 12-16 or a stable node budget.

## Eval ladder

1. Contract/parity smoke.
2. Searchless policy/value proxy.
3. Historical calibrated arena at v16.
4. Historical calibrated arena at v64.
5. Historical calibrated arena at v128.
6. Runtime benchmark for relevant targets when the candidate is serious enough to compare frontiers.
7. Arena blunder attribution from those games.
8. Fixed tactical/failure-suite blunder check.
9. Value calibration summary.
10. Emit card JSON/Markdown.

Final deployable candidates additionally require:

11. Browser WASM/WebGPU compatibility check.
12. Lazy-loaded export variant selection.
13. Browser warmup/benchmark and adaptive visit selection.
14. Quantization report if PTQ/QAT export is considered; quality loss must be effectively zero.

## Interpretation

A candidate improves the frontier if it is stronger at the same speed/bytes/complexity, equally strong but faster/smaller/simpler, or stronger at the relevant fixed-time deployment budget. Do not promote solely on better supervised loss. Do not use a naive hard MB cap before quantization and runtime evidence; params, FLOPs/MACs, bytes, latency, and strength-per-wall-clock are separate axes.
