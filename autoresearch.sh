#!/usr/bin/env bash
set -euo pipefail

score=0

# Bootstrap project fitness score. Deterministic and intentionally transparent.
# Replace with a real Elo/loss/latency benchmark under a new benchmark_id later.

[[ -f package.json ]] && score=$((score + 5))
[[ -f tsconfig.json ]] && score=$((score + 3))
[[ -d src ]] && score=$((score + 2))
[[ -d tests ]] && score=$((score + 2))

# Core chess implementation milestones.
[[ -f src/chess/board.ts || -f src/chess/board.js ]] && score=$((score + 8))
[[ -f src/chess/movegen.ts || -f src/chess/movegen.js ]] && score=$((score + 10))
[[ -f src/chess/moveCodec.ts || -f src/chess/moveCodec.js ]] && score=$((score + 10))
[[ -f src/nn/features.ts || -f src/nn/features.js ]] && score=$((score + 10))
[[ -f src/search/puct.ts || -f src/search/puct.js ]] && score=$((score + 12))
[[ -f src/nn/evaluator.ts || -f src/nn/evaluator.js ]] && score=$((score + 8))
[[ -f src/browser/worker.ts || -f src/browser/worker.js ]] && score=$((score + 6))

# Training/deployment/evaluation milestones.
[[ -f training/model_spec.md || -f training/model.py ]] && score=$((score + 6))
[[ -f scripts/export_onnx.py || -f scripts/export_onnx.sh ]] && score=$((score + 5))
[[ -f scripts/quantize_onnx.py || -f scripts/quantize_onnx.sh ]] && score=$((score + 5))
[[ -f eval/gauntlet.ts || -f eval/gauntlet.js || -f eval/gauntlet.py ]] && score=$((score + 6))
[[ -f docs/move_encoding.md ]] && score=$((score + 4))
[[ -f docs/browser_runtime.md ]] && score=$((score + 4))
[[ -f docs/research_phases.md ]] && score=$((score + 6))
[[ -f eval/benchmark_spec.json ]] && score=$((score + 8))
[[ -f eval/phase_b_metrics.mjs ]] && score=$((score + 6))

if [[ -f eval/phase_b_metrics.mjs ]] && node eval/phase_b_metrics.mjs >/tmp/tiny-leela-phase-b.log 2>&1; then
  score=$((score + 12))
fi

# Verified tests are the main backpressure: only count them when they pass.
if [[ -f package.json ]] && command -v npm >/dev/null 2>&1; then
  if npm test >/tmp/tiny-leela-test.log 2>&1; then
    score=$((score + 20))
  fi
fi

printf 'METRIC tiny_leela_score=%s\n' "$score"
