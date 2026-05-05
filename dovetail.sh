#!/usr/bin/env bash
set -euo pipefail

# Phase D self-play mix arena benchmark.
# Fixed small CPU workload for comparable early self-play research: generate a
# deterministic PUCT self-play buffer from the current student, train candidates
# with several teacher/self-play mix weights, then evaluate each candidate against
# the current 64x6 teacher-distilled baseline in the arena.
npm run selfplay:mix-sweep --silent -- \
  --backend=rust \
  --weights=0,0.05,0.1,0.25 \
  --games=2 \
  --selfplay-visits=32 \
  --max-plies=8 \
  --epochs=40 \
  --arena-games=2 \
  --arena-visits=32 \
  --adjudicate=value \
  --adjudicate-threshold=0.02 \
  --primary-conv-arch=64x6 \
  --selfplay=artifacts/selfplay_mix_arena_v1.jsonl \
  --candidate-prefix=artifacts/selfplay_mix_arena_candidate \
  --regenerate

npm run eval:playable --silent
