#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/logs
DOVETAIL_LOG="artifacts/logs/dovetail-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$DOVETAIL_LOG") 2>&1
echo "[dovetail] log=$DOVETAIL_LOG"
echo "[dovetail] phase=selfplay_mix_sweep start ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

echo "[dovetail] phase=selfplay_mix_sweep done ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[dovetail] phase=playable_suite start ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
npm run eval:playable --silent
echo "[dovetail] phase=playable_suite done ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
