#!/usr/bin/env bash
set -euo pipefail

# Phase C fixed teacher-distillation benchmark.
# This replaces the saturated bootstrap-readiness score with a real model-quality
# metric over the frozen v2 lc0+Stockfish seed labels. The workload is fixed:
# dependency-free linear student, deterministic seed/split, consensus merge of
# duplicate FEN teacher labels, and a stable output artifact path.
python3 training/train_student.py \
  --merge-fen \
  --average-weights \
  --average-policy-only \
  --report-folds \
  --compare-conv-archs \
  --train data/teacher_labels.jsonl data/stockfish_teacher_labels.jsonl \
  --out artifacts/student_distill_benchmark.json
