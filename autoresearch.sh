#!/bin/bash
set -euo pipefail
python3 -m py_compile training/train_board_cnn.py
npx tsc --noEmit >/dev/null
cargo check --quiet --manifest-path rust/tiny_leela_core/Cargo.toml --bin tiny-leela-rust-eval
npm run eval:playable --silent
rust/tiny_leela_core/target/debug/tiny-leela-rust-eval artifacts/student_distill_benchmark.json 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' 16 0 \
  | grep -E '^(best_move|METRIC rust_student|policy_legal_count|wdl=)'
