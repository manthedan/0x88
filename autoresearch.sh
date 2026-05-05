#!/bin/bash
set -euo pipefail
python3 -m py_compile training/train_board_cnn.py
export CUDA_HOME="$PWD/.venv-tinygrad/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
.venv-tinygrad/bin/python training/train_board_cnn.py \
  --train data/lichess_training_400k_2000elo_2016-01.jsonl data/tcec_training_100k.jsonl \
  --max-rows "${BOARD_CNN_ROWS:-20000}" \
  --epochs "${BOARD_CNN_EPOCHS:-3}" \
  --channels "${BOARD_CNN_CHANNELS:-16}" \
  --lr "${BOARD_CNN_LR:-0.001}" \
  --out artifacts/student_board_cnn_autoresearch.json
