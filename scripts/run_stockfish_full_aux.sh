#!/usr/bin/env bash
set -euo pipefail
cd /home/ddbb/projects/tiny_leela
mkdir -p logs data/stockfish_aug data/cache artifacts/checkpoints
WORKERS=${STOCKFISH_WORKERS:-16}
ROWS=${ROWS:-3633118}
DEPTH=${DEPTH:-8}
MPV=${MPV:-4}
AUX_W=${AUX_Q_WEIGHT:-0.05}
LR=${LR:-0.00001}
LOG=logs/stockfish_full_aux.log
{
  echo "full stockfish aux start $(date) workers=$WORKERS rows=$ROWS depth=$DEPTH mpv=$MPV aux=$AUX_W lr=$LR"
  python3 scripts/stockfish_cp_loss_parallel.py \
    --input data/balanced_history_train_2026mix_5m.jsonl \
    --out data/stockfish_aug/cp_loss_2026mix_train_full_d${DEPTH}_mpv${MPV}.jsonl \
    --max-rows "$ROWS" --depth "$DEPTH" --multipv "$MPV" --workers "$WORKERS"
  echo "apply aux $(date)"
  python3 scripts/apply_stockfish_aux.py \
    --input data/balanced_history_train_2026mix_5m.jsonl \
    --labels data/stockfish_aug/cp_loss_2026mix_train_full_d${DEPTH}_mpv${MPV}.jsonl \
    --out data/stockfish_aug/balanced_history_train_2026mix_full_sfaux_d${DEPTH}_mpv${MPV}.jsonl \
    --max-rows "$ROWS" --weight-mode none
  echo "build train cache $(date)"
  .venv-onnx/bin/python training/build_residual_feature_cache.py \
    --input data/stockfish_aug/balanced_history_train_2026mix_full_sfaux_d${DEPTH}_mpv${MPV}.jsonl \
    --out data/cache/residual_2026mix_full_h2_sfaux_d${DEPTH}_mpv${MPV} \
    --history-plies 2 --state-planes
  if [ ! -f data/cache/residual_2026mix_dev_216k_h2/meta.json ]; then
    echo "build dev cache $(date)"
    .venv-onnx/bin/python training/build_residual_feature_cache.py \
      --input data/balanced_history_dev_2026mix_250k.jsonl \
      --out data/cache/residual_2026mix_dev_216k_h2 \
      --history-plies 2 --state-planes
  fi
  echo "train full aux $(date)"
  .venv-onnx/bin/python training/train_residual_aux_cache_torch.py \
    --cache data/cache/residual_2026mix_full_h2_sfaux_d${DEPTH}_mpv${MPV} \
    --dev-cache data/cache/residual_2026mix_dev_216k_h2 \
    --resume artifacts/checkpoints/residual_48x5_history2_2026mix_3633k_e100.best.pt \
    --out artifacts/residual_48x5_history2_2026mix_sfauxfull_d${DEPTH}_mpv${MPV}.pt \
    --onnx-out artifacts/residual_48x5_history2_2026mix_sfauxfull_d${DEPTH}_mpv${MPV}.onnx \
    --meta-out artifacts/residual_48x5_history2_2026mix_sfauxfull_d${DEPTH}_mpv${MPV}.meta.json \
    --checkpoint artifacts/checkpoints/residual_48x5_history2_2026mix_sfauxfull_d${DEPTH}_mpv${MPV}.pt \
    --epochs 1 --lr "$LR" --aux-q-weight "$AUX_W" --max-dev-policy-ce 2.85 --device cuda
  echo "done $(date)"
} 2>&1 | tee "$LOG"
