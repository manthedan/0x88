#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
PY=${PY:-$ROOT/.venv-onnx/bin/python}
WAIT_FILE=${WAIT_FILE:-$ROOT/artifacts/cnn_av_v2_64x6_phase2_low_lr/model.onnx}
RUN=${RUN:-$ROOT/artifacts/moveformer_cnn_av_tiny_gpu_smoke_after_chain}
SIDECAR=${SIDECAR:-$ROOT/artifacts/moveformer_sidecar_100k_no_board/cache}
BOARD_CACHE=${BOARD_CACHE:-$ROOT/data/datasets/supervised_100m_elite_tcec_v1/cache_residual_h2/train/shard_0000}
ROWS=${ROWS:-100000}
MAX_STEPS=${MAX_STEPS:-200}
BATCH_SIZE=${BATCH_SIZE:-256}
CHANNELS=${CHANNELS:-32}
BLOCKS=${BLOCKS:-2}
MOVE_DIM=${MOVE_DIM:-64}
HEADS=${HEADS:-4}
LAYERS=${LAYERS:-1}
FF_DIM=${FF_DIM:-128}
LR=${LR:-3e-4}
mkdir -p "$RUN/logs" "$RUN/status" "$RUN/checkpoints"
log(){ echo "[$(date -Is)] $*" | tee -a "$RUN/logs/queue.log"; }
state(){ echo "$1" > "$RUN/status/state.txt"; }
cd "$ROOT"
state waiting_for_chain
log "waiting for $WAIT_FILE before launching MoveFormer tiny GPU smoke"
while [[ ! -s "$WAIT_FILE" ]]; do sleep 120; done
state launching
log "launching MoveFormer tiny smoke run=$RUN rows=$ROWS max_steps=$MAX_STEPS batch=$BATCH_SIZE"
nohup bash -lc "cd '$ROOT' && '$PY' training/train_moveformer_cnn_av_torch.py \
  --sidecar-cache '$SIDECAR' \
  --board-cache '$BOARD_CACHE' \
  --out '$RUN/model.pt' \
  --onnx-out '$RUN/model.onnx' \
  --meta-out '$RUN/model.meta.json' \
  --checkpoint-dir '$RUN/checkpoints' \
  --rows '$ROWS' --epochs 1 --max-steps '$MAX_STEPS' --batch-size '$BATCH_SIZE' \
  --channels '$CHANNELS' --blocks '$BLOCKS' --move-dim '$MOVE_DIM' --heads '$HEADS' --layers '$LAYERS' --ff-dim '$FF_DIM' \
  --lr '$LR' --device cuda --amp --amp-dtype bf16 --progress-every 20 \
  --onnx-legal-ks 32,64,128" > "$RUN/logs/train.log" 2>&1 &
echo $! > "$RUN/status/train.pid"
date -Is > "$RUN/status/train.started"
state running
log "started pid=$(cat "$RUN/status/train.pid")"
wait "$(cat "$RUN/status/train.pid")"
status=$?
if [[ $status -eq 0 && -s "$RUN/model.pt" ]]; then state done; touch "$RUN/status/train.done"; log "done"; else state failed; log "failed status=$status"; exit $status; fi
