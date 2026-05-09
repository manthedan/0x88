#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"
REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_probe_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/mac_mini_gumbel_probe_$STAMP}
MODEL=${MODEL:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx}
META=${META:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json}
VISITS=${VISITS:-8,16}
DEEP_VISITS=${DEEP_VISITS:-64}
SEEDS=${SEEDS:-1,2}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
STOCKFISH_NODES=${STOCKFISH_NODES:-2000}
ORT_NUM_THREADS_REMOTE=${ORT_NUM_THREADS_REMOTE:-6}
KEEP_REMOTE=${KEEP_REMOTE:-1}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }
need "$MODEL"; need "$META"; need eval/gumbel_deep_stockfish_demo.mjs; need scripts/uci_stockfish_js_wrapper.mjs
need node_modules/onnxruntime-web; need node_modules/onnxruntime-common; need node_modules/stockfish

log "remote=$REMOTE rdir=$RDIR local_out=$LOCAL_OUT"
ssh "$REMOTE" "rm -rf '$RDIR' && mkdir -p '$RDIR'/node_modules '$RDIR'/$(dirname "$MODEL") '$RDIR'/artifacts/gumbel_demos/offload_probe"
log "sync source + node runtime subset"
rsync -az src eval scripts package.json "$REMOTE:$RDIR/"
rsync -az node_modules/onnxruntime-web node_modules/onnxruntime-common node_modules/stockfish "$REMOTE:$RDIR/node_modules/"
rsync -az "$MODEL" "$META" "$REMOTE:$RDIR/$(dirname "$MODEL")/"

log "run remote Gumbel deep/Stockfish probe"
ssh "$REMOTE" "cd '$RDIR' && ORT_NUM_THREADS=$ORT_NUM_THREADS_REMOTE node --experimental-strip-types eval/gumbel_deep_stockfish_demo.mjs \
  --model '$MODEL' \
  --meta '$META' \
  --out artifacts/gumbel_demos/offload_probe/demo.json \
  --md-out artifacts/gumbel_demos/offload_probe/summary.md \
  --visits '$VISITS' \
  --deep-visits '$DEEP_VISITS' \
  --seeds '$SEEDS' \
  --candidate-count '$CANDIDATE_COUNT' \
  --batch-size '$BATCH_SIZE' \
  --stockfish-nodes '$STOCKFISH_NODES'"

mkdir -p "$LOCAL_OUT"
rsync -az "$REMOTE:$RDIR/artifacts/gumbel_demos/offload_probe/" "$LOCAL_OUT/"
log "retrieved results to $LOCAL_OUT"
sed -n '1,80p' "$LOCAL_OUT/summary.md"
if [[ "$KEEP_REMOTE" != "1" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
  log "removed remote workdir $RDIR"
else
  log "kept remote workdir $RDIR"
fi
