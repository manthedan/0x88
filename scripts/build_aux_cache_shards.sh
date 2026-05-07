#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT:-/home/ddbb/projects/tiny_leela}"
SHARDS=${SHARDS:-16}
JOBS=${JOBS:-8}
DEPTH=${DEPTH:-8}
MPV=${MPV:-4}
MAX_ROWS=${ROWS:-3633118}
TRAIN=data/balanced_history_train_2026mix_5m.jsonl
LABELS=data/stockfish_aug/cp_loss_2026mix_train_full_d${DEPTH}_mpv${MPV}.jsonl
AUX_DIR=data/stockfish_aug/shards_full_sfaux_d${DEPTH}_mpv${MPV}
CACHE_DIR=data/cache/residual_2026mix_full_h2_sfaux_d${DEPTH}_mpv${MPV}_shards
mkdir -p "$AUX_DIR" "$CACHE_DIR" logs
run_one() {
  local s=$1
  local aux="$AUX_DIR/shard_$(printf '%03d' "$s").jsonl"
  local cache="$CACHE_DIR/shard_$(printf '%03d' "$s")"
  if [ ! -f "$aux.done" ]; then
    python3 scripts/apply_stockfish_aux_shard.py --input "$TRAIN" --labels "$LABELS" --out "$aux" --shards "$SHARDS" --shard "$s" --max-rows "$MAX_ROWS" > "logs/apply_aux_shard_$(printf '%03d' "$s").log" 2>&1
    touch "$aux.done"
  fi
  if [ ! -f "$cache/meta.json" ]; then
    .venv-onnx/bin/python training/build_residual_feature_cache.py --input "$aux" --out "$cache" --history-plies 2 --state-planes > "logs/build_cache_shard_$(printf '%03d' "$s").log" 2>&1
  fi
}
export -f run_one
export ROOT SHARDS DEPTH MPV MAX_ROWS TRAIN LABELS AUX_DIR CACHE_DIR
seq 0 $((SHARDS-1)) | xargs -I{} -P "$JOBS" bash -lc 'run_one "$@"' _ {}
python3 - <<PY
import json, pathlib
root=pathlib.Path('$CACHE_DIR')
rows=0
for p in sorted(root.glob('shard_*/meta.json')):
 rows += json.loads(p.read_text())['rows']
(root/'manifest.json').write_text(json.dumps({'shards':[str(p.parent) for p in sorted(root.glob('shard_*/meta.json'))],'rows':rows},indent=2))
print(f'METRIC shard_cache_total_rows={rows}')
PY
