#!/usr/bin/env bash
set -Eeuo pipefail

# Offload FP32-vs-dynamic-INT8 ONNX parity, inference latency, and PUCT depth sweeps to Mac mini.
# Default SPECS are current CNN96 e08 and MF80 e01. Format per line:
#   label|fp_model|fp_meta|int8_model|int8_meta

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"
ACTION=${ACTION:-launch}   # launch | pull | clean
REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
JOB_NAME=${JOB_NAME:-mac_mini_quantized_eval_$STAMP}
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_quantized_eval_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/$JOB_NAME}
DETACH=${DETACH:-1}
CLEAN_START=${CLEAN_START:-1}
KEEP_REMOTE=${KEEP_REMOTE:-1}
ORT_THREADS=${ORT_THREADS:-6}
EVAL_CACHE_ENTRIES_REMOTE=${EVAL_CACHE_ENTRIES_REMOTE:-150000}
POSITIONS=${POSITIONS:-128}
REPEATS=${REPEATS:-5}
INFERENCE_BATCHES=${INFERENCE_BATCHES:-1,4,8,16,32}
PUCT_VISITS=${PUCT_VISITS:-32,64,128,256}
PUCT_BATCHES=${PUCT_BATCHES:-1,8,16,32}
PUCT_POSITIONS=${PUCT_POSITIONS:-4}
PUCT_REPEATS=${PUCT_REPEATS:-3}
TACTICAL_VISITS=${TACTICAL_VISITS:-16,32,64,128}
TACTICAL_CPUCTS=${TACTICAL_CPUCTS:-1.2,1.5,1.8}
RUN_TACTICAL_SWEEP=${RUN_TACTICAL_SWEEP:-1}

DEFAULT_SPECS=$'cnn96_e8|artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx|artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json|artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/cnn96x8_100m_e8_dynamic_int8.onnx|artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/cnn96x8_100m_e8_dynamic_int8.meta.json\nmf80_e1_k128|artifacts/top3_100m_overnight_20260509/mf80_100m/e01/model_k128.onnx|artifacts/top3_100m_overnight_20260509/mf80_100m/e01/model_k128.meta.json|artifacts/top3_100m_overnight_20260509/mf80_100m/e01/mf80_100m_e1_k128_dynamic_int8.onnx|artifacts/top3_100m_overnight_20260509/mf80_100m/e01/mf80_100m_e1_k128_dynamic_int8.meta.json'
SPECS=${SPECS:-$DEFAULT_SPECS}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }

pull_results(){
  mkdir -p "$LOCAL_OUT"
  if ssh "$REMOTE" "test -d '$RDIR/artifacts/remote_offload/$JOB_NAME'"; then
    rsync -az "$REMOTE:$RDIR/artifacts/remote_offload/$JOB_NAME/" "$LOCAL_OUT/"
  fi
  if ssh "$REMOTE" "test -f '$RDIR/run.log'"; then rsync -az "$REMOTE:$RDIR/run.log" "$LOCAL_OUT/remote_run.log"; fi
  if ssh "$REMOTE" "test -f '$RDIR/status.txt'"; then rsync -az "$REMOTE:$RDIR/status.txt" "$LOCAL_OUT/remote_status.txt"; fi
  log "pulled results to $LOCAL_OUT"
  [[ -f "$LOCAL_OUT/remote_status.txt" ]] && tail -30 "$LOCAL_OUT/remote_status.txt" || true
  find "$LOCAL_OUT" -maxdepth 2 -type f \( -name '*summary.tsv' -o -name 'status.final' \) -print 2>/dev/null | sort || true
}

if [[ "$ACTION" == "pull" ]]; then pull_results; exit 0; fi
if [[ "$ACTION" == "clean" ]]; then ssh "$REMOTE" "rm -rf '$RDIR'"; log "removed remote workdir $RDIR"; exit 0; fi
if [[ "$ACTION" != "launch" ]]; then echo "bad ACTION=$ACTION (expected launch|pull|clean)" >&2; exit 2; fi

need eval/onnx_parity_check.mjs
need eval/onnx_inference_benchmark.mjs
need eval/puct_batch_benchmark.mjs
need eval/puct_sweep_onnx.mjs
need node_modules/onnxruntime-web
need node_modules/onnxruntime-common
while IFS='|' read -r label fp fpmeta int8 int8meta; do
  [[ -n "${label:-}" ]] || continue
  need "$fp"; need "$fpmeta"; need "$int8"; need "$int8meta"
done <<< "$SPECS"

sync_file(){
  local p="$1"
  [[ -n "$p" ]] || return 0
  [[ -e "$p" ]] || return 0
  ssh "$REMOTE" "mkdir -p '$RDIR/$(dirname "$p")'"
  rsync -az "$p" "$REMOTE:$RDIR/$(dirname "$p")/"
}

log "remote=$REMOTE rdir=$RDIR local_out=$LOCAL_OUT detach=$DETACH"
if [[ "$CLEAN_START" == "1" ]]; then ssh "$REMOTE" "rm -rf '$RDIR'"; fi
ssh "$REMOTE" "mkdir -p '$RDIR'/node_modules '$RDIR/artifacts/remote_offload/$JOB_NAME'"
log "sync source + ONNX runtime subset + models"
rsync -az src eval scripts package.json "$REMOTE:$RDIR/"
rsync -az node_modules/onnxruntime-web node_modules/onnxruntime-common "$REMOTE:$RDIR/node_modules/"
while IFS='|' read -r label fp fpmeta int8 int8meta; do
  [[ -n "${label:-}" ]] || continue
  sync_file "$fp"; sync_file "$fpmeta"; sync_file "$int8"; sync_file "$int8meta"
done <<< "$SPECS"

RUNNER=$(mktemp)
cat > "$RUNNER" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
cd __RDIR__
BASE='artifacts/remote_offload/__JOB_NAME__'
STATUS='status.txt'
mkdir -p "$BASE"
mark(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"; }
trap 'rc=$?; mark "FAILED rc=$rc phase=${PHASE:-unknown}"; echo failed > "$BASE/status.final"; exit $rc' ERR
: > "$STATUS"
export ORT_NUM_THREADS='__ORT_THREADS__'
export ORT_INTRA_OP_NUM_THREADS='__ORT_THREADS__'
export EVAL_CACHE_ENTRIES='__EVAL_CACHE_ENTRIES_REMOTE__'
SPECS_FILE="$BASE/specs.tsv"
cat > "$SPECS_FILE" <<'SPECS_EOF'
__SPECS__
SPECS_EOF
printf 'label\tmodel_kind\tbatch\tmedian_ms_per_pos\tpositions_per_s\n' > "$BASE/inference_summary.tsv"
printf 'label\tmodel_kind\tvisits\tbatch\tmedian_ms\tvisits_per_s\n' > "$BASE/puct_latency_summary.tsv"
printf 'label\tmetric\tvalue\n' > "$BASE/parity_summary.tsv"
mark 'START quantized eval job=__JOB_NAME__ positions=__POSITIONS__ repeats=__REPEATS__ puct_visits=__PUCT_VISITS__'
while IFS='|' read -r label fp fpmeta int8 int8meta; do
  [[ -n "${label:-}" ]] || continue
  PHASE="parity_${label}"
  OUT="$BASE/$label"; mkdir -p "$OUT"
  mark "PARITY label=$label"
  node --experimental-strip-types eval/onnx_parity_check.mjs \
    --model-a "$fp" --model-b "$int8" --meta "$fpmeta" --positions __POSITIONS__ --tolerance 999 \
    > "$OUT/parity.log" 2>&1 || true
  awk -v label="$label" '/^METRIC /{sub(/^METRIC /,""); split($0,a,"="); print label "\t" a[1] "\t" a[2]}' "$OUT/parity.log" >> "$BASE/parity_summary.tsv" || true
  for kind in fp int8; do
    if [[ "$kind" == fp ]]; then model="$fp"; meta="$fpmeta"; else model="$int8"; meta="$int8meta"; fi
    PHASE="inference_${label}_${kind}"
    mark "INFERENCE label=$label kind=$kind"
    node --experimental-strip-types eval/onnx_inference_benchmark.mjs \
      --model "$model" --meta "$meta" --label "${label}_${kind}" \
      --positions __POSITIONS__ --repeats __REPEATS__ --batches __INFERENCE_BATCHES__ \
      > "$OUT/inference_${kind}.log" 2>&1
    awk -v label="$label" -v kind="$kind" '/^RESULT /{b=""; ms=""; ps=""; for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="batch")b=a[2]; if(a[1]=="median_ms_per_pos")ms=a[2]; if(a[1]=="median_positions_per_s")ps=a[2];} if(b!="") print label "\t" kind "\t" b "\t" ms "\t" ps}' "$OUT/inference_${kind}.log" >> "$BASE/inference_summary.tsv"
    PHASE="puct_${label}_${kind}"
    mark "PUCT_LATENCY label=$label kind=$kind"
    node --experimental-strip-types eval/puct_batch_benchmark.mjs \
      --model "$model" --meta "$meta" --visits __PUCT_VISITS__ --batches __PUCT_BATCHES__ \
      --positions __PUCT_POSITIONS__ --repeats __PUCT_REPEATS__ \
      > "$OUT/puct_latency_${kind}.log" 2>&1
    awk -v label="$label" -v kind="$kind" '/^RESULT /{v=""; b=""; ms=""; nps=""; for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="visits")v=a[2]; if(a[1]=="batch")b=a[2]; if(a[1]=="median_ms")ms=a[2]; if(a[1]=="visits_per_s")nps=a[2];} if(v!="") print label "\t" kind "\t" v "\t" b "\t" ms "\t" nps}' "$OUT/puct_latency_${kind}.log" >> "$BASE/puct_latency_summary.tsv"
    if [[ '__RUN_TACTICAL_SWEEP__' == '1' ]]; then
      PHASE="tactical_${label}_${kind}"
      mark "TACTICAL_SWEEP label=$label kind=$kind"
      node --experimental-strip-types eval/puct_sweep_onnx.mjs \
        --model "$model" --meta "$meta" --out "$OUT/tactical_sweep_${kind}.json" \
        --visits-list __TACTICAL_VISITS__ --cpuct-list __TACTICAL_CPUCTS__ \
        > "$OUT/tactical_sweep_${kind}.log" 2>&1 || true
    fi
  done
done < "$SPECS_FILE"
mark 'DONE quantized eval'
echo succeeded > "$BASE/status.final"
EOF
python3 - <<'PY' "$RUNNER" "$RDIR" "$JOB_NAME" "$ORT_THREADS" "$EVAL_CACHE_ENTRIES_REMOTE" "$SPECS" "$POSITIONS" "$REPEATS" "$INFERENCE_BATCHES" "$PUCT_VISITS" "$PUCT_BATCHES" "$PUCT_POSITIONS" "$PUCT_REPEATS" "$TACTICAL_VISITS" "$TACTICAL_CPUCTS" "$RUN_TACTICAL_SWEEP"
from pathlib import Path
import sys
path=Path(sys.argv[1])
keys=['__RDIR__','__JOB_NAME__','__ORT_THREADS__','__EVAL_CACHE_ENTRIES_REMOTE__','__SPECS__','__POSITIONS__','__REPEATS__','__INFERENCE_BATCHES__','__PUCT_VISITS__','__PUCT_BATCHES__','__PUCT_POSITIONS__','__PUCT_REPEATS__','__TACTICAL_VISITS__','__TACTICAL_CPUCTS__','__RUN_TACTICAL_SWEEP__']
text=path.read_text()
for k,v in zip(keys, sys.argv[2:]): text=text.replace(k, v)
path.write_text(text)
PY
chmod +x "$RUNNER"
rsync -az "$RUNNER" "$REMOTE:$RDIR/run_remote.sh"
rm -f "$RUNNER"

mkdir -p "$LOCAL_OUT"
cat > "$LOCAL_OUT/remote_info.env" <<EOF
REMOTE='$REMOTE'
RDIR='$RDIR'
LOCAL_OUT='$LOCAL_OUT'
JOB_NAME='$JOB_NAME'
POSITIONS='$POSITIONS'
REPEATS='$REPEATS'
INFERENCE_BATCHES='$INFERENCE_BATCHES'
PUCT_VISITS='$PUCT_VISITS'
PUCT_BATCHES='$PUCT_BATCHES'
PUCT_POSITIONS='$PUCT_POSITIONS'
PUCT_REPEATS='$PUCT_REPEATS'
RUN_TACTICAL_SWEEP='$RUN_TACTICAL_SWEEP'
EOF
if [[ "$DETACH" == "1" ]]; then
  ssh "$REMOTE" "cd '$RDIR' && nohup ./run_remote.sh > run.log 2>&1 < /dev/null & echo \$! > pid"
  log "launched remote detached; pid=$(ssh "$REMOTE" "cat '$RDIR/pid'" 2>/dev/null || true)"
  log "pull later: ACTION=pull RDIR='$RDIR' LOCAL_OUT='$LOCAL_OUT' $0"
else
  ssh "$REMOTE" "cd '$RDIR' && ./run_remote.sh" | tee "$LOCAL_OUT/remote_run.log"
  pull_results
fi
