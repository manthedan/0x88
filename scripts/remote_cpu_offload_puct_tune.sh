#!/usr/bin/env bash
set -Eeuo pipefail

# Offload a bounded Bayesian aux-PUCT tuning job to the Mac mini.
# Defaults target CNN96 100M e08 and compare aux-PUCT candidates against classic PUCT.
# Override with env vars, e.g. VISITS=64 ITERATIONS=24 GAMES_PER_CANDIDATE=8 ./scripts/remote_cpu_offload_puct_tune.sh

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"

REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
JOB_NAME=${JOB_NAME:-mac_mini_puct_bayes_$STAMP}
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_puct_bayes_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/$JOB_NAME}

MODEL=${MODEL:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx}
META=${META:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json}
JUDGE_MODEL=${JUDGE_MODEL:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx}
JUDGE_META=${JUDGE_META:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.meta.json}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}

VISITS=${VISITS:-32}
BATCH_SIZE=${BATCH_SIZE:-16}
GAMES_PER_CANDIDATE=${GAMES_PER_CANDIDATE:-6}
ITERATIONS=${ITERATIONS:-12}
CPUCT=${CPUCT:-1.5}
MAX_PLIES=${MAX_PLIES:-80}
MAX_WEIGHT=${MAX_WEIGHT:-0.02}
POOL_SIZE=${POOL_SIZE:-256}
SEED=${SEED:-2309}
ORT_THREADS=${ORT_THREADS:-6}
EVAL_CACHE_ENTRIES_REMOTE=${EVAL_CACHE_ENTRIES_REMOTE:-100000}
DIMS=${DIMS:-av,rank,regret}
BETA=${BETA:-0.9}
LENGTH_SCALE=${LENGTH_SCALE:-0.35}
MAX_OPENINGS=${MAX_OPENINGS:-64}
ADJUDICATE_THRESHOLD=${ADJUDICATE_THRESHOLD:-0.03}
KEEP_REMOTE=${KEEP_REMOTE:-1}
CLEAN_START=${CLEAN_START:-1}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }

need "$MODEL"; need "$META"; need "$OPENINGS"
need eval/bayesian_aux_puct_tune.mjs; need eval/search_mode_arena.mjs
need node_modules/onnxruntime-web; need node_modules/onnxruntime-common
if [[ -n "$JUDGE_MODEL" || -n "$JUDGE_META" ]]; then
  need "$JUDGE_MODEL"; need "$JUDGE_META"
fi

sync_file(){
  local p="$1"
  [[ -n "$p" ]] || return 0
  [[ -e "$p" ]] || return 0
  ssh "$REMOTE" "mkdir -p '$RDIR/$(dirname "$p")'"
  rsync -az "$p" "$REMOTE:$RDIR/$(dirname "$p")/"
}

log "remote=$REMOTE rdir=$RDIR local_out=$LOCAL_OUT"
if [[ "$CLEAN_START" == "1" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
fi
ssh "$REMOTE" "mkdir -p '$RDIR'/node_modules '$RDIR/artifacts/remote_offload/$JOB_NAME'"

log "sync source + ONNX runtime subset"
rsync -az src eval scripts package.json "$REMOTE:$RDIR/"
rsync -az node_modules/onnxruntime-web node_modules/onnxruntime-common "$REMOTE:$RDIR/node_modules/"
sync_file "$MODEL"
sync_file "$META"
sync_file "$JUDGE_MODEL"
sync_file "$JUDGE_META"
sync_file "$OPENINGS"

REMOTE_OUT="artifacts/remote_offload/$JOB_NAME/bayes"
log "run remote Bayesian aux-PUCT tune: visits=$VISITS games/candidate=$GAMES_PER_CANDIDATE iterations=$ITERATIONS dims=$DIMS"
JUDGE_ARGS=""
if [[ -n "$JUDGE_MODEL" && -n "$JUDGE_META" ]]; then
  JUDGE_ARGS="--judge-model '$JUDGE_MODEL' --judge-meta '$JUDGE_META' --adjudicate-threshold '$ADJUDICATE_THRESHOLD'"
fi
ssh "$REMOTE" "cd '$RDIR' && \
  ORT_NUM_THREADS='$ORT_THREADS' ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS' EVAL_CACHE_ENTRIES='$EVAL_CACHE_ENTRIES_REMOTE' \
  node --experimental-strip-types eval/bayesian_aux_puct_tune.mjs \
    --model '$MODEL' \
    --meta '$META' \
    --out-dir '$REMOTE_OUT' \
    --visits '$VISITS' \
    --batch-size '$BATCH_SIZE' \
    --games-per-candidate '$GAMES_PER_CANDIDATE' \
    --iterations '$ITERATIONS' \
    --cpuct '$CPUCT' \
    --max-plies '$MAX_PLIES' \
    --max-weight '$MAX_WEIGHT' \
    --pool-size '$POOL_SIZE' \
    --seed '$SEED' \
    --ort-threads '$ORT_THREADS' \
    --openings-file '$OPENINGS' \
    --max-openings '$MAX_OPENINGS' \
    --dims '$DIMS' \
    --beta '$BETA' \
    --length-scale '$LENGTH_SCALE' \
    $JUDGE_ARGS"

mkdir -p "$LOCAL_OUT"
rsync -az "$REMOTE:$RDIR/artifacts/remote_offload/$JOB_NAME/" "$LOCAL_OUT/"
log "retrieved results to $LOCAL_OUT"
if [[ -f "$LOCAL_OUT/bayes/summary.tsv" ]]; then
  tail -20 "$LOCAL_OUT/bayes/summary.tsv"
fi

if [[ "$KEEP_REMOTE" != "1" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
  log "removed remote workdir $RDIR"
else
  log "kept remote workdir $RDIR"
fi
