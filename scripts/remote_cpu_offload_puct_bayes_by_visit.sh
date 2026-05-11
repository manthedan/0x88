#!/usr/bin/env bash
set -Eeuo pipefail

# Offload Bayesian aux-PUCT tuning for multiple visit counts to the Mac mini.
# Default behavior is detached. Use ACTION=pull with the printed RDIR/LOCAL_OUT.
# This tunes aux-PUCT weights (av/rank/regret by default) at each VISIT_STEPS value.

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"

ACTION=${ACTION:-launch}   # launch | pull | clean
REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -z "${JOB_NAME:-}" && -n "${LOCAL_OUT:-}" ]]; then
  JOB_NAME=$(basename "$LOCAL_OUT")
else
  JOB_NAME=${JOB_NAME:-mac_mini_cnn96_puct_bayes_by_visit_$STAMP}
fi
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_puct_bayes_by_visit_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/$JOB_NAME}
DETACH=${DETACH:-1}
CLEAN_START=${CLEAN_START:-1}
KEEP_REMOTE=${KEEP_REMOTE:-1}

MODEL=${MODEL:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx}
META=${META:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json}
JUDGE_MODEL=${JUDGE_MODEL:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx}
JUDGE_META=${JUDGE_META:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.meta.json}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}

VISIT_STEPS=${VISIT_STEPS:-"32 64 128 256 512 1024"}
BATCH_SIZE=${BATCH_SIZE:-16}
GAMES_PER_CANDIDATE=${GAMES_PER_CANDIDATE:-4}
ITERATIONS=${ITERATIONS:-20}
INITIAL_BUDGET=${INITIAL_BUDGET:-8}
CPUCT=${CPUCT:-1.5}
FPU=${FPU:-0}
CPUCT_VALUES=${CPUCT_VALUES:-1.2,1.5,1.8}
FPU_VALUES=${FPU_VALUES:--0.1,0,0.1}
MAX_PLIES=${MAX_PLIES:-80}
MAX_WEIGHT=${MAX_WEIGHT:-0.02}
POOL_SIZE=${POOL_SIZE:-384}
SEED=${SEED:-2309}
ORT_THREADS=${ORT_THREADS:-6}
EVAL_CACHE_ENTRIES_REMOTE=${EVAL_CACHE_ENTRIES_REMOTE:-150000}
DIMS=${DIMS:-av,rank,regret}
BETA=${BETA:-0.9}
LENGTH_SCALE=${LENGTH_SCALE:-0.35}
COST_AWARE=${COST_AWARE:-1}
PRIOR_BEST_TSV=${PRIOR_BEST_TSV:-artifacts/remote_offload/mac_mini_cnn96_puct_bayes_by_visit_20260509T193020Z/best_by_visit.tsv}
CONFIRM_TOP_K=${CONFIRM_TOP_K:-3}
CONFIRM_GAMES=${CONFIRM_GAMES:-24}
CONFIRM_OPENING_OFFSET=${CONFIRM_OPENING_OFFSET:-97}
MAX_OPENINGS=${MAX_OPENINGS:-64}
ADJUDICATE_THRESHOLD=${ADJUDICATE_THRESHOLD:-0.03}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }

pull_results(){
  mkdir -p "$LOCAL_OUT"
  if ssh "$REMOTE" "test -d '$RDIR/artifacts/remote_offload/$JOB_NAME'"; then
    rsync -az "$REMOTE:$RDIR/artifacts/remote_offload/$JOB_NAME/" "$LOCAL_OUT/"
  fi
  if ssh "$REMOTE" "test -f '$RDIR/run.log'"; then
    rsync -az "$REMOTE:$RDIR/run.log" "$LOCAL_OUT/remote_run.log"
  fi
  if ssh "$REMOTE" "test -f '$RDIR/status.txt'"; then
    rsync -az "$REMOTE:$RDIR/status.txt" "$LOCAL_OUT/remote_status.txt"
  fi
  log "pulled results to $LOCAL_OUT"
  [[ -f "$LOCAL_OUT/remote_status.txt" ]] && tail -20 "$LOCAL_OUT/remote_status.txt" || true
  [[ -f "$LOCAL_OUT/best_by_visit.tsv" ]] && column -t -s $'\t' "$LOCAL_OUT/best_by_visit.tsv" | tail -20 || true
}

if [[ "$ACTION" == "pull" ]]; then
  pull_results
  exit 0
fi
if [[ "$ACTION" == "clean" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
  log "removed remote workdir $RDIR"
  exit 0
fi
if [[ "$ACTION" != "launch" ]]; then
  echo "bad ACTION=$ACTION (expected launch|pull|clean)" >&2
  exit 2
fi

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

log "remote=$REMOTE rdir=$RDIR local_out=$LOCAL_OUT detach=$DETACH"
if [[ "$CLEAN_START" == "1" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
fi
ssh "$REMOTE" "mkdir -p '$RDIR'/node_modules '$RDIR/artifacts/remote_offload/$JOB_NAME'"

log "sync source + ONNX runtime subset + models"
rsync -az src eval scripts package.json "$REMOTE:$RDIR/"
rsync -az node_modules/onnxruntime-web node_modules/onnxruntime-common "$REMOTE:$RDIR/node_modules/"
sync_file "$MODEL"
sync_file "$META"
sync_file "$JUDGE_MODEL"
sync_file "$JUDGE_META"
sync_file "$OPENINGS"
sync_file "$PRIOR_BEST_TSV"

RUNNER=$(mktemp)
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd '$RDIR'
BASE='artifacts/remote_offload/$JOB_NAME'
STATUS='status.txt'
BEST="\$BASE/best_by_visit.tsv"
mkdir -p "\$BASE"
mark(){ printf '%s %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$*" | tee -a "\$STATUS"; }
trap 'rc=\$?; mark "FAILED rc=\$rc phase=\${PHASE:-unknown}"; echo failed > "\$BASE/status.final"; exit \$rc' ERR
: > "\$STATUS"
printf 'visits\tbest_name\tscore_rate\tscore\tgames\tav\trank\tregret\trisk\tuncertainty\tcpuct\tfpu\tout_dir\n' > "\$BEST"
export ORT_NUM_THREADS='$ORT_THREADS'
export ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS'
export EVAL_CACHE_ENTRIES='$EVAL_CACHE_ENTRIES_REMOTE'
mark 'START Protein/CARBS Bayesian aux-PUCT by visit job=$JOB_NAME visits=$VISIT_STEPS iterations=$ITERATIONS games_per_candidate=$GAMES_PER_CANDIDATE dims=$DIMS cpuct_values=$CPUCT_VALUES fpu_values=$FPU_VALUES confirm_top_k=$CONFIRM_TOP_K confirm_games=$CONFIRM_GAMES'
for V in $VISIT_STEPS; do
  PHASE="bayes_v\$V"
  OUT="\$BASE/bayes_v\$V"
  mkdir -p "\$OUT"
  mark "TUNE visits=\$V out=\$OUT"
  node --experimental-strip-types eval/bayesian_aux_puct_tune.mjs \
    --model '$MODEL' \
    --meta '$META' \
    --out-dir "\$OUT" \
    --visits "\$V" \
    --batch-size '$BATCH_SIZE' \
    --games-per-candidate '$GAMES_PER_CANDIDATE' \
    --iterations '$ITERATIONS' \
    --cpuct '$CPUCT' \
    --fpu '$FPU' \
    --cpuct-values '$CPUCT_VALUES' \
    --fpu-values '$FPU_VALUES' \
    --max-plies '$MAX_PLIES' \
    --max-weight '$MAX_WEIGHT' \
    --pool-size '$POOL_SIZE' \
    --seed '$SEED' \
    --initial-budget '$INITIAL_BUDGET' \
    --cost-aware '$COST_AWARE' \
    --prior-best-tsv '$PRIOR_BEST_TSV' \
    --confirm-top-k '$CONFIRM_TOP_K' \
    --confirm-games '$CONFIRM_GAMES' \
    --confirm-opening-offset '$CONFIRM_OPENING_OFFSET' \
    --ort-threads '$ORT_THREADS' \
    --openings-file '$OPENINGS' \
    --max-openings '$MAX_OPENINGS' \
    --dims '$DIMS' \
    --beta '$BETA' \
    --length-scale '$LENGTH_SCALE' \
    --judge-model '$JUDGE_MODEL' \
    --judge-meta '$JUDGE_META' \
    --adjudicate-threshold '$ADJUDICATE_THRESHOLD' \
    > "\$OUT/run.log" 2>&1
  python3 - <<'PY' "\$OUT" "\$BEST" "\$V"
import json, pathlib, sys
out, best_path, visits = sys.argv[1:]
state = pathlib.Path(out) / 'state.json'
if not state.exists():
    raise SystemExit(f'missing {state}')
data = json.loads(state.read_text())
b = data.get('best') or {}
w = b.get('weights') or {}
with open(best_path, 'a') as f:
    f.write('\t'.join([
        str(visits), str(b.get('name','')), f"{float(b.get('scoreRate',0)):.6f}",
        str(b.get('score','')), str(b.get('games','')),
        str(w.get('av',0)), str(w.get('rank',0)), str(w.get('regret',0)),
        str(w.get('risk',0)), str(w.get('uncertainty',0)),
        str(w.get('cpuct', data.get('protocol',{}).get('baseCpuct', data.get('protocol',{}).get('cpuct', 0)))),
        str(w.get('fpu', data.get('protocol',{}).get('baseFpu', data.get('protocol',{}).get('fpu', 0)))), out,
    ]) + '\n')
PY
  mark "DONE visits=\$V"
done
mark 'DONE Bayesian aux-PUCT by visit'
echo succeeded > "\$BASE/status.final"
EOF
chmod +x "$RUNNER"
rsync -az "$RUNNER" "$REMOTE:$RDIR/run_remote.sh"
rm -f "$RUNNER"

mkdir -p "$LOCAL_OUT"
cat > "$LOCAL_OUT/remote_info.env" <<EOF
REMOTE='$REMOTE'
RDIR='$RDIR'
LOCAL_OUT='$LOCAL_OUT'
JOB_NAME='$JOB_NAME'
VISIT_STEPS='$VISIT_STEPS'
GAMES_PER_CANDIDATE='$GAMES_PER_CANDIDATE'
ITERATIONS='$ITERATIONS'
INITIAL_BUDGET='$INITIAL_BUDGET'
CPUCT='$CPUCT'
FPU='$FPU'
CPUCT_VALUES='$CPUCT_VALUES'
FPU_VALUES='$FPU_VALUES'
COST_AWARE='$COST_AWARE'
CONFIRM_TOP_K='$CONFIRM_TOP_K'
CONFIRM_GAMES='$CONFIRM_GAMES'
PRIOR_BEST_TSV='$PRIOR_BEST_TSV'
MODEL='$MODEL'
META='$META'
EOF

if [[ "$DETACH" == "1" ]]; then
  log "launch remote detached job"
  ssh "$REMOTE" "cd '$RDIR' && nohup ./run_remote.sh > run.log 2>&1 < /dev/null & echo \$! > pid && echo PID=\$(cat pid)"
  log "launched. Info: $LOCAL_OUT/remote_info.env"
  log "pull with: ACTION=pull RDIR='$RDIR' LOCAL_OUT='$LOCAL_OUT' ./scripts/remote_cpu_offload_puct_bayes_by_visit.sh"
else
  log "run remote foreground job"
  ssh "$REMOTE" "cd '$RDIR' && ./run_remote.sh 2>&1 | tee run.log"
  pull_results
  if [[ "$KEEP_REMOTE" != "1" ]]; then
    ssh "$REMOTE" "rm -rf '$RDIR'"
    log "removed remote workdir $RDIR"
  fi
fi
