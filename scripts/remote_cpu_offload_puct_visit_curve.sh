#!/usr/bin/env bash
set -Eeuo pipefail

# Offload a CNN96 classic-PUCT audit + visit-curve sweep to the Mac mini.
# Default visits: 32 64 128 256 512 1024.
# Default behavior is detached; use ACTION=pull with the printed RDIR/LOCAL_OUT to retrieve partial/final results.

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"

ACTION=${ACTION:-launch}   # launch | pull | clean
REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -z "${JOB_NAME:-}" && -n "${LOCAL_OUT:-}" ]]; then
  JOB_NAME=$(basename "$LOCAL_OUT")
else
  JOB_NAME=${JOB_NAME:-mac_mini_cnn96_puct_visit_curve_$STAMP}
fi
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_puct_visit_curve_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/$JOB_NAME}
DETACH=${DETACH:-1}
CLEAN_START=${CLEAN_START:-1}
KEEP_REMOTE=${KEEP_REMOTE:-1}

MODEL=${MODEL:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx}
META=${META:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json}
ANCHOR_MODEL=${ANCHOR_MODEL:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx}
ANCHOR_META=${ANCHOR_META:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.meta.json}
JUDGE_MODEL=${JUDGE_MODEL:-$ANCHOR_MODEL}
JUDGE_META=${JUDGE_META:-$ANCHOR_META}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}

VISIT_STEPS=${VISIT_STEPS:-"32 64 128 256 512 1024"}
VISITS_CSV=${VISITS_CSV:-$(printf '%s' "$VISIT_STEPS" | tr ' ' ',')}
GAMES_PER_PAIR=${GAMES_PER_PAIR:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
CPUCT=${CPUCT:-1.5}
MAX_PLIES=${MAX_PLIES:-80}
MAX_OPENINGS=${MAX_OPENINGS:-64}
AUDIT_LIMIT=${AUDIT_LIMIT:-64}
BENCH_REPEATS=${BENCH_REPEATS:-2}
BENCH_POSITIONS=${BENCH_POSITIONS:-2}
BENCH_BATCHES=${BENCH_BATCHES:-1,8,16,32}
ORT_THREADS=${ORT_THREADS:-6}
EVAL_CACHE_ENTRIES_REMOTE=${EVAL_CACHE_ENTRIES_REMOTE:-150000}
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
  [[ -f "$LOCAL_OUT/summary.tsv" ]] && tail -20 "$LOCAL_OUT/summary.tsv" || true
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

need "$MODEL"; need "$META"; need "$ANCHOR_MODEL"; need "$ANCHOR_META"; need "$JUDGE_MODEL"; need "$JUDGE_META"; need "$OPENINGS"
need eval/search_mode_arena.mjs; need eval/puct_core_tests.mjs; need eval/puct_root_prior_parity.mjs; need eval/puct_consistency_check.mjs; need eval/puct_batch_benchmark.mjs; need eval/pareto_sweep_report.py
need node_modules/onnxruntime-web; need node_modules/onnxruntime-common

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
sync_file "$ANCHOR_MODEL"
sync_file "$ANCHOR_META"
sync_file "$JUDGE_MODEL"
sync_file "$JUDGE_META"
sync_file "$OPENINGS"

RUNNER=$(mktemp)
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd '$RDIR'
BASE='artifacts/remote_offload/$JOB_NAME'
STATUS='status.txt'
SUMMARY="\$BASE/summary.tsv"
LEDGER="\$BASE/ledger.jsonl"
POSITIONS="\$BASE/puct_audit_positions.json"
mkdir -p "\$BASE"
mark(){ printf '%s %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$*" | tee -a "\$STATUS"; }
trap 'rc=\$?; mark "FAILED rc=\$rc phase=\${PHASE:-unknown}"; echo failed > "\$BASE/status.final"; exit \$rc' ERR
: > "\$STATUS"
: > "\$LEDGER"
printf 'visits\\tcnn96_score_rate\\tcnn96_score\\tgames\\telapsed_s\\tarena_json\\n' > "\$SUMMARY"
export ORT_NUM_THREADS='$ORT_THREADS'
export ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS'
export EVAL_CACHE_ENTRIES='$EVAL_CACHE_ENTRIES_REMOTE'
mark 'START CNN96 PUCT audit + visit curve job=$JOB_NAME visits=$VISIT_STEPS games_per_pair=$GAMES_PER_PAIR'
PHASE=positions
python3 - <<'PY' '$OPENINGS' "\$POSITIONS"
import json, sys
src, out = sys.argv[1:]
fens = []
for line in open(src):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if ' #' in line:
        line = line.split(' #', 1)[0].strip()
    fens.append({'fen': line})
json.dump({'positions': fens}, open(out, 'w'), indent=2)
print(f'wrote {len(fens)} positions to {out}')
PY
PHASE=puct_core
mark 'AUDIT puct_core_tests'
node --experimental-strip-types eval/puct_core_tests.mjs > "\$BASE/puct_core_tests.log" 2>&1
PHASE=root_prior
mark 'AUDIT root_prior_parity limit=$AUDIT_LIMIT'
node --experimental-strip-types eval/puct_root_prior_parity.mjs \
  --model '$MODEL' --meta '$META' --positions-json "\$POSITIONS" --limit '$AUDIT_LIMIT' --topk 8 \
  > "\$BASE/root_prior_parity.log" 2>&1
PHASE=consistency
mark 'AUDIT puct_consistency visits=$VISITS_CSV limit=$AUDIT_LIMIT'
node --experimental-strip-types eval/puct_consistency_check.mjs \
  --model '$MODEL' --meta '$META' --positions-json "\$POSITIONS" --limit '$AUDIT_LIMIT' --visits '$VISITS_CSV' \
  --out "\$BASE/puct_consistency.json" > "\$BASE/puct_consistency.log" 2>&1
PHASE=batch_benchmark
mark 'AUDIT puct_batch_benchmark visits=$VISITS_CSV batches=$BENCH_BATCHES'
node --experimental-strip-types eval/puct_batch_benchmark.mjs \
  --model '$MODEL' --meta '$META' --visits '$VISITS_CSV' --batches '$BENCH_BATCHES' \
  --batch-size '$BATCH_SIZE' --cpuct '$CPUCT' --repeats '$BENCH_REPEATS' --positions '$BENCH_POSITIONS' \
  > "\$BASE/puct_batch_benchmark.log" 2>&1
PHASE=visit_curve
for V in $VISIT_STEPS; do
  mark "SWEEP visits=\$V"
  STARTED=\$(date +%s)
  ARENA="\$BASE/visit_\${V}.json"
  node --experimental-strip-types eval/search_mode_arena.mjs \
    --players='cnn96:$MODEL:$META:puct,champ80:$ANCHOR_MODEL:$ANCHOR_META:puct' \
    --games-per-pair '$GAMES_PER_PAIR' \
    --visits "\$V" \
    --batch-size '$BATCH_SIZE' \
    --cpuct '$CPUCT' \
    --max-plies '$MAX_PLIES' \
    --openings-file '$OPENINGS' \
    --max-openings '$MAX_OPENINGS' \
    --judge-model '$JUDGE_MODEL' \
    --judge-meta '$JUDGE_META' \
    --adjudicate-threshold '$ADJUDICATE_THRESHOLD' \
    --out "\$ARENA" > "\$BASE/visit_\${V}.log" 2>&1
  ELAPSED=\$((\$(date +%s) - STARTED))
  python3 - <<'PY' "\$ARENA" "\$LEDGER" "\$SUMMARY" "\$V" "\$ELAPSED"
import json, sys
arena_path, ledger_path, summary_path, visits, elapsed = sys.argv[1:]
data = json.load(open(arena_path))
pair = data['pairs'][0]
if pair['a'] == 'cnn96':
    score = pair['aScore']
elif pair['b'] == 'cnn96':
    score = pair['games'] - pair['aScore']
else:
    raise SystemExit('cnn96 missing from pair')
games = max(1, int(pair['games']))
rate = score / games
row = {
    'trial_id': f'cnn96_puct_v{visits}',
    'status': 'succeeded',
    'score': rate,
    'cost': {'wall_seconds': int(elapsed), 'visits': int(visits), 'games': games, 'objective': int(visits) * games},
    'params': {'model': 'cnn96_e08_100m', 'mode': 'puct', 'visits': int(visits), 'cpuct': data.get('protocol', {}).get('cpuct')},
    'arena': arena_path,
    'games': games,
    'raw_score': score,
}
with open(ledger_path, 'a') as f:
    f.write(json.dumps(row, sort_keys=True) + '\n')
with open(summary_path, 'a') as f:
    f.write(f"{visits}\t{rate:.6f}\t{score}\t{games}\t{elapsed}\t{arena_path}\n")
print(json.dumps(row, sort_keys=True))
PY
  mark "DONE visits=\$V elapsed_s=\$ELAPSED"
done
PHASE=pareto
python3 eval/pareto_sweep_report.py "\$LEDGER" --top 20 > "\$BASE/pareto.txt"
mark 'DONE CNN96 PUCT audit + visit curve'
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
GAMES_PER_PAIR='$GAMES_PER_PAIR'
MODEL='$MODEL'
META='$META'
ANCHOR_MODEL='$ANCHOR_MODEL'
ANCHOR_META='$ANCHOR_META'
EOF

if [[ "$DETACH" == "1" ]]; then
  log "launch remote detached job"
  ssh "$REMOTE" "cd '$RDIR' && nohup ./run_remote.sh > run.log 2>&1 < /dev/null & echo \$! > pid && echo PID=\$(cat pid)"
  log "launched. Info: $LOCAL_OUT/remote_info.env"
  log "pull with: ACTION=pull RDIR='$RDIR' LOCAL_OUT='$LOCAL_OUT' ./scripts/remote_cpu_offload_puct_visit_curve.sh"
else
  log "run remote foreground job"
  ssh "$REMOTE" "cd '$RDIR' && ./run_remote.sh 2>&1 | tee run.log"
  pull_results
  if [[ "$KEEP_REMOTE" != "1" ]]; then
    ssh "$REMOTE" "rm -rf '$RDIR'"
    log "removed remote workdir $RDIR"
  fi
fi
