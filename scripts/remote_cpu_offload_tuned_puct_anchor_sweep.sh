#!/usr/bin/env bash
set -Eeuo pipefail

# Sweep best per-visit Bayesian aux-PUCT candidates against UCI anchors.
# Mac default supports stockfish-js Elo anchors and stockfish-js full-single node anchors.
# Maia requires lc0+weights on the remote; provide MAIA_UCI_ANCHORS and INCLUDE_MAIA=1 when available,
# or run the same candidate specs locally with eval/uci_anchor_arena.mjs against .local_engines/maia/*.sh.

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"

ACTION=${ACTION:-launch}   # launch | pull | clean | candidates
REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -z "${JOB_NAME:-}" && -n "${LOCAL_OUT:-}" ]]; then
  JOB_NAME=$(basename "$LOCAL_OUT")
else
  JOB_NAME=${JOB_NAME:-mac_mini_cnn96_tuned_puct_anchor_sweep_$STAMP}
fi
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_tuned_puct_anchor_sweep_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/$JOB_NAME}
DETACH=${DETACH:-1}
CLEAN_START=${CLEAN_START:-1}
KEEP_REMOTE=${KEEP_REMOTE:-1}

BAYES_LOCAL=${BAYES_LOCAL:-}
CANDIDATES_TSV=${CANDIDATES_TSV:-}
MODEL=${MODEL:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx}
META=${META:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}

PAIRS=${PAIRS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
CPUCT=${CPUCT:-1.5}
MAX_PLIES=${MAX_PLIES:-100}
OPENING_COUNT=${OPENING_COUNT:-64}
ORT_THREADS=${ORT_THREADS:-6}
EVAL_CACHE_ENTRIES_REMOTE=${EVAL_CACHE_ENTRIES_REMOTE:-150000}
HASH=${HASH:-16}
THREADS=${THREADS:-1}

# Elo-limited stockfish-js anchors.
STOCKFISH_ELO_VARIANT=${STOCKFISH_ELO_VARIANT:-lite-single}
STOCKFISH_ELO_LEVELS=${STOCKFISH_ELO_LEVELS:-1320,1600,1800}
STOCKFISH_ELO_NODES=${STOCKFISH_ELO_NODES:-64}

# Full, non-Elo-limited stockfish-js anchors at lower node budgets.
FULL_STOCKFISH_VARIANT=${FULL_STOCKFISH_VARIANT:-full-single}
FULL_STOCKFISH_NODES=${FULL_STOCKFISH_NODES:-"8 16 32"}

INCLUDE_MAIA=${INCLUDE_MAIA:-0}
MAIA_UCI_ANCHORS=${MAIA_UCI_ANCHORS:-}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }

make_candidates(){
  local out="$1"
  [[ -n "$BAYES_LOCAL" ]] || { echo "BAYES_LOCAL is required unless CANDIDATES_TSV is supplied" >&2; exit 2; }
  [[ -d "$BAYES_LOCAL" ]] || { echo "BAYES_LOCAL not found: $BAYES_LOCAL" >&2; exit 2; }
  python3 - <<'PY' "$BAYES_LOCAL" "$out" "$MODEL" "$META"
import glob, json, pathlib, sys
base, out, model, meta = sys.argv[1:]
rows = []
for state_path in sorted(glob.glob(str(pathlib.Path(base) / 'bayes_v*' / 'state.json'))):
    state = json.load(open(state_path))
    visit = state.get('protocol', {}).get('visits')
    b = state.get('best') or {}
    w = b.get('weights') or {}
    if visit is None or not b:
        continue
    label = f"cnn96_v{visit}_bayes_aux"
    spec = f"{label}:{model}:{meta}:aux:{w.get('av',0)}:{w.get('rank',0)}:{w.get('regret',0)}:{w.get('risk',0)}:{w.get('uncertainty',0)}"
    rows.append((int(visit), label, spec, b.get('scoreRate',0), b.get('games',0), w))
if not rows:
    raise SystemExit(f'no bayes_v*/state.json candidates found under {base}')
rows.sort(key=lambda r: r[0])
with open(out, 'w') as f:
    f.write('visits\tlabel\tcandidate_spec\ttune_score_rate\ttune_games\tav\trank\tregret\trisk\tuncertainty\n')
    for visit, label, spec, sr, games, w in rows:
        f.write('\t'.join(map(str, [visit, label, spec, f'{float(sr):.6f}', games, w.get('av',0), w.get('rank',0), w.get('regret',0), w.get('risk',0), w.get('uncertainty',0)])) + '\n')
print(out)
PY
}

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
  [[ -f "$LOCAL_OUT/anchor_summary.tsv" ]] && tail -40 "$LOCAL_OUT/anchor_summary.tsv" || true
}

if [[ "$ACTION" == "pull" ]]; then pull_results; exit 0; fi
if [[ "$ACTION" == "clean" ]]; then ssh "$REMOTE" "rm -rf '$RDIR'"; log "removed remote workdir $RDIR"; exit 0; fi
if [[ "$ACTION" == "candidates" ]]; then
  mkdir -p "$LOCAL_OUT"
  make_candidates "$LOCAL_OUT/candidates.tsv"
  exit 0
fi
if [[ "$ACTION" != "launch" ]]; then echo "bad ACTION=$ACTION" >&2; exit 2; fi

need "$MODEL"; need "$META"; need "$OPENINGS"
need eval/uci_anchor_arena.mjs; need scripts/uci_stockfish_js_wrapper.mjs
need node_modules/onnxruntime-web; need node_modules/onnxruntime-common; need node_modules/stockfish

mkdir -p "$LOCAL_OUT"
if [[ -n "$CANDIDATES_TSV" ]]; then
  need "$CANDIDATES_TSV"
  cp "$CANDIDATES_TSV" "$LOCAL_OUT/candidates.tsv"
else
  make_candidates "$LOCAL_OUT/candidates.tsv"
fi
CANDIDATES_TSV="$LOCAL_OUT/candidates.tsv"

sync_file(){
  local p="$1"
  [[ -n "$p" ]] || return 0
  [[ -e "$p" ]] || return 0
  ssh "$REMOTE" "mkdir -p '$RDIR/$(dirname "$p")'"
  rsync -az "$p" "$REMOTE:$RDIR/$(dirname "$p")/"
}

log "remote=$REMOTE rdir=$RDIR local_out=$LOCAL_OUT detach=$DETACH"
if [[ "$CLEAN_START" == "1" ]]; then ssh "$REMOTE" "rm -rf '$RDIR'"; fi
ssh "$REMOTE" "mkdir -p '$RDIR'/node_modules '$RDIR/artifacts/remote_offload/$JOB_NAME' '$RDIR/remote_engines'"

log "sync source + ONNX runtime + stockfish-js + models"
rsync -az src eval scripts package.json "$REMOTE:$RDIR/"
rsync -az node_modules/onnxruntime-web node_modules/onnxruntime-common node_modules/stockfish "$REMOTE:$RDIR/node_modules/"
sync_file "$MODEL"; sync_file "$META"; sync_file "$OPENINGS"
rsync -az "$CANDIDATES_TSV" "$REMOTE:$RDIR/candidates.tsv"

# Generate wrapper commands with no spaces so uci_anchor_arena can spawn them directly.
ssh "$REMOTE" "cat > '$RDIR/remote_engines/stockfish_elo.sh' <<'EOF'
#!/usr/bin/env bash
cd '$RDIR'
exec node scripts/uci_stockfish_js_wrapper.mjs '$STOCKFISH_ELO_VARIANT'
EOF
cat > '$RDIR/remote_engines/stockfish_full.sh' <<'EOF'
#!/usr/bin/env bash
cd '$RDIR'
exec node scripts/uci_stockfish_js_wrapper.mjs '$FULL_STOCKFISH_VARIANT'
EOF
chmod +x '$RDIR/remote_engines/stockfish_elo.sh' '$RDIR/remote_engines/stockfish_full.sh'"

CUSTOM_ANCHORS=""
for n in $FULL_STOCKFISH_NODES; do
  entry="stockfish_full_nodes${n}|$RDIR/remote_engines/stockfish_full.sh|nodes=${n}"
  if [[ -z "$CUSTOM_ANCHORS" ]]; then CUSTOM_ANCHORS="$entry"; else CUSTOM_ANCHORS+=",$entry"; fi
done
if [[ "$INCLUDE_MAIA" == "1" ]]; then
  if [[ -z "$MAIA_UCI_ANCHORS" ]]; then
    echo "INCLUDE_MAIA=1 requires MAIA_UCI_ANCHORS='maia1100|/path/lc0-wrapper|nodes=32,...' on the remote" >&2
    exit 2
  fi
  CUSTOM_ANCHORS+=",$MAIA_UCI_ANCHORS"
fi

RUNNER=$(mktemp)
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd '$RDIR'
BASE='artifacts/remote_offload/$JOB_NAME'
STATUS='status.txt'
SUMMARY="\$BASE/anchor_summary.tsv"
mkdir -p "\$BASE"
mark(){ printf '%s %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$*" | tee -a "\$STATUS"; }
trap 'rc=\$?; mark "FAILED rc=\$rc phase=\${PHASE:-unknown}"; echo failed > "\$BASE/status.final"; exit \$rc' ERR
: > "\$STATUS"
printf 'visits\tlabel\tanchor\tgames\twins\tdraws\tlosses\tscore_rate\telo_diff\tillegal\tarena_json\n' > "\$SUMMARY"
export ORT_NUM_THREADS='$ORT_THREADS'
export ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS'
export EVAL_CACHE_ENTRIES='$EVAL_CACHE_ENTRIES_REMOTE'
mark 'START tuned PUCT anchor sweep job=$JOB_NAME pairs=$PAIRS stockfish_elo_levels=$STOCKFISH_ELO_LEVELS full_nodes=$FULL_STOCKFISH_NODES include_maia=$INCLUDE_MAIA'
tail -n +2 candidates.tsv | while IFS=\$'\t' read -r VISITS LABEL CANDIDATE_SPEC TUNE_SCORE TUNE_GAMES AV RANK REGRET RISK UNCERTAINTY; do
  PHASE="anchor_\${LABEL}"
  OUT="\$BASE/\${LABEL}_anchors.json"
  mark "ANCHOR label=\$LABEL visits=\$VISITS tune_score=\$TUNE_SCORE"
  node --experimental-strip-types eval/uci_anchor_arena.mjs \
    --candidate="\$CANDIDATE_SPEC" \
    --openings-file '$OPENINGS' \
    --opening-count '$OPENING_COUNT' \
    --pairs '$PAIRS' \
    --visits "\$VISITS" \
    --cpuct '$CPUCT' \
    --batch-size '$BATCH_SIZE' \
    --max-plies '$MAX_PLIES' \
    --stockfish '$RDIR/remote_engines/stockfish_elo.sh' \
    --stockfish-levels '$STOCKFISH_ELO_LEVELS' \
    --stockfish-nodes '$STOCKFISH_ELO_NODES' \
    --threads '$THREADS' \
    --hash '$HASH' \
    --uci-anchors='$CUSTOM_ANCHORS' \
    --out "\$OUT" > "\$BASE/\${LABEL}_anchors.log" 2>&1
  python3 - <<'PY' "\$OUT" "\$SUMMARY" "\$VISITS" "\$LABEL"
import json, sys
path, summary, visits, label = sys.argv[1:]
data = json.load(open(path))
with open(summary, 'a') as f:
    for s in data.get('summaries', []):
        f.write('\t'.join(map(str, [visits, label, s.get('anchor'), s.get('games'), s.get('wins'), s.get('draws'), s.get('losses'), f"{float(s.get('scoreRate',0)):.6f}", f"{float(s.get('eloDiff',0)):.2f}", s.get('illegal'), path])) + '\n')
PY
  mark "DONE label=\$LABEL"
done
mark 'DONE tuned PUCT anchor sweep'
echo succeeded > "\$BASE/status.final"
EOF
chmod +x "$RUNNER"
rsync -az "$RUNNER" "$REMOTE:$RDIR/run_remote.sh"
rm -f "$RUNNER"

cat > "$LOCAL_OUT/remote_info.env" <<EOF
REMOTE='$REMOTE'
RDIR='$RDIR'
LOCAL_OUT='$LOCAL_OUT'
JOB_NAME='$JOB_NAME'
BAYES_LOCAL='$BAYES_LOCAL'
PAIRS='$PAIRS'
STOCKFISH_ELO_LEVELS='$STOCKFISH_ELO_LEVELS'
STOCKFISH_ELO_NODES='$STOCKFISH_ELO_NODES'
FULL_STOCKFISH_NODES='$FULL_STOCKFISH_NODES'
INCLUDE_MAIA='$INCLUDE_MAIA'
EOF

if [[ "$DETACH" == "1" ]]; then
  log "launch remote detached job"
  ssh "$REMOTE" "cd '$RDIR' && nohup ./run_remote.sh > run.log 2>&1 < /dev/null & echo \$! > pid && echo PID=\$(cat pid)"
  log "launched. Info: $LOCAL_OUT/remote_info.env"
  log "pull with: ACTION=pull RDIR='$RDIR' LOCAL_OUT='$LOCAL_OUT' ./scripts/remote_cpu_offload_tuned_puct_anchor_sweep.sh"
else
  log "run remote foreground job"
  ssh "$REMOTE" "cd '$RDIR' && ./run_remote.sh 2>&1 | tee run.log"
  pull_results
  if [[ "$KEEP_REMOTE" != "1" ]]; then ssh "$REMOTE" "rm -rf '$RDIR'"; log "removed remote workdir $RDIR"; fi
fi
