#!/usr/bin/env bash
set -Eeuo pipefail

# Offload a low-budget Protein/CARBS-style aux-PUCT grid to the Mac mini.
# Produces an arena JSON, JSONL sweep ledger, and Pareto report.
# Tune env vars: WEIGHTS="0.0025 0.005 0.01", VISITS=32, GAMES_PER_PAIR=4, MODES="av rank regret all".

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"

REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
JOB_NAME=${JOB_NAME:-mac_mini_puct_sweep_$STAMP}
RDIR=${RDIR:-/Users/minime/tiny_leela_offload_puct_sweep_$STAMP}
LOCAL_OUT=${LOCAL_OUT:-artifacts/remote_offload/$JOB_NAME}

MODEL=${MODEL:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx}
META=${META:-artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json}
JUDGE_MODEL=${JUDGE_MODEL:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.onnx}
JUDGE_META=${JUDGE_META:-artifacts/arena_10m_guarded/80x5_hybrid_e12_ema.meta.json}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}

VISITS=${VISITS:-32}
BATCH_SIZE=${BATCH_SIZE:-16}
GAMES_PER_PAIR=${GAMES_PER_PAIR:-4}
CPUCT=${CPUCT:-1.5}
MAX_PLIES=${MAX_PLIES:-80}
MAX_OPENINGS=${MAX_OPENINGS:-64}
ADJUDICATE_THRESHOLD=${ADJUDICATE_THRESHOLD:-0.03}
ORT_THREADS=${ORT_THREADS:-6}
EVAL_CACHE_ENTRIES_REMOTE=${EVAL_CACHE_ENTRIES_REMOTE:-100000}
WEIGHTS=${WEIGHTS:-"0.0025 0.005 0.01"}
MODES=${MODES:-"av rank regret all"}
KEEP_REMOTE=${KEEP_REMOTE:-1}
CLEAN_START=${CLEAN_START:-1}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }
label_weight(){ printf '%s' "$1" | sed -e 's/^0\.//' -e 's/\.//g'; }

need "$MODEL"; need "$META"; need "$OPENINGS"
need eval/search_mode_arena.mjs; need eval/pareto_sweep_report.py
need node_modules/onnxruntime-web; need node_modules/onnxruntime-common
if [[ -n "$JUDGE_MODEL" || -n "$JUDGE_META" ]]; then
  need "$JUDGE_MODEL"; need "$JUDGE_META"
fi

PLAYERS="classic:$MODEL:$META:puct"
for w in $WEIGHTS; do
  lab=$(label_weight "$w")
  for mode in $MODES; do
    case "$mode" in
      av) PLAYERS+="",av${lab}:"$MODEL":"$META":aux:"$w":0:0 ;;
      rank) PLAYERS+="",rank${lab}:"$MODEL":"$META":aux:0:"$w":0 ;;
      regret) PLAYERS+="",regret${lab}:"$MODEL":"$META":aux:0:0:"$w" ;;
      all) PLAYERS+="",all${lab}:"$MODEL":"$META":aux:"$w":"$w":"$w" ;;
      *) echo "unknown mode in MODES: $mode" >&2; exit 2 ;;
    esac
  done
done

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

REMOTE_BASE="artifacts/remote_offload/$JOB_NAME"
REMOTE_ARENA="$REMOTE_BASE/arena.json"
REMOTE_LEDGER="$REMOTE_BASE/ledger.jsonl"
REMOTE_PARETO="$REMOTE_BASE/pareto.txt"
log "run remote aux-PUCT sweep: visits=$VISITS games/pair=$GAMES_PER_PAIR weights=[$WEIGHTS] modes=[$MODES]"
JUDGE_ARGS=""
if [[ -n "$JUDGE_MODEL" && -n "$JUDGE_META" ]]; then
  JUDGE_ARGS="--judge-model '$JUDGE_MODEL' --judge-meta '$JUDGE_META' --adjudicate-threshold '$ADJUDICATE_THRESHOLD'"
fi
ssh "$REMOTE" "cd '$RDIR' && \
  STARTED=\$(date +%s) && \
  ORT_NUM_THREADS='$ORT_THREADS' ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS' EVAL_CACHE_ENTRIES='$EVAL_CACHE_ENTRIES_REMOTE' \
  node --experimental-strip-types eval/search_mode_arena.mjs \
    --players='$PLAYERS' \
    --anchor-player classic \
    --games-per-pair '$GAMES_PER_PAIR' \
    --visits '$VISITS' \
    --batch-size '$BATCH_SIZE' \
    --cpuct '$CPUCT' \
    --max-plies '$MAX_PLIES' \
    --openings-file '$OPENINGS' \
    --max-openings '$MAX_OPENINGS' \
    --out '$REMOTE_ARENA' \
    $JUDGE_ARGS && \
  python3 - <<'PY' '$REMOTE_ARENA' '$REMOTE_LEDGER' \"\$STARTED\" '$VISITS' '$GAMES_PER_PAIR'
import json, sys, time
arena_path, ledger_path, started, visits, gpp = sys.argv[1:]
started = int(started)
wall = max(1, int(time.time()) - started)
data = json.load(open(arena_path))
protocol = data.get('protocol', {})
players = {p.get('name'): p for p in protocol.get('players', [])}
with open(ledger_path, 'w') as f:
    for pair in data.get('pairs', []):
        if pair.get('a') == 'classic':
            name = pair.get('b'); score = pair.get('bScore', pair.get('games', 0) - pair.get('aScore', 0))
        elif pair.get('b') == 'classic':
            name = pair.get('a'); score = pair.get('aScore', 0)
        else:
            continue
        games = max(1, int(pair.get('games', 0)))
        p = players.get(name, {})
        row = {
            'trial_id': name,
            'status': 'succeeded',
            'score': float(score) / games,
            'cost': {'wall_seconds': wall, 'cpu_hours': wall * int(protocol.get('ortThreads') or 1) / 3600.0, 'games': games, 'visits': int(visits)},
            'params': {'mode': p.get('mode'), 'av': p.get('avWeight', 0), 'rank': p.get('rankWeight', 0), 'regret': p.get('regretWeight', 0), 'visits': int(visits), 'games_per_pair': int(gpp)},
            'arena': arena_path,
            'games': games,
            'raw_score': score,
        }
        f.write(json.dumps(row, sort_keys=True) + '\n')
PY
  python3 eval/pareto_sweep_report.py '$REMOTE_LEDGER' --top 20 > '$REMOTE_PARETO'"

mkdir -p "$LOCAL_OUT"
rsync -az "$REMOTE:$RDIR/artifacts/remote_offload/$JOB_NAME/" "$LOCAL_OUT/"
log "retrieved results to $LOCAL_OUT"
if [[ -f "$LOCAL_OUT/pareto.txt" ]]; then
  sed -n '1,80p' "$LOCAL_OUT/pareto.txt"
fi

if [[ "$KEEP_REMOTE" != "1" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
  log "removed remote workdir $RDIR"
else
  log "kept remote workdir $RDIR"
fi
