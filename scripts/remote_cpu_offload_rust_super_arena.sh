#!/usr/bin/env bash
set -Eeuo pipefail

# Launch/pull/status/clean a sharded Rust/native-ORT canonical super arena on the Mac mini.
# This is the maintained version of the one-off 2026-05-11 Rust super-arena offload.
#
# Examples:
#   scripts/remote_cpu_offload_rust_super_arena.sh
#   ACTION=status RDIR=/Users/minime/tiny_leela_mac_mini_super_100m_rust_parallel_... JOB_NAME=... scripts/remote_cpu_offload_rust_super_arena.sh
#   ACTION=pull   RDIR=/Users/minime/tiny_leela_mac_mini_super_100m_rust_parallel_... JOB_NAME=... scripts/remote_cpu_offload_rust_super_arena.sh

ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"

ACTION=${ACTION:-launch}   # launch | pull | status | clean
REMOTE=${REMOTE:-mac-mini}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
JOB_NAME=${JOB_NAME:-mac_mini_super_100m_rust_parallel_$STAMP}
RDIR=${RDIR:-/Users/minime/tiny_leela_$JOB_NAME}
LOCAL_OUT=${LOCAL_OUT:-artifacts/canonical_super_100m_arena_20260511/$JOB_NAME}
WORKERS=${WORKERS:-6}
VISITS=${VISITS:-128}
GAMES_PER_PAIR=${GAMES_PER_PAIR:-2}
MAX_PLIES=${MAX_PLIES:-100}
MAX_OPENINGS=${MAX_OPENINGS:-24}
PROGRESS_EVERY=${PROGRESS_EVERY:-20}
ORT_THREADS=${ORT_THREADS:-1}
ADJUDICATE=${ADJUDICATE:-value}
ADJUDICATE_THRESHOLD=${ADJUDICATE_THRESHOLD:-0.05}
OPENINGS=${OPENINGS:-eval/opening_suite_uho_lite_v1.fen}
DETACH=${DETACH:-1}
CLEAN_START=${CLEAN_START:-1}
KEEP_REMOTE=${KEEP_REMOTE:-1}

DEFAULT_PLAYERS='cnn96_e08_aux128:artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx:artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json:aux:0.0035:0:0.0185:0:0:1.2:-0.1,mf80_e08_aux128:artifacts/top3_100m_overnight_20260509/mf80_100m/e08/model_k128.onnx:artifacts/top3_100m_overnight_20260509/mf80_100m/e08/model_k128.meta.json:aux:0.0115:0.007:0:0:0:1.8:-0.1,bt4_h8e5_classic128:artifacts/eval_puct_after_100m_limits_20260510/models/bt4_h8_100m_e5_merged.onnx:artifacts/lc0_lite_squareformer/h7_h8_100m/h8_static_relation_100m_e9/model.meta.json:puct:0:0:0:0:0:1.5:0,cnn80_e3_puct128:artifacts/100m_canonical/cnn_80x5_100m_e3.onnx:artifacts/100m_canonical/cnn_80x5_100m_e3.meta.json:puct:0:0:0:0:0:1.5:0,cnn80_top8_av010:artifacts/cnn_av_v2_80x5_phase3_tiny_lr/model.onnx:artifacts/cnn_av_v2_80x5_phase3_tiny_lr/model.meta.json:av:0.10:0:0:0:0:1.5:0,cnn64_top8_av015:artifacts/cnn_av_v2_64x6_phase2_low_lr/model.onnx:artifacts/cnn_av_v2_64x6_phase2_low_lr/model.meta.json:av:0.15:0:0:0:0:1.5:0,cnn64_c48_av010:artifacts/cnn_av_v3_64x6_c48_phase1/model.onnx:artifacts/cnn_av_v3_64x6_c48_phase1/model.meta.json:av:0.10:0:0:0:0:1.5:0,chessformer_e3_classic128:artifacts/100m_canonical/chessformer_v1_100m_e3.onnx:artifacts/100m_canonical/chessformer_v1_100m_e3.meta.json:puct:0:0:0:0:0:1.5:0'
PLAYERS=${PLAYERS:-$DEFAULT_PLAYERS}

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
need(){ [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 2; }; }

remote_base="artifacts/canonical_super_100m_arena_20260511/$JOB_NAME"
merged_name="super_100m_rust_aux_av_v${VISITS}_g${GAMES_PER_PAIR}_merged.json"

write_local_info(){
  mkdir -p "$LOCAL_OUT"
  cat > "$LOCAL_OUT/remote_info.env" <<EOF_INFO
REMOTE='$REMOTE'
RDIR='$RDIR'
LOCAL_OUT='$LOCAL_OUT'
JOB_NAME='$JOB_NAME'
WORKERS='$WORKERS'
VISITS='$VISITS'
GAMES_PER_PAIR='$GAMES_PER_PAIR'
MAX_PLIES='$MAX_PLIES'
MAX_OPENINGS='$MAX_OPENINGS'
ORT_THREADS='$ORT_THREADS'
OPENINGS='$OPENINGS'
PLAYERS='$PLAYERS'
EOF_INFO
}

pull_results(){
  mkdir -p "$LOCAL_OUT"
  if ssh "$REMOTE" "test -d '$RDIR/$remote_base'"; then
    rsync -az "$REMOTE:$RDIR/$remote_base/" "$LOCAL_OUT/"
  fi
  if ssh "$REMOTE" "test -f '$RDIR/run.log'"; then
    rsync -az "$REMOTE:$RDIR/run.log" "$LOCAL_OUT/remote_run.log"
  fi
  log "pulled results to $LOCAL_OUT"
  [[ -f "$LOCAL_OUT/status.txt" ]] && tail -20 "$LOCAL_OUT/status.txt" || true
  [[ -f "$LOCAL_OUT/status.final" ]] && printf 'status.final=%s\n' "$(cat "$LOCAL_OUT/status.final")" || true
  [[ -f "$LOCAL_OUT/summary.tsv" ]] && tail -20 "$LOCAL_OUT/summary.tsv" || true
}

remote_status(){
  ssh "$REMOTE" "cd '$RDIR' 2>/dev/null && echo '=== status ===' && tail -40 '$remote_base/status.txt' 2>/dev/null || true; echo '=== final ==='; cat '$remote_base/status.final' 2>/dev/null || true; echo '=== procs ==='; ps -axo pid,pcpu,pmem,etime,comm,args | grep -E 'tiny-leela-rust-arena|run_remote' | grep -v grep || true; echo '=== shard tails ==='; for f in '$remote_base'/shards/shard_*.log; do test -f \"\$f\" || continue; echo --- \$(basename \"\$f\"); tail -5 \"\$f\"; done"
}

if [[ "$ACTION" == "pull" ]]; then
  write_local_info
  pull_results
  exit 0
fi
if [[ "$ACTION" == "status" ]]; then
  remote_status
  exit 0
fi
if [[ "$ACTION" == "clean" ]]; then
  if [[ "$KEEP_REMOTE" == "1" ]]; then
    echo "KEEP_REMOTE=1; refusing to clean $REMOTE:$RDIR. Set KEEP_REMOTE=0 to remove." >&2
    exit 2
  fi
  ssh "$REMOTE" "rm -rf '$RDIR'"
  log "removed remote workdir $RDIR"
  exit 0
fi
if [[ "$ACTION" != "launch" ]]; then
  echo "bad ACTION=$ACTION (expected launch|pull|status|clean)" >&2
  exit 2
fi

need rust/tiny_leela_core/Cargo.toml
need scripts/merge_rust_arena_shards.py
need "$OPENINGS"

write_local_info
python3 - <<'PY' "$PLAYERS" "$OPENINGS" "$LOCAL_OUT/paths_to_sync.txt"
import os, sys
players, openings, out = sys.argv[1:]
paths = {openings}
missing = []
for spec in players.split(','):
    if not spec.strip():
        continue
    parts = spec.split(':')
    if len(parts) < 3:
        raise SystemExit(f'invalid player spec: {spec}')
    for p in parts[1:3]:
        if os.path.exists(p):
            paths.add(p)
            if os.path.exists(p + '.data'):
                paths.add(p + '.data')
        else:
            missing.append(p)
if missing:
    raise SystemExit('missing required player paths:\n' + '\n'.join(sorted(missing)))
with open(out, 'w') as f:
    for p in sorted(paths):
        f.write(p + '\n')
print(f'wrote {len(paths)} sync paths to {out}')
PY

log "remote=$REMOTE rdir=$RDIR local_out=$LOCAL_OUT workers=$WORKERS visits=$VISITS games_per_pair=$GAMES_PER_PAIR detach=$DETACH"
if [[ "$CLEAN_START" == "1" ]]; then
  ssh "$REMOTE" "rm -rf '$RDIR'"
fi
ssh "$REMOTE" "mkdir -p '$RDIR/scripts' '$RDIR/eval' '$RDIR/$remote_base/shards'"

log "sync Rust source, merger, openings, and model artifacts"
rsync -az --exclude target rust "$REMOTE:$RDIR/"
rsync -az scripts/merge_rust_arena_shards.py "$REMOTE:$RDIR/scripts/"
rsync -azR --files-from="$LOCAL_OUT/paths_to_sync.txt" . "$REMOTE:$RDIR/"

RUNNER=$(mktemp)
cat > "$RUNNER" <<EOF_RUNNER
#!/usr/bin/env bash
set -Eeuo pipefail
cd '$RDIR'
BASE='$remote_base'
STATUS="\$BASE/status.txt"
MERGED="\$BASE/$merged_name"
mkdir -p "\$BASE/shards"
: > "\$STATUS"
mark(){ printf '%s %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$*" | tee -a "\$STATUS"; }
trap 'rc=\$?; mark "FAILED rc=\$rc phase=\${PHASE:-unknown}"; echo failed > "\$BASE/status.final"; exit \$rc' ERR
export ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS'
export ORT_NUM_THREADS='$ORT_THREADS'
export RUST_BACKTRACE=1
PHASE=build
mark 'START build rust arena job=$JOB_NAME workers=$WORKERS visits=$VISITS games_per_pair=$GAMES_PER_PAIR'
cargo build --release --features native-ort --manifest-path rust/tiny_leela_core/Cargo.toml --bin tiny-leela-rust-arena > "\$BASE/build.log" 2>&1
BIN='rust/tiny_leela_core/target/release/tiny-leela-rust-arena'
PHASE=openings
python3 - <<'PY' '$OPENINGS' '$MAX_OPENINGS' "\$BASE/openings.selected.fen"
import sys
src, limit_s, out = sys.argv[1:]
limit = int(limit_s)
fens = []
for line in open(src):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if ' #' in line:
        line = line.split(' #', 1)[0].strip()
    fens.append(line)
if limit > 0:
    fens = fens[:limit]
with open(out, 'w') as f:
    for fen in fens:
        f.write(fen + '\n')
print(f'wrote {len(fens)} openings to {out}')
PY
PHASE=arena
mark 'START parallel shards'
pids=()
for i in \$(seq 0 $((WORKERS-1))); do
  (
    export ORT_INTRA_OP_NUM_THREADS='$ORT_THREADS'
    export ORT_NUM_THREADS='$ORT_THREADS'
    "\$BIN" \\
      --players '$PLAYERS' \\
      --games-per-pair '$GAMES_PER_PAIR' \\
      --visits '$VISITS' \\
      --max-plies '$MAX_PLIES' \\
      --progress-every '$PROGRESS_EVERY' \\
      --adjudicate '$ADJUDICATE' \\
      --adjudicate-threshold '$ADJUDICATE_THRESHOLD' \\
      --openings-file "\$BASE/openings.selected.fen" \\
      --shard-count '$WORKERS' \\
      --shard-index "\$i" \\
      --out "\$BASE/shards/shard_\$i.json"
  ) > "\$BASE/shards/shard_\$i.log" 2>&1 &
  pids+=("\$!")
  echo "\$!" > "\$BASE/shards/shard_\$i.pid"
  mark "LAUNCHED shard=\$i pid=\$!"
done
fail=0
for pid in "\${pids[@]}"; do
  if ! wait "\$pid"; then
    fail=1
  fi
done
if [[ "\$fail" != 0 ]]; then
  mark 'FAILED one_or_more_shards'
  echo failed > "\$BASE/status.final"
  exit 1
fi
PHASE=merge
mark 'MERGE shards'
python3 scripts/merge_rust_arena_shards.py "\$BASE"/shards/shard_*.json --out "\$MERGED" --summary-tsv "\$BASE/summary.tsv" > "\$BASE/merge.log" 2>&1
mark 'DONE parallel rust super arena'
echo succeeded > "\$BASE/status.final"
EOF_RUNNER
chmod +x "$RUNNER"
rsync -az "$RUNNER" "$REMOTE:$RDIR/run_remote.sh"
rm -f "$RUNNER"

if [[ "$DETACH" == "1" ]]; then
  ssh "$REMOTE" "cd '$RDIR' && nohup ./run_remote.sh > run.log 2>&1 < /dev/null & echo \$! > pid && echo PID=\$(cat pid)"
else
  ssh "$REMOTE" "cd '$RDIR' && ./run_remote.sh"
fi
printf 'RUNNING mac-mini rust parallel job=%s started_utc=%s workers=%s remote=%s rdir=%s local_out=%s\n' \
  "$JOB_NAME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$WORKERS" "$REMOTE" "$RDIR" "$LOCAL_OUT" \
  > artifacts/canonical_super_100m_arena_20260511/status.txt
cat artifacts/canonical_super_100m_arena_20260511/status.txt
