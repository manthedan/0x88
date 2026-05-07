#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/run_model_release_gate.sh --name NAME --model MODEL.onnx --meta MODEL.meta.json [options]

Runs the non-architecture release gate for one model and writes outputs under --out-dir.

Options:
  --out-dir DIR              default artifacts/release_gates/NAME_TIMESTAMP
  --positions-json PATH      default artifacts/diagnostics/queen_risk_fixed_suite_drop300_or_capture.json
  --dev-jsonl-zst PATH       default data/datasets/supervised_100m_elite_tcec_v1/dev/dev_1000000.jsonl.zst
  --openings-file PATH       default eval/opening_suite_uho_lite_v1.fen
  --bucket-rows N            default 5000
  --quick-pairs N            default 3
  --full-pairs N             default 0 (skip full anchor)
  --visits-list CSV          default 1,32,128,512
  --full-visits N            default 512
  --cpuct X                  default 1.5
  --skip-build               skip npm run build:client
EOF
}

NAME=""; MODEL=""; META=""; OUT_DIR=""; POSITIONS="artifacts/diagnostics/queen_risk_fixed_suite_drop300_or_capture.json"; DEV="data/datasets/supervised_100m_elite_tcec_v1/dev/dev_1000000.jsonl.zst"; OPENINGS="eval/opening_suite_uho_lite_v1.fen"; BUCKET_ROWS=5000; QUICK_PAIRS=3; FULL_PAIRS=0; VISITS_LIST="1,32,128,512"; FULL_VISITS=512; CPUCT=1.5; SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --meta) META="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --positions-json) POSITIONS="$2"; shift 2;;
    --dev-jsonl-zst) DEV="$2"; shift 2;;
    --openings-file) OPENINGS="$2"; shift 2;;
    --bucket-rows) BUCKET_ROWS="$2"; shift 2;;
    --quick-pairs) QUICK_PAIRS="$2"; shift 2;;
    --full-pairs) FULL_PAIRS="$2"; shift 2;;
    --visits-list) VISITS_LIST="$2"; shift 2;;
    --full-visits) FULL_VISITS="$2"; shift 2;;
    --cpuct) CPUCT="$2"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done
[[ -n "$NAME" && -n "$MODEL" && -n "$META" ]] || { usage >&2; exit 2; }
if [[ -z "$OUT_DIR" ]]; then OUT_DIR="artifacts/release_gates/${NAME}_$(date +%Y%m%d_%H%M%S)"; fi
mkdir -p "$OUT_DIR"

run() {
  local label="$1"; shift
  echo "[gate] $label"
  "$@" > "$OUT_DIR/${label}.log" 2>&1
}

cat > "$OUT_DIR/protocol_card.json" <<EOF
{
  "name": "$NAME",
  "model": "$MODEL",
  "meta": "$META",
  "positions_json": "$POSITIONS",
  "dev_jsonl_zst": "$DEV",
  "openings_file": "$OPENINGS",
  "bucket_rows": $BUCKET_ROWS,
  "quick_pairs": $QUICK_PAIRS,
  "full_pairs": $FULL_PAIRS,
  "visits_list": "$VISITS_LIST",
  "full_visits": $FULL_VISITS,
  "cpuct": $CPUCT,
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
}
EOF

if [[ "$SKIP_BUILD" != 1 ]]; then run build_client npm run build:client; fi
run puct_core node --experimental-strip-types eval/puct_core_tests.mjs
run puct_node_test node --test tests/puct_core.test.mjs
run puct_consistency node --experimental-strip-types eval/puct_consistency_check.mjs --model "$MODEL" --meta "$META" --positions-json "$POSITIONS" --limit 100 --visits "$VISITS_LIST" --out "$OUT_DIR/puct_consistency.json"
run queen_fixed node --experimental-strip-types eval/queen_plumbing_diagnostic.mjs --model "$MODEL" --meta "$META" --positions-json "$POSITIONS" --out "$OUT_DIR/queen_fixed.json"
run bucket_eval node --experimental-strip-types eval/onnx_bucket_eval_jsonl.mjs --input "$DEV" --model "$MODEL" --meta "$META" --out "$OUT_DIR/bucket_eval.json" --max-rows-per-bucket "$BUCKET_ROWS"

IFS=',' read -ra VISITS <<< "$VISITS_LIST"
for v in "${VISITS[@]}"; do
  run "anchor_quick_v${v}" node --experimental-strip-types eval/uci_anchor_arena.mjs \
    --candidate="${NAME}_v${v}:${MODEL}:${META}" \
    --openings-file="$OPENINGS" \
    --pairs="$QUICK_PAIRS" \
    --visits="$v" \
    --cpuct="$CPUCT" \
    --stockfish-levels=1320,1600 \
    --stockfish-nodes=32 \
    --max-plies=100 \
    --uci-anchors='maia1100|.local_engines/maia/lc0-maia-1100.sh|32' \
    --out "$OUT_DIR/anchor_quick_v${v}.json"
done

if [[ "$FULL_PAIRS" -gt 0 ]]; then
  run "anchor_full_v${FULL_VISITS}" node --experimental-strip-types eval/uci_anchor_arena.mjs \
    --candidate="${NAME}_v${FULL_VISITS}:${MODEL}:${META}" \
    --openings-file="$OPENINGS" \
    --pairs="$FULL_PAIRS" \
    --visits="$FULL_VISITS" \
    --cpuct="$CPUCT" \
    --stockfish-levels=1320,1600 \
    --stockfish-nodes=32 \
    --max-plies=100 \
    --uci-anchors='maia1100|.local_engines/maia/lc0-maia-1100.sh|32,maia1500|.local_engines/maia/lc0-maia-1500.sh|32,maia1900|.local_engines/maia/lc0-maia-1900.sh|32' \
    --out "$OUT_DIR/anchor_full_v${FULL_VISITS}.json"
fi

echo "[gate] wrote $OUT_DIR"
