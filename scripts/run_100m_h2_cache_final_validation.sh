#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Validate the 100M h2_state residual cache after the cache builder finishes.

Default behavior fails fast if artifacts/cache_build_100m_h2_state/done is missing.
Use --allow-partial to run a progress/preflight validation while the cache is still building.
Use --wait to poll until the done marker appears.

Usage:
  scripts/run_100m_h2_cache_final_validation.sh
  scripts/run_100m_h2_cache_final_validation.sh --allow-partial
  scripts/run_100m_h2_cache_final_validation.sh --wait
USAGE
}

ROOT="artifacts/cache_build_100m_h2_state"
DONE="$ROOT/done"
OUT="artifacts/analysis/100m_cache_h2_state_schema_check.final.json"
PY="${PYTHON_BIN:-.venv-onnx/bin/python}"
ALLOW_PARTIAL=0
WAIT=0
SLEEP=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-partial) ALLOW_PARTIAL=1; shift;;
    --wait) WAIT=1; shift;;
    --sleep) SLEEP="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

if [[ "$WAIT" == 1 ]]; then
  echo "WAIT for $DONE"
  while [[ ! -e "$DONE" ]]; do
    date -Is
    train_count=$(find data/datasets/supervised_100m_elite_tcec_v1/cache_h2_state/train -maxdepth 2 -name meta.json 2>/dev/null | wc -l | tr -d ' ')
    dev_count=$(find data/datasets/supervised_100m_elite_tcec_v1/cache_h2_state/dev -maxdepth 2 -name meta.json 2>/dev/null | wc -l | tr -d ' ')
    echo "cache metas train=$train_count dev=$dev_count"
    sleep "$SLEEP"
  done
elif [[ ! -e "$DONE" && "$ALLOW_PARTIAL" != 1 ]]; then
  echo "missing done marker: $DONE" >&2
  echo "Use --allow-partial for in-progress preflight, or --wait to block." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
args=(
  scripts/validate_cache_schema.py
  --dataset-manifest data/datasets/supervised_100m_elite_tcec_v1/manifest.json
  --cache-dir data/datasets/supervised_100m_elite_tcec_v1/cache_h2_state
  --expect-input-planes 46
  --expect-policy-size 1968
  --expect-history-plies 2
  --expect-state-planes true
  --out "$OUT"
)
if [[ "$ALLOW_PARTIAL" == 1 && ! -e "$DONE" ]]; then
  args+=(--allow-partial)
fi

echo "RUN $PY ${args[*]}"
"$PY" "${args[@]}"
echo "WROTE $OUT"
