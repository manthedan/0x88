#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="data/selfplay/pilot_$(date -u +%Y%m%dT%H%M%SZ)"
MODEL=""
META=""
MODEL_ID=""
GAMES=100
VISITS=64
MAX_PLIES=160
SEED=1
STOCKFISH=".local_engines/stockfish_pkg/usr/games/stockfish"
STOCKFISH_DEPTH=8
STOCKFISH_NODES=0
MOCK_STOCKFISH=0
OPENING_FENS=""
POLICY_MODE="classic"
PROGRESS_EVERY=20
TRAINING_LANE="supervised_sp"
SOURCE_MODEL=""

usage(){ cat <<'EOF'
usage: scripts/selfplay_pilot_pipeline.sh --model MODEL [--meta META] [options]

Runs a small correctness-first self-play pilot:
  raw selfplay_chunk_v1 -> validate -> Stockfish sidecar -> agent diagnostics
  -> joined pipeline manifest -> expanded training rows.

Options:
  --out-dir DIR              output directory under data/selfplay or artifacts staging
  --model PATH               student JSON or ONNX model path
  --meta PATH                ONNX meta JSON path
  --model-id ID              provenance model id (defaults to model basename)
  --games N                  default 100
  --visits N                 default 64
  --max-plies N              default 160
  --seed N                   default 1
  --opening-fens PATH        optional opening FEN list
  --policy-mode NAME         rust search policy mode, default classic
  --stockfish PATH           UCI stockfish path
  --stockfish-depth N        default 8
  --stockfish-nodes N        if >0, use fixed-node Stockfish
  --mock-stockfish           dry-run sidecar without UCI Stockfish
  --source-model ID          training-row source_model, defaults to model-id
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --meta) META="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;
    --games) GAMES="$2"; shift 2;;
    --visits) VISITS="$2"; shift 2;;
    --max-plies) MAX_PLIES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --opening-fens) OPENING_FENS="$2"; shift 2;;
    --policy-mode) POLICY_MODE="$2"; shift 2;;
    --stockfish) STOCKFISH="$2"; shift 2;;
    --stockfish-depth) STOCKFISH_DEPTH="$2"; shift 2;;
    --stockfish-nodes) STOCKFISH_NODES="$2"; shift 2;;
    --mock-stockfish) MOCK_STOCKFISH=1; shift;;
    --source-model) SOURCE_MODEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

if [[ -z "$MODEL" ]]; then echo "--model is required" >&2; usage >&2; exit 2; fi
if [[ -z "$MODEL_ID" ]]; then MODEL_ID="$(basename "$MODEL")"; fi
if [[ -z "$SOURCE_MODEL" ]]; then SOURCE_MODEL="$MODEL_ID"; fi
mkdir -p "$OUT_DIR"

CHUNK="$OUT_DIR/chunk.selfplay_chunk_v1.jsonl.zst"
CHUNK_MANIFEST="$OUT_DIR/chunk.manifest.json"
STOCKFISH_ANN="$OUT_DIR/stockfish.selfplay_annotation_v1.jsonl.zst"
AGENT_ANN="$OUT_DIR/agent.selfplay_annotation_v1.jsonl.zst"
FAILURE_DIR="$OUT_DIR/failure_packets"
PIPELINE_MANIFEST="$OUT_DIR/pipeline.manifest.json"
TRAINING_ROWS="$OUT_DIR/training_expanded.jsonl.zst"
TRAINING_MANIFEST="$OUT_DIR/training_rows.manifest.json"

GEN=(node --experimental-strip-types scripts/selfplay_generate.mjs --backend rust --model "$MODEL" --out "$CHUNK" --manifest-out "$CHUNK_MANIFEST" --lane sup_sp --model-id "$MODEL_ID" --games "$GAMES" --visits "$VISITS" --max-plies "$MAX_PLIES" --seed "$SEED" --policy-mode "$POLICY_MODE" --progress-every "$PROGRESS_EVERY")
if [[ -n "$META" ]]; then GEN+=(--meta "$META"); fi
if [[ -n "$OPENING_FENS" ]]; then GEN+=(--opening-fens "$OPENING_FENS"); fi

echo "[pilot] generate chunk: $CHUNK"
"${GEN[@]}"

.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$CHUNK"

ANN=(.venv-onnx/bin/python scripts/selfplay_stockfish_annotate.py --input "$CHUNK" --out "$STOCKFISH_ANN" --stockfish "$STOCKFISH" --depth "$STOCKFISH_DEPTH" --nodes "$STOCKFISH_NODES")
if [[ "$MOCK_STOCKFISH" == 1 ]]; then ANN+=(--mock-stockfish); fi

echo "[pilot] stockfish annotate: $STOCKFISH_ANN"
"${ANN[@]}"
.venv-onnx/bin/python scripts/selfplay_annotation_validate.py "$STOCKFISH_ANN"

echo "[pilot] agent diagnostics: $AGENT_ANN"
.venv-onnx/bin/python scripts/selfplay_agent_diagnostics.py \
  --input "$CHUNK" \
  --annotation "$STOCKFISH_ANN" \
  --out "$AGENT_ANN" \
  --failure-dir "$FAILURE_DIR" \
  --model-id "$MODEL_ID"
.venv-onnx/bin/python scripts/selfplay_annotation_validate.py "$AGENT_ANN"

echo "[pilot] pipeline manifest: $PIPELINE_MANIFEST"
.venv-onnx/bin/python scripts/selfplay_pipeline_manifest.py \
  --chunk "$CHUNK" \
  --annotation "$STOCKFISH_ANN" \
  --annotation "$AGENT_ANN" \
  --out "$PIPELINE_MANIFEST" \
  --strict-annotations

echo "[pilot] training rows: $TRAINING_ROWS"
.venv-onnx/bin/python scripts/selfplay_manifest_to_training.py \
  --manifest "$PIPELINE_MANIFEST" \
  --output "$TRAINING_ROWS" \
  --manifest-out "$TRAINING_MANIFEST" \
  --lane "$TRAINING_LANE" \
  --source-model "$SOURCE_MODEL" \
  --mode expanded \
  --value-target result
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$TRAINING_ROWS" --min-policy-mass 0.99 --max-policy-mass 1.01

cat <<EOF
[pilot] done
chunk=$CHUNK
stockfish_annotation=$STOCKFISH_ANN
agent_annotation=$AGENT_ANN
pipeline_manifest=$PIPELINE_MANIFEST
training_rows=$TRAINING_ROWS
training_manifest=$TRAINING_MANIFEST
failure_packets=$FAILURE_DIR
EOF
