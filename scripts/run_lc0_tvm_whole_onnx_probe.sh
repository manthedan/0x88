#!/usr/bin/env bash
set -euo pipefail

# Reproducible whole-model ONNX -> TVM Relax -> target probe for LC0 browser models.
# Run from anywhere; durable TVM state lives under /Users/macthedan/projects/lc0_browser.

ROOT="${LC0_BROWSER_ROOT:-/Users/macthedan/projects/lc0_browser}"
REPO="${LC0_WEB_REPO:-$ROOT/leelaweb}"
TVM_SRC="${TVM_SRC:-$ROOT/.deps/tvm-webgpu-src}"
TVM_ENV="${TVM_ENV:-$ROOT/.envs/tvm-mlc-py313}"
TARGET="${TVM_TARGET:-webgpu}"
OUT_DIR="${OUT_DIR:-$REPO/artifacts/tvm}"
MODEL_DIR="${MODEL_DIR:-$REPO/public/models/lc0}"
TVM_BUILD_DIR="${TVM_BUILD_DIR:-build}"
TVM_LIB_DIR="$TVM_SRC/$TVM_BUILD_DIR/lib"

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:$TVM_ENV/bin:${PATH:-}"
export TVM_LIBRARY_PATH="$TVM_LIB_DIR"
export DYLD_LIBRARY_PATH="$TVM_LIB_DIR:/opt/homebrew/opt/llvm/lib:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_SRC/python:${PYTHONPATH:-}"

extra_args=()
if [[ "${CAST_INT64_INITIALIZERS_TO_INT32:-0}" == "1" ]]; then
  extra_args+=(--cast-int64-initializers-to-int32)
fi
if [[ "${TRUST_NONNEGATIVE_GATHER_INDICES:-0}" == "1" ]]; then
  extra_args+=(--trust-nonnegative-gather-indices)
fi
if [[ "${SANITIZE_ONNX_NAMES:-0}" == "1" ]]; then
  extra_args+=(--sanitize-onnx-names)
fi
if [[ "${CAPTURE_MODULE_SOURCES:-0}" == "1" ]]; then
  extra_args+=(--capture-module-sources)
fi
if [[ -n "${TVM_HOST_TARGET:-}" ]]; then
  extra_args+=(--host-target "$TVM_HOST_TARGET")
fi
if [[ "${EXPORT_TVMJS_WASM:-0}" == "1" ]]; then
  extra_args+=(--export-tvmjs-wasm)
fi

MODEL_FAMILY="${LC0_TVMJS_MODEL_FAMILY:-t1-256x10-distilled-swa-2432500}"
DTYPE="${LC0_TVMJS_DTYPE:-f16}"
BATCHES_CSV="${LC0_TVMJS_BATCHES:-1,4,8}"
if [[ -n "${LC0_TVMJS_MODEL_TEMPLATE:-}" ]]; then
  MODEL_TEMPLATE="$LC0_TVMJS_MODEL_TEMPLATE"
else
  MODEL_TEMPLATE='{family}.batch{batch}.{dtype}.onnx'
fi

IFS=',' read -r -a batches <<< "$BATCHES_CSV"
models=()
for raw_batch in "${batches[@]}"; do
  batch="${raw_batch//[[:space:]]/}"
  if [[ -z "$batch" ]]; then
    echo "Invalid empty batch token in LC0_TVMJS_BATCHES=$BATCHES_CSV" >&2
    exit 2
  fi
  if [[ ! "$batch" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid positive-integer batch token '$batch' in LC0_TVMJS_BATCHES=$BATCHES_CSV" >&2
    exit 2
  fi
  model="$MODEL_TEMPLATE"
  model="${model//\{family\}/$MODEL_FAMILY}"
  model="${model//\{modelFamily\}/$MODEL_FAMILY}"
  model="${model//\{batch\}/$batch}"
  model="${model//\{dtype\}/$DTYPE}"
  models+=("$model")
done

if [[ "${#models[@]}" -eq 0 ]]; then
  echo "No models resolved from LC0_TVMJS_BATCHES=$BATCHES_CSV" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[lc0-tvm-probe] repo=%s\n' "$REPO"
  printf '[lc0-tvm-probe] model_dir=%s\n' "$MODEL_DIR"
  printf '[lc0-tvm-probe] out_dir=%s\n' "$OUT_DIR"
  printf '[lc0-tvm-probe] target=%s\n' "$TARGET"
  printf '[lc0-tvm-probe] models:\n'
  printf '  %s\n' "${models[@]}"
  exit 0
fi

if [[ ! -x "$TVM_ENV/bin/python" ]]; then
  echo "Missing TVM python: $TVM_ENV/bin/python" >&2
  exit 2
fi
if [[ ! -d "$TVM_LIB_DIR" ]]; then
  echo "Missing TVM build lib dir: $TVM_LIB_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

for model in "${models[@]}"; do
  model_path="$MODEL_DIR/$model"
  out_suffix="${OUT_SUFFIX:-}"
  if [[ -z "$out_suffix" && "${EXPORT_TVMJS_WASM:-0}" == "1" ]]; then
    out_suffix=".tvmjs-wasm"
  fi
  out_name="${model%.onnx}.${TARGET}${out_suffix}.probe.json"
  echo "[lc0-tvm-probe] $model -> $TARGET"
  "$TVM_ENV/bin/python" "$REPO/scripts/lc0_tvm_whole_onnx_probe.py" \
    --model "$model_path" \
    --target "$TARGET" \
    --out "$OUT_DIR/$out_name" \
    "${extra_args[@]}"
done
