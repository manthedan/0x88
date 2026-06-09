#!/usr/bin/env bash
set -euo pipefail

# Reproducible whole-model ONNX -> TVM Relax -> target probe for LC0 browser models.
# Run from anywhere; durable TVM state lives under /Users/macthedan/projects/lc0_browser.

ROOT="${LC0_BROWSER_ROOT:-/Users/macthedan/projects/lc0_browser}"
REPO="$ROOT/leelaweb"
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

if [[ ! -x "$TVM_ENV/bin/python" ]]; then
  echo "Missing TVM python: $TVM_ENV/bin/python" >&2
  exit 2
fi
if [[ ! -d "$TVM_LIB_DIR" ]]; then
  echo "Missing TVM build lib dir: $TVM_LIB_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

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

models=(
  "t1-256x10-distilled-swa-2432500.batch1.f16.onnx"
  "t1-256x10-distilled-swa-2432500.batch4.f16.onnx"
  "t1-256x10-distilled-swa-2432500.batch8.f16.onnx"
)

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
