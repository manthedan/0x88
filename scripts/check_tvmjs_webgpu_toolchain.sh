#!/usr/bin/env bash
set -euo pipefail

ROOT="${LC0_BROWSER_ROOT:-/Users/macthedan/projects/lc0_browser}"
REPO="$ROOT/leelaweb"
TVM_SRC="${TVM_SRC:-$ROOT/.deps/tvm-webgpu-src}"
TVM_ENV="${TVM_ENV:-$ROOT/.envs/tvm-mlc-py313}"
TVM_BUILD_DIR="${TVM_BUILD_DIR:-build-tvmjs}"
TVM_LIB_DIR="$TVM_SRC/$TVM_BUILD_DIR/lib"

if [[ ! -x "$TVM_ENV/bin/python" ]]; then
  echo "Missing TVM python: $TVM_ENV/bin/python" >&2
  exit 2
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:$TVM_ENV/bin:${PATH:-}"
export TVM_SRC TVM_BUILD_DIR
export TVM_LIBRARY_PATH="$TVM_LIB_DIR"
export DYLD_LIBRARY_PATH="$TVM_LIB_DIR:/opt/homebrew/opt/llvm/lib:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_SRC/python:${PYTHONPATH:-}"

exec "$TVM_ENV/bin/python" "$REPO/scripts/check_tvmjs_webgpu_toolchain.py"
