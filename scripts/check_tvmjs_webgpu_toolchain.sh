#!/usr/bin/env bash
set -euo pipefail

REPO="${LC0_WEB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${LC0_BROWSER_ROOT:-$(dirname "$REPO")}"
TVM_SRC="${TVM_SRC:-$ROOT/.deps/tvm-webgpu-src}"
TVM_ENV="${TVM_ENV:-$ROOT/.envs/tvm-mlc-py313}"
TVM_BUILD_DIR="${TVM_BUILD_DIR:-build-tvmjs}"
TVM_LIB_DIR="$TVM_SRC/$TVM_BUILD_DIR/lib"
HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
LLVM_PREFIX="${LLVM_PREFIX:-$HOMEBREW_PREFIX/opt/llvm}"

if [[ ! -x "$TVM_ENV/bin/python" ]]; then
  echo "Missing TVM python: $TVM_ENV/bin/python" >&2
  exit 2
fi

export PATH="$HOMEBREW_PREFIX/bin:$LLVM_PREFIX/bin:$TVM_ENV/bin:${PATH:-}"
export TVM_SRC TVM_BUILD_DIR
export TVM_LIBRARY_PATH="$TVM_LIB_DIR"
export DYLD_LIBRARY_PATH="$TVM_LIB_DIR:$LLVM_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_SRC/python:${PYTHONPATH:-}"

exec "$TVM_ENV/bin/python" "$REPO/scripts/check_tvmjs_webgpu_toolchain.py"
