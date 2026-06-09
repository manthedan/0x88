#!/usr/bin/env bash
set -euo pipefail

# Build the optional TVMJS/WebGPU export toolchain beside the native/source TVM build.
# This does not replace $TVM_SRC/build; it creates/updates $TVM_SRC/build-tvmjs.

ROOT="${LC0_BROWSER_ROOT:-/Users/macthedan/projects/lc0_browser}"
TVM_SRC="${TVM_SRC:-$ROOT/.deps/tvm-webgpu-src}"
TVM_ENV="${TVM_ENV:-$ROOT/.envs/tvm-mlc-py313}"
TVM_BUILD_DIR="${TVM_BUILD_DIR:-build-tvmjs}"
LLVM_CONFIG="${LLVM_CONFIG:-/opt/homebrew/opt/llvm/bin/llvm-config}"

if [[ ! -x "$TVM_ENV/bin/python" ]]; then
  echo "Missing TVM python env: $TVM_ENV/bin/python" >&2
  exit 2
fi
if [[ ! -x "$TVM_ENV/bin/cmake" || ! -x "$TVM_ENV/bin/ninja" ]]; then
  echo "Missing cmake/ninja in TVM env: $TVM_ENV/bin" >&2
  exit 2
fi
if [[ ! -x "$LLVM_CONFIG" ]]; then
  echo "Missing LLVM config: $LLVM_CONFIG" >&2
  echo "Install LLVM, e.g. brew install llvm, or pass LLVM_CONFIG=/path/to/llvm-config." >&2
  exit 2
fi
if ! command -v emcc >/dev/null 2>&1; then
  echo "Missing emcc on PATH. Install/activate Emscripten, e.g. brew install emscripten." >&2
  exit 2
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:$TVM_ENV/bin:${PATH:-}"
export TVM_BUILD_DIR LLVM_CONFIG
cd "$TVM_SRC"
mkdir -p "$TVM_BUILD_DIR"
if [[ -f build/config.cmake && ! -f "$TVM_BUILD_DIR/config.cmake" ]]; then
  cp build/config.cmake "$TVM_BUILD_DIR/config.cmake"
elif [[ ! -f "$TVM_BUILD_DIR/config.cmake" ]]; then
  cp cmake/config.cmake "$TVM_BUILD_DIR/config.cmake"
fi
"$TVM_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
p = Path(os.environ.get('TVM_BUILD_DIR', 'build-tvmjs')) / 'config.cmake'
llvm_config = os.environ['LLVM_CONFIG']
s = p.read_text()
if 'set(USE_LLVM OFF)' in s:
    s = s.replace('set(USE_LLVM OFF)', f'set(USE_LLVM {llvm_config})')
elif 'set(USE_LLVM ' in s:
    import re
    s = re.sub(r'set\(USE_LLVM .*?\)', f'set(USE_LLVM {llvm_config})', s)
else:
    s += f'\nset(USE_LLVM {llvm_config})\n'
p.write_text(s)
PY
"$TVM_ENV/bin/cmake" -S . -B "$TVM_BUILD_DIR" -G Ninja
"$TVM_ENV/bin/ninja" -C "$TVM_BUILD_DIR"
make -C web -j"${JOBS:-4}"
if [[ ! -d web/node_modules ]]; then
  npm --prefix web install
fi
npm --prefix web run bundle

echo "TVMJS/WebGPU toolchain ready:"
echo "  TVM_LIBRARY_PATH=$TVM_SRC/$TVM_BUILD_DIR/lib"
echo "  web runtime bitcode=$TVM_SRC/web/dist/wasm"
echo "  tvmjs bundle=$TVM_SRC/web/dist/tvmjs.bundle.js"
