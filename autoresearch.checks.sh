#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

git diff --check

TYPECHECK_LOG="/tmp/lc0_autoresearch_typecheck_$$.log"
if ! npm run typecheck >"$TYPECHECK_LOG" 2>&1; then
  tail -80 "$TYPECHECK_LOG" >&2
  exit 1
fi

BUILD_LOG="/tmp/lc0_autoresearch_build_$$.log"
if ! npm run build:client >"$BUILD_LOG" 2>&1; then
  tail -80 "$BUILD_LOG" >&2
  exit 1
fi
