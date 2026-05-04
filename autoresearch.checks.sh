#!/usr/bin/env bash
set -euo pipefail

if [[ -f package.json ]] && command -v npm >/dev/null 2>&1; then
  npm test -- --run || npm test
fi

if [[ -f tsconfig.json ]] && command -v npx >/dev/null 2>&1; then
  npx tsc --noEmit
fi
