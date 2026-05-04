#!/usr/bin/env bash
set -euo pipefail

if [[ -f package.json ]] && command -v npm >/dev/null 2>&1; then
  npm test
fi

if [[ -x node_modules/.bin/tsc ]]; then
  node_modules/.bin/tsc --noEmit
fi
