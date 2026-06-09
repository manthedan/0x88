# LC0 browser b4/depth1 safe benchmark protocol

This protocol is for opt-in LC0 browser diagnostics only. It must not change stable defaults and must not treat browser cleanup as a runtime speedup.

## Scope

Canonical lane under test:

- runtime: `hybrid-wgsl-heads`
- head backend: `wgsl` (explicit opt-in)
- input backend: `wasm` (explicit opt-in)
- legal priors: `js`
- encoder kernel: current WGSL-head b4 lane `mixed-tvm-ffn`
- leaf batch: `4`
- batch pipeline depth: `1`

Do not use `batchPipelineDepth>1`, standalone GPU legal priors, WASM legal priors, or smolgen-project as promotion evidence unless a separate plan says so and drift passes.

## Harness cleanup hygiene

Before a measured fixed-suite run, clear only wrapper-owned browser harness state and benchmark Vite ports. This makes adjacent browser measurements less noisy; it is not a runtime optimization.

```bash
pkill -f '/node_modules/agent-browser/bin/agent-browser-darwin-arm64' >/dev/null 2>&1 || true
pkill -f '/\.agent-browser/browsers/.*/Google Chrome for Testing' >/dev/null 2>&1 || true
vite_pids="$( { lsof -tiTCP:5179 -sTCP:LISTEN 2>/dev/null; lsof -tiTCP:5180 -sTCP:LISTEN 2>/dev/null; } | sort -u || true )"
if [[ -n "$vite_pids" ]]; then kill $vite_pids >/dev/null 2>&1 || true; fi
sleep 20
```

Use short run IDs/session names. The benchmark launchers now truncate/hash long `--session` values, but short names keep artifacts readable.

## 1. Fixture smoke / search parity

Run a small fixture smoke before spending fixed-suite time:

```bash
node scripts/lc0_browser_hybrid_search_fixture_parity.mjs \
  --preset lc0-webgpu-research-b4 \
  --fixture-limit 3 \
  --visits 32 \
  --batch-pipeline-depths 1 \
  --trace-root-children \
  --out /tmp/lc0_b4_depth1_fixture_smoke.json \
  --timeout 300000
```

Expected: `LC0_SEARCH_FIXTURE_PARITY_DONE`, no mismatches, and depth-1 semantics only.

## 2. Cleaned fixed-suite confirmation

For the short local confirmation used during iteration, use committed browser scripts directly:

```bash
node --experimental-strip-types scripts/lc0_browser_runtime_fixed_suite.mjs \
  --fens artifacts/lc0_runtime_arena_20260605/batch_matrix/fixed_suite_32_fens.txt \
  --max-positions 4 \
  --runtimes hybrid-wgsl-heads \
  --movetime 500 \
  --stockfish-score-depth 1 \
  --lc0-batch-size 4 \
  --batch-pipeline-depth 1 \
  --input-backend wasm \
  --encoder-kernel mixed-tvm-ffn \
  --legal-priors-backend js \
  --session rshort1-fixed \
  --out /tmp/lc0_b4_depth1_fixed_suite.json \
  --summary-only \
  --timeout 180000
```

Then run the committed drift harness:

```bash
node --experimental-strip-types scripts/lc0_browser_hybrid_drift.mjs \
  --limit 3 \
  --head-backend wgsl \
  --input-backend wasm \
  --encoder-kernel mixed-tvm-ffn \
  --legal-priors-backend js \
  --timeout 180000 > /tmp/lc0_b4_depth1_drift.json
```

Extract the core fixed-suite metric from the fixed-suite artifact:

```bash
node - <<'NODE'
const fs = require('fs');
const report = JSON.parse(fs.readFileSync('/tmp/lc0_b4_depth1_fixed_suite.json', 'utf8'));
const row = report.summary?.[0];
const evalsPerSecond = Number(row?.evalsPerSecond ?? 0);
console.log(`METRIC hybrid_b4_ms_per_eval=${evalsPerSecond > 0 ? 1000 / evalsPerSecond : 0}`);
console.log(`METRIC evals_per_second=${evalsPerSecond}`);
console.log(`ARTIFACT fixed_suite=/tmp/lc0_b4_depth1_fixed_suite.json`);
console.log('ARTIFACT drift=/tmp/lc0_b4_depth1_drift.json');
NODE
```

For a stronger local confirmation, raise `--max-positions`/repeat the fixed-suite command only after the short run and drift pass. A local-only wrapper may automate these same steps in some checkouts, but this protocol must remain reproducible from committed scripts.

Record:

- `hybrid_b4_ms_per_eval`
- `evals_per_second`
- fixed-suite artifact path
- drift artifact path
- optional backend timing fields from `positions[*].lc0Search.evalBackendTimingPerPositionMeans`: `readbackSyncedMs`, `readbackBytes`, `readbackMapCount`, `dispatchCount`

## 3. Drift gates

Treat the run as failed if any drift guard fails:

- f32 best-move matches: all fixtures
- native best-move matches: all fixtures
- f32 WDL max abs diff: `<= 0.005`
- f32 top-prior max abs diff: `<= 0.01`

A faster fixed-suite number with failed drift is diagnostic only and must not be promoted.

## 4. Optional readback attribution

Use the depth-1 b4 attribution alias for nearby JS-legal vs GPU-legal controls:

```bash
npm run lc0:browser-readback-depth1-b4-matrix -- \
  --out /tmp/lc0-readback-depth1-b4.json \
  --max-positions 4 \
  --repeats 1 \
  --timeout 240000
```

Interpretation rule: standalone GPU legal priors reducing bytes is insufficient. It must also win end-to-end under adjacent controls and pass drift; otherwise only revisit fused legal-mask/softmax/top-k/readback-shrink designs.

## 5. Closeout checks

Before committing code or docs from a benchmark loop:

```bash
npm run typecheck
npm run build:client
git diff --check
```

Run autoreview for non-trivial code changes. Keep any local-only autoresearch launchers out of the canonical protocol unless they are committed under a supported name.
