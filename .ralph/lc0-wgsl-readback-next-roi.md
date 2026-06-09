# Ralph Loop: LC0 WGSL readback/scheduler next ROI

Branch: `autoresearch/lc0-wgsl-readback-b4-20260605`

Rules:
- Keep stable defaults unchanged; all WGSL/TVM/readback/scheduler variants remain explicit opt-ins.
- Do not treat `batchPipelineDepth>1` as parity-preserving promotion evidence.
- Do not overfit one noisy browser benchmark. Prefer adjacent controls, drift guards, and harness fixes.
- Keep the local autoresearch launcher disabled as `autoresearch.sh.disabled-local` unless explicitly requested otherwise.

Items:

- [x] 1. Confirm and document the benchmark browser-harness cleanup hook.
  - Run an adjacent cleaned `mixed-tvm-ffn`, WASM-input, JS-legal, b4/depth1 confirmation using the disabled local launcher.
  - Treat the cleanup as benchmark hygiene, not a runtime optimization.
  - Record artifacts and metric; do not claim a speedup from cleanup alone.

- [x] 2. Port/stabilize the LC0 hybrid fixture harness improvements.
  - Use strict Vite readiness/progress polling so fixture preflights fail fast with useful diagnostics instead of hanging.
  - Smoke the WASM-input WGSL-head b4/depth1 fixture path only; avoid depth>1 promotion evidence.

- [x] 3. Add/run a depth1-only readback attribution matrix for the current b4/depth1 lane.
  - Focus on map wait, synced readback, maps/eval, bytes/eval, and dispatch count.
  - Keep the result diagnostic unless it motivates a parity-preserving implementation change with drift.

- [x] 4. Productize explicit opt-in WGSL lane documentation.
  - Document WASM input + JS legal + mixed TVM FFN / WGSL heads as opt-in.
  - Include drift gates, cleanup-hygiene caveat, and defaults-unchanged warning.

- [x] 5. Scope the fused legal-mask/top-k/readback-shrink path.
  - Do not repeat standalone GPU legal priors.
  - Produce a short implementation plan/gate list for fused mask/softmax/top-k/readback shrink.

Progress notes:

- Item 1: First attempt with a long `LC0_AR_RUN_ID` failed because agent-browser session socket paths exceeded the macOS limit; no metric was emitted. Retried with short id `r1c113329` using `bash autoresearch.sh.disabled-local`. Cleaned fixed-suite confirmation passed with `hybrid_b4_ms_per_eval=5.508`, `evals_per_second=181.55`, `readback_synced_ms=5.267`, `readback_bytes=7444`, `readback_maps=0.260`, `dispatch_count=160`; drift passed `3/3` with f32 WDL max `0.000141`, f32 top-prior max `0.000403`. Artifacts: `/tmp/lc0_autoresearch/r1c113329_rep3.json`, `/tmp/lc0_autoresearch/r1c113329_drift.json`. Cleanup documented as measurement hygiene only in `docs/lc0web_custom_inference_checkpoint.md`.
- Item 2: Ported strict Vite readiness, progress polling, progress timeout, fixture IDs, root-child trace controls, depth semantics labels, root-visit L1 diagnostics, and max visit-L1 gate from the Ralph fixture harness. Smoke passed: `node --experimental-strip-types scripts/lc0_browser_hybrid_search_fixture_parity.mjs --port 5197 --head-backend wgsl --input-backend wasm --encoder-kernel mixed-tvm-ffn --batch 4 --batch-pipeline-depths 1 --fixture-limit 1 --visits 32 --timeout 180000 --progress-timeout 60000 --out /tmp/lc0-b4-wgsl-fixture-ralph-item2.json`.
- Item 3: Added npm alias `lc0:browser-readback-depth1-b4-matrix` and ran it against the current b4/depth1 lane. Artifact `/tmp/lc0-readback-depth1-b4-ralph-item3.json`: `wgsl-pipe1` median `181.6 visits/s`, search readback sync `18.36 ms`, eval readback bytes `7444`; standalone `wgsl-gpu-legal` reduced readback bytes but slowed to `173.0 visits/s` and higher search readback sync. Diagnostic only; no runtime promotion.
- Item 4: Updated `docs/lc0web_custom_inference_checkpoint.md` with explicit opt-in lane, cleanup caveat, drift artifacts, and readback matrix summary. Defaults remain unchanged.
- Item 5: Added `docs/lc0_fused_legal_topk_readback_plan.md` with non-goals, opt-in shape, gates, risks, and first isolated-parity milestone. It explicitly rejects repeating standalone GPU legal priors as a promotion path.
