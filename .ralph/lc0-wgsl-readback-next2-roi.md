# Ralph Loop: LC0 WGSL readback ROI continuation

Branch: `autoresearch/lc0-wgsl-readback-b4-20260605`

Rules:
- Stable defaults stay unchanged.
- WGSL heads, WASM input, GPU/legal-top-k paths, TVM/smolgen kernels, and cleanup protocols remain explicit opt-ins.
- Do not treat cleanup as a runtime speedup; compare only similarly cleaned controls.
- Do not use `batchPipelineDepth>1` as promotion evidence.
- Keep local `autoresearch.sh` disabled unless explicitly requested.

Items:

- [x] 1. Add an isolated GPU legal/top-k parity route.
  - Browser probe only: compare JS legal priors against WGSL GPU legal-prior scaffold on native fixtures.
  - Report top-K agreement, max prior/logit/WDL drift, and compact top-K byte estimates.
  - No search wiring or default change.

- [x] 2. Run a short alternating JS-legal vs GPU/top-k prototype matrix.
  - Same-session/nearby controls only; diagnostic, not promotion.
  - Capture bytes/maps/dispatch/readback and speed.

- [x] 3. Reconfirm the stronger current opt-in encoder kernel stack.
  - Use `mixed-tvm-ffn-smolgen-project` with WASM input, JS legal, b4/depth1, cleanup, and drift.
  - Align docs with the actual stronger opt-in where evidence passes.

- [x] 4. Add session-name length guards to browser benchmark launchers.
  - Prevent agent-browser socket-path failures from long run IDs by shortening/hashing session names.
  - Cover fixed-suite and fixture harnesses.

- [x] 5. Add a canonical cleaned b4/depth1 protocol script/doc.
  - One safe opt-in confirmation protocol that runs cleanup, fixture smoke, fixed suite, and drift.
  - Avoid promoting cleanup or experimental lanes to defaults.

Progress notes:

2026-06-08 item 1:
- Added explicit `gpuLegalParity=1` browser route in `src/lc0/policyOnlyBrowser.ts`.
- Added `scripts/lc0_browser_gpu_legal_parity.mjs` and npm alias `lc0:browser-gpu-legal-parity`.
- Probe compares JS legal priors vs opt-in WGSL GPU legal priors using WGSL heads, reports top-K parity, drift, and full-vs-compact readback byte estimates.
- Validation:
  - `npm run typecheck` passed.
  - `npm run build:client` passed.
  - `npm run lc0:browser-gpu-legal-parity -- --fixture-limit 1 --top-k 8 --out /tmp/lc0-gpu-legal-parity-item1.json --timeout 180000` passed.
  - `npm run lc0:browser-gpu-legal-parity -- --fixture-limit 3 --top-k 16 --out /tmp/lc0-gpu-legal-parity-item1-limit3.json --timeout 240000` passed: best 3/3, top16 3/3, max prior drift 0.0002193465, compact estimate 204B vs full 3084B.

2026-06-08 item 2:
- Ran short alternating JS-legal vs GPU-legal b4/depth1 prototype matrix with existing opt-in readback matrix.
- Command: `npm run lc0:browser-readback-depth1-b4-matrix -- --out /tmp/lc0-readback-depth1-b4-item2.json --max-positions 2 --repeats 1 --timeout 240000`.
- Result artifact: `/tmp/lc0-readback-depth1-b4-item2.json`.
- Summary: `wgsl-pipe1` median 179.8 visits/s, search readback bytes ~27295; `wgsl-gpu-legal` median 138.6 visits/s, search readback bytes ~11308.
- Interpretation: GPU legal path reduces bytes but remains slower in this short nearby-control run; no promotion.

2026-06-08 item 3:
- Reconfirmed `mixed-tvm-ffn-smolgen-project` instead of assuming the historical 4.863 ms/eval lane still applies to current WGSL-head b4/depth1.
- WGSL-head run: `LC0_AR_RUN_ID=rsproj1 ... LC0_AR_ENCODER_KERNEL=mixed-tvm-ffn-smolgen-project bash autoresearch.sh.disabled-local` produced `5.978 ms/eval` but failed drift (`f32 top-prior max abs diff 0.0231 > 0.01`). Artifacts: `/tmp/lc0_autoresearch/rsproj1_rep1.json`, `/tmp/lc0_autoresearch/rsproj1_drift.json`.
- ORT-head check: `rsprojort1` passed 3-fixture drift but was slow (`11.852 ms/eval`), so it does not support WGSL-head b4 promotion. Artifacts: `/tmp/lc0_autoresearch/rsprojort1_rep1.json`, `/tmp/lc0_autoresearch/rsprojort1_drift.json`.
- Nearby WGSL-head control: `LC0_AR_RUN_ID=rffnctrl1 ... LC0_AR_ENCODER_KERNEL=mixed-tvm-ffn bash autoresearch.sh.disabled-local` passed drift and measured `5.804 ms/eval`. Artifacts: `/tmp/lc0_autoresearch/rffnctrl1_rep1.json`, `/tmp/lc0_autoresearch/rffnctrl1_drift.json`.
- Updated docs to keep `mixed-tvm-ffn` as the current documented WGSL-head b4 lane and demote smolgen-project evidence to historical until a new cleaned full pass restores it.

2026-06-08 item 4:
- Added agent-browser session-name sanitization/truncation with SHA-1 suffix in `scripts/lc0_browser_runtime_fixed_suite.mjs` and `scripts/lc0_browser_hybrid_search_fixture_parity.mjs`.
- Also used short session sanitization in the new GPU legal parity launcher.
- Validation: fixed-suite `--help` loads with `node --experimental-strip-types`; fixture parity long-session `--dry-run` succeeds.

2026-06-08 item 5:
- Added canonical safe protocol doc `docs/lc0_b4_depth1_safe_benchmark_protocol.md`.
- Protocol covers wrapper-owned browser cleanup hygiene, b4/depth1 fixture smoke, cleaned fixed-suite confirmation, drift gates, optional readback attribution, and closeout checks.
- Linked it from `docs/lc0web_custom_inference_checkpoint.md`.
