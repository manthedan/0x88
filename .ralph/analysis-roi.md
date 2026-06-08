# Analysis ROI Loop

Work through the five highest-ROI post-merge tasks one item at a time. Keep stable defaults unchanged, keep LC0 WGSL/WASM paths opt-in, and record verification for each item.

## Goals
- Add reliable automated browser smoke coverage for lc0-analysis.html.
- Stabilize the LC0 autoresearch/browser harness timeouts before further scheduler micro-optimization.
- Harden browser engine asset availability flow and documentation.
- Polish local PGN DB workflow and error/empty states.
- Improve engine profile UX without persisting custom URL variants yet.

## Checklist
- [x] 1. Add automated browser smoke for lc0-analysis.html: assert engine family selector, profile selector, PGN DB controls, runtime info, and clean console.
- [x] 2. Fix/stabilize LC0 autoresearch/browser harness timeouts for hybrid-wgsl-heads at lc0BatchSize=4 and batchPipelineDepth=1.
- [ ] 3. Harden asset availability flow for BT4/Reckless/Viridithas/Berserk/PlentyChess with one documented prep/check path.
- [ ] 4. Polish PGN DB workflow: saved collection list, search affordances, backup/restore confidence, empty/error states.
- [ ] 5. Improve engine profile UX: better built-in presets/summaries/import-export; do not persist custom URL variants.

## Verification
- Item 1: Added `scripts/lc0_analysis_browser_smoke.mjs` plus `npm run lc0:analysis-browser-smoke`.
- Item 1 smoke passed: `npm run lc0:analysis-browser-smoke -- --timeout 45000 --out /tmp/lc0-analysis-browser-smoke.json` returned `LC0_ANALYSIS_BROWSER_SMOKE_DONE`, found all engine families (`lc0`, `tiny`, `sf`, `reckless`, `viridithas`, `berserk`, `plentychess`), found engine profile controls, found PGN DB controls, runtime text included all expected engine/runtime sections, and `consoleErrors: []`.
- Item 2: Hardened `scripts/lc0_browser_hybrid_search_fixture_parity.mjs` to poll `#benchResult` progress, fail with the last progress text on total/progress timeout, and fail early if the auto-started strict-port Vite server exits before readiness. This avoids silently measuring a stale/wrong server when the default port is already occupied.
- Item 2 target smoke passed from this worktree: `npm run lc0:browser-hybrid-search-fixture-b4-wgsl-smoke -- --out /tmp/lc0-b4-wgsl-fixture-smoke.json` returned `HYBRID_SEARCH_FIXTURE_PARITY_DONE`, `cells: 1`, `nativeMatches: 1`, `depthBaselineMatches: 1`, `maxDepthBaselineVisitL1: 0`, `mismatches: []` for `headBackend=wgsl`, `inputBackend=wasm`, `batch=4`, `batchPipelineDepths=1`.

## Notes
- Ralph extension is installed, but this current Pi session does not expose ralph_start/ralph_done tools until reload. This task file is compatible with /ralph start/resume after reload; proceeding manually one item at a time in this session.
- Item 1 complete.
- Item 2 complete. Proceed to item 3 next: one documented prep/check path for BT4/Reckless/Viridithas/Berserk/PlentyChess asset availability.
