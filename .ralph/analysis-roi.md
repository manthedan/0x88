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
- [x] 3. Harden asset availability flow for BT4/Reckless/Viridithas/Berserk/PlentyChess with one documented prep/check path.
- [x] 4. Polish PGN DB workflow: saved collection list, search affordances, backup/restore confidence, empty/error states.
- [x] 5. Improve engine profile UX: better built-in presets/summaries/import-export; do not persist custom URL variants.

## Verification
- Item 1: Added `scripts/lc0_analysis_browser_smoke.mjs` plus `npm run lc0:analysis-browser-smoke`.
- Item 1 smoke passed: `npm run lc0:analysis-browser-smoke -- --timeout 45000 --out /tmp/lc0-analysis-browser-smoke.json` returned `LC0_ANALYSIS_BROWSER_SMOKE_DONE`, found all engine families (`lc0`, `tiny`, `sf`, `reckless`, `viridithas`, `berserk`, `plentychess`), found engine profile controls, found PGN DB controls, runtime text included all expected engine/runtime sections, and `consoleErrors: []`.
- Item 2: Hardened `scripts/lc0_browser_hybrid_search_fixture_parity.mjs` to poll `#benchResult` progress, fail with the last progress text on total/progress timeout, and fail early if the auto-started strict-port Vite server exits before readiness. This avoids silently measuring a stale/wrong server when the default port is already occupied.
- Item 2 target smoke passed from this worktree: `npm run lc0:browser-hybrid-search-fixture-b4-wgsl-smoke -- --out /tmp/lc0-b4-wgsl-fixture-smoke.json` returned `HYBRID_SEARCH_FIXTURE_PARITY_DONE`, `cells: 1`, `nativeMatches: 1`, `depthBaselineMatches: 1`, `maxDepthBaselineVisitL1: 0`, `mismatches: []` for `headBackend=wgsl`, `inputBackend=wasm`, `batch=4`, `batchPipelineDepths=1`.
- Item 3: Added `scripts/check_browser_engine_assets.mjs`, `npm run engine-artifacts:check-browser`, and `docs/engine_catalog.md` fast-path docs. The checker reports BT4/Reckless/Viridithas/Berserk/PlentyChess same-origin public assets, bytes present, missing URL paths, and exact prep/build commands.
- Item 3 check passed in audit mode: `npm run engine-artifacts:check-browser -- --allow-missing` completed and correctly reported local missing `lc0` BT4 and `reckless` assets while confirming `viridithas`, `berserk`, and `plentychess` assets present. JSON subset artifact written with `npm run engine-artifacts:check-browser -- --only viridithas,berserk,plentychess --json --allow-missing > /tmp/browser-engine-asset-check.json`.
- Item 4: Added visible PGN DB collection cards (`#pgnDbList`), position-search hit cards (`#pgnDbSearchResults`), better empty states, clickable collection selection, and backup/import messages that state raw PGN is authoritative and position indexes are rebuildable/rebuilt.
- Item 4 browser smoke passed: `npm run lc0:analysis-browser-smoke -- --timeout 45000 --out /tmp/lc0-analysis-browser-smoke-pgn-polish.json` found `#pgnDbList`/`#pgnDbSearchResults` and no console errors. Manual browser PGN flow on port 5198 saved `pi pgn polish smoke`, rendered collection-card text, searched start position, rendered hit-card text `e4 1`, and cleanup removed the smoke IndexedDB record. Targeted tests passed: `node --experimental-strip-types --test tests/pgn_database.test.mjs tests/lc0_opening_stats.test.mjs`.
- Item 5: Added built-in profile presets (`Lc0 + Stockfish`, `Browser-native survey`, `LC0 WGSL heads probe`), profile summaries, saved-profile export/import JSON, and kept custom URL variants excluded through existing row sanitization and save checks.
- Item 5 browser smoke passed: `npm run lc0:analysis-browser-smoke -- --timeout 45000 --out /tmp/lc0-analysis-browser-smoke-profiles.json` found the built-in profile options, import/export controls, profile summary, and no console errors. Manual browser selection of `builtin:lc0-wgsl-heads` on port 5199 set runtime to `hybrid-wgsl-heads`, kept one LC0 row, and showed the opt-in/stable-defaults summary.
- Final loop validation passed: `npm run typecheck`; `node --experimental-strip-types --test tests/lc0_analysis_format.test.mjs tests/pgn_database.test.mjs tests/lc0_opening_stats.test.mjs tests/engine_catalog.test.mjs`; `npm run build:client`; `npm run lc0:analysis-browser-smoke -- --timeout 45000 --out /tmp/lc0-analysis-browser-smoke-final.json`; `npm run lc0:browser-hybrid-search-fixture-b4-wgsl-smoke -- --out /tmp/lc0-b4-wgsl-fixture-smoke-final.json`; `npm run engine-artifacts:check-browser -- --allow-missing`; `git diff --check`.

## Notes
- Ralph extension is installed, but this current Pi session does not expose ralph_start/ralph_done tools until reload. This task file is compatible with /ralph start/resume after reload; proceeding manually one item at a time in this session.
- Item 1 complete.
- Item 2 complete.
- Item 3 complete.
- Item 4 complete.
- Item 5 complete. All loop items complete.
