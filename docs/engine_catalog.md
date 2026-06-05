# Browser engine catalog and onboarding process

Last updated: 2026-06-05

This is the working catalog for engine families exposed by `lc0-arena.html` and `lc0-analysis.html`. It records what we adapted, which artifacts/variants the UI can select, how complete the browser integration is, and where to resume optimization later.

Status labels:

- **Stable default**: used by default for normal UI flows.
- **Experimental**: selectable, useful for smoke/benchmark work, but not a default or release promise.
- **Benchmark-only**: exists to measure a hypothesis, not for interactive play/analysis.
- **Intake candidate**: documented upstream target, not yet built or exposed in UI.
- **Proof-build only**: reproducible local build/smoke exists, but no browser adapter/UI selector yet.
- **Browser-smoke only**: browser adapter and smoke page exist, but no arena/analysis selector yet.
- **Local asset**: generated or symlinked locally; do not assume the blob is committed.

## Engine card schema

Every engine family should have one card with these fields:

| Field | Meaning |
| --- | --- |
| Family id / UI label | Stable id used in UI state and a human label. |
| Upstream/source version | Package version, upstream commit, model checkpoint, or exact source archive. |
| License/distribution | License obligations and whether generated artifacts may be committed. |
| Integration status | Stable/experimental/benchmark-only plus current UX scope. |
| Runtime adapter | Worker/API style: ORT/WebGPU, UCI worker, WASI one-shot, persistent WASI, direct browser API, etc. |
| Variants | UI-selectable variants, URLs, feature flags, default/fallback behavior. |
| Strength knob | Visits/depth/movetime/skill and supported ranges. |
| Feature parity | Move generation, analysis/MultiPV, hash/new-game reset, stop/abort behavior, threads, NNUE/model loading. |
| Artifact footprint | Raw sizes and cacheability; note external model/NNUE splits. |
| Speed snapshot | Latest meaningful benchmark link and headline numbers. |
| Validation | Smoke/bench/test commands and browser pages used. |
| Open work | The next tuning/onboarding tasks and known caveats. |

## Current family matrix

| Family | Variants in UI | Default posture | Main adapter | Version/source anchor | Footprint headline | Current note |
| --- | --- | --- | --- | --- | --- | --- |
| Lc0 | `small`, `bt4` | `small` stable; `bt4` gated/cautious | ORT/WebGPU/WASM worker + custom lc0web pack paths | Small `t1-256x10-distilled-swa-2432500`; BT4 `BT4-1024x15x32h-swa-6147500` | Small f32 ONNX 80.9 MB or f16 pack ~40.7 MB; BT4 f16 ONNX 370.6 MB | Our browser-native neural/search lane. Most runtime variants are still benchmark-gated. |
| Stockfish | `lite`, `full` | Lite single-thread default opponent | NPM `stockfish` JS/WASM UCI worker | `stockfish@18.0.7` | Lite WASM ~7 MB; full WASM ~108 MB | Mature UCI baseline; strongest/large full variant is optional. |
| Reckless | `simd`, `full`, `browser-api`, `browser-api-simd`, `browser-api-simd-external`, `lite` | SIMD if available; scalar fallback | Patched Rust `wasm32-wasip1` UCI; optional direct browser API | Upstream `codedeliveryservice/Reckless` commit `0010617448bd` + local patches | Integrated full WASM ~62 MB; external API WASM ~1.2 MB + NNUE ~60 MB | Best current non-Stockfish browser engine candidate; SIMD WASI/UCI is fastest proven path. |
| Viridithas | `default`, `simd` | Experimental opt-in | Patched Rust `wasm32-wasip1` UCI, one-shot/persistent/batch | Upstream `cosmobobak/viridithas` commit `20d7402065ca` + v106 `atlantis-b800.nnue.zst` + local patches | WASM ~55 MB with compressed NNUE embedded | Integration works for shallow arena/analysis, but stop/abort and throughput remain experimental. |
| Berserk | `emscripten` (WASI `default`/`simd` planned only) | Experimental opt-in | Patched single-thread Emscripten UCI worker; WASI still unpromoted | Upstream `jhonnold/berserk` tag `14` commit `8ae895a6151695be4a50d4fb65b0c131659c513a` + network `berserk-9b84c340af7e.nn` | Emscripten emits JS glue + ~128 KB WASM + ~24 MB preload data; generated/local only | Strong GPL C UCI candidate; staged UI integration is experimental while lifecycle/benchmark data accumulates. |
| PlentyChess | `emscripten` | Experimental opt-in | Patched single-thread Emscripten UCI worker | `Yoshie2000/PlentyChess` commit `58d8ba2505ae2b49f48dd410d214a457d15c12c6` + network `0134-2r24-s0.bin` | JS ~71 KB + WASM ~390 KB + data/processed NNUE ~63 MB; generated/local only | Node/browser lifecycle smoke and arena/analysis selector smoke pass; depth-7 rotated-FEN benchmark is ~718k NPS; large sidecar and GPL policy keep it experimental. |

## Family cards

### Lc0 family

- **Family id / UI label:** `lc0` / `Lc0`.
- **Integration status:** stable for the small model path; BT4 is gated by WebGPU support and a one-time memory/download warning.
- **Source/version anchors:**
  - Small model: `t1-256x10-distilled-swa-2432500`.
  - Stable browser pack: `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json`.
  - BT4 model: `/models/lc0/BT4-1024x15x32h-swa-6147500.batch1.f16.onnx`.
  - Model manifest: `public/models/lc0/manifest.json` with bytes and sha256 values.
- **Runtime adapter:**
  - Small/default: `searchWorker.ts` with ORT and the stable `lc0web-wgsl-encoder-ort-heads` path when requested.
  - Experimental LC0 runtime lanes include WGSL heads, WASM input encoder, WASM/GPU legal priors, deferred readback, TVM-packed kernels, and pipeline-depth experiments. They are not normal UI defaults.
  - BT4: lazy dedicated `Bt4WorkerSearcher`, WebGPU-only by policy, disposable worker per searcher.
- **UI variants:**
  - `small`: label `Lc0`, strength unit `visits`.
  - `bt4`: label `Lc0 BT4`, strength unit `visits`, disabled until `probeBt4Support()` succeeds.
- **Strength knob:** visits in arena/analysis; movetime paths exist in lower-level search APIs but the staged UI currently uses visits.
- **Artifact footprint:**
  - f32 batch1 ONNX: 80,895,900 bytes.
  - f16 lc0web pack: metadata 296,585 bytes + shards 40,418,450 bytes.
  - BT4 f16 ONNX: 370,635,179 bytes (~353 MiB UI warning).
  - Helper WASM: `lc0_input_encoder.wasm` ~6.5 KB; `lc0_legal_priors.wasm` ~16 KB.
- **Feature parity:**
  - Move generation/search: yes for normal browser play/search.
  - Analysis: yes; MultiPV support via search result formatting.
  - Abort: supported through worker cancellation for LC0 search paths; BT4 has explicit `cancel()` and worker disposal.
  - Threads: WebGPU/ORT worker model, not UCI thread controls.
  - Model caching: Cache API/local symlink flow exists; BT4 warns because of size.
- **Speed snapshot:** see `docs/lc0web_custom_inference_checkpoint.md`. Representative final local 32 visits/batch 4: stable ORT heads ~65 visits/s; experimental WGSL heads ~110 visits/s, but experimental paths stay opt-in until parity/lifecycle gates are stronger.
- **Validation:** `npm run lc0:browser-ci-smoke`, LC0 fixture parity tests, model manifest sha256 checks, and browser search matrix artifacts.
- **Open work:** keep extracting shared catalog metadata into UI config; keep experimental LC0 runtime lanes behind explicit params until repeated parity, leak, and cross-host latency gates pass.

### Stockfish family

- **Family id / UI label:** `sf` / `Stockfish`.
- **Integration status:** stable baseline/opponent.
- **Source/version anchor:** NPM package `stockfish@18.0.7` from `package-lock.json`.
- **License/distribution:** GPL-3.0 package. Treat vendored/symlinked assets as third-party distribution artifacts.
- **Runtime adapter:** JS worker UCI protocol via `StockfishEngine`.
- **UI variants:**
  - `lite`: `Stockfish Lite`, URL `/stockfish/stockfish-18-lite-single.js` by default.
  - `full`: `Stockfish`, URL `/stockfish/stockfish-18-single.js` in current arena/analysis wiring.
  - Threaded flavors exist (`lite-threaded`, `threaded`) and require cross-origin isolation; arena can cap requested threads to 1 when isolation is unavailable.
- **Strength knob:** depth by default; `movetimeMs`, `skillLevel`, and `threads` are supported in `StockfishOptions`. Staged UI uses depth.
- **Artifact footprint:**
  - Lite single WASM: ~7.0 MB plus ~20 KB JS wrapper.
  - Lite threaded WASM: ~6.8 MB plus ~31 KB JS wrapper.
  - Full single/threaded WASM: ~108 MB plus wrapper.
- **Feature parity:**
  - Move search: yes.
  - Analysis/MultiPV: yes through parsed UCI `info ... multipv` lines.
  - Stop/abort: worker can be terminated/recreated on error; UCI stop handling is mature enough for UI use.
  - New game/hash reset: standard UCI path.
  - Threads: supported by Stockfish, browser availability depends on SAB/cross-origin isolation and flavor.
- **Speed snapshot:** not cataloged in a current Stockfish-specific browser benchmark doc yet. Use it primarily as the known-strong baseline; add a benchmark card before tuning UI defaults around full/threaded variants.
- **Validation:** existing arena/analysis smokes and `StockfishEngine` parsing tests where present.
- **Open work:** document full/threaded benchmark numbers under the same protocol as Reckless/Viridithas; decide whether threaded variants belong in staged selectors or remain URL/config-only.

### Reckless family

- **Family id / UI label:** `reckless` / `Reckless`.
- **Integration status:** SIMD WASI/UCI is the strongest current browser-native candidate; browser API and lite variants remain experimental.
- **Source/version anchor:** upstream `codedeliveryservice/Reckless` commit `0010617448bd` in local build dirs, with local WASI/browser patches. Full artifacts are v60; lite candidate uses v53 `v53-0ba42a8c.nnue` when available.
- **License/distribution:** upstream AGPL-3.0. Generated WASM/NNUE blobs are intentionally ignored; distribution must include corresponding source archives.
- **Runtime adapter:**
  - WASI/UCI one-shot fallback.
  - Persistent shared-stdin WASI worker when `SharedArrayBuffer` and cross-origin isolation are available.
  - Direct browser API variants bypass UCI text for structured calls, but remain experimental.
- **UI variants:**
  - `simd`: `Reckless Full SIMD`, `/reckless/reckless-simd128.wasm`; preferred default when wasm SIMD validates.
  - `full`: `Reckless Full scalar fallback`, `/reckless/reckless.wasm`.
  - `browser-api`: `/reckless/reckless-browser-api.wasm`.
  - `browser-api-simd`: `/reckless/reckless-browser-api-simd128.wasm`.
  - `browser-api-simd-external`: `/reckless/reckless-browser-api-simd128-external.wasm` + `/reckless/reckless-v60-7f587dfb.nnue`.
  - `lite`: `/reckless/reckless-v53-l1-512.wasm`, optional experimental smaller/weaker build.
  - `custom`: URL param escape hatch via `?recklessWasm=`.
- **Strength knob:** depth in UI; movetime supported by benchmark/runtime plumbing.
- **Artifact footprint:**
  - Full scalar/SIMD/browser-api integrated artifacts: ~62 MB each, because the NNUE is embedded.
  - External-NNUE browser API SIMD: ~1.2 MB WASM + ~60 MB cacheable NNUE.
  - Generated source archives are part of production/deploy obligations.
- **Feature parity:**
  - Move search: yes.
  - Analysis/MultiPV: yes via UCI-like info lines/direct API formatting.
  - Persistent process: yes in isolated browsers.
  - Hash/new-game reset: clear-hash/new-game path corrected and benchmarked.
  - Stop/abort: abort terminates/recreates worker; graceful in-search `stop` cannot preserve the persistent wasm process in the current single-threaded patch.
  - Syzygy/tablebases: disabled for browser/WASI builds.
- **Speed snapshot:** see `docs/reckless_browser_benchmarks.md`.
  - Corrected depth 7/8/9 run: SIMD WASI/UCI had ~2.6-2.7x engine-reported NPS over scalar at deeper depths.
  - Browser API SIMD did not beat SIMD WASI/UCI in corrected runs; keep as experimental/control-plane groundwork.
- **Validation:** `reckless-benchmark.html`, Reckless variant tests, one-shot/persistent browser benchmark JSON artifacts, asset HEAD checks in UI.
- **Open work:** improve graceful stop/persistent process control; keep browser API only if it earns latency or control-plane wins; finish deployment packaging for generated artifacts/source archives.

### Berserk family

- **Family id / UI label:** `berserk` / `Berserk`.
- **Integration status:** experimental opt-in. The Emscripten worker adapter passed Node and browser smokes and is available in staged arena/analysis selectors, but not as a default engine.
- **Source/version anchor:** upstream `jhonnold/berserk` release tag `14`, commit `8ae895a6151695be4a50d4fb65b0c131659c513a`; default branch HEAD observed at `27212a24c16d9e5f9bc9180a75264c1c632808bb` during intake. The browser port pins tag `14` unless a newer release is deliberately selected.
- **License/distribution:** GPL-3.0. Generated JS/WASM/data and copied NNUE assets must be treated as redistributable GPL engine artifacts with corresponding source/build instructions. Do not commit generated blobs until the project has an explicit release/source-archive policy for Berserk.
- **Runtime adapter:** current first smoke is `patches/berserk-emscripten.patch` + `npm run berserk:build-emscripten`, exporting `command()`, `isReady()`, and `isSearching()` like Stockfish.js. It disables tablebases and uses synchronous single-thread search to avoid pthread/SAB requirements. WASI remains unpromoted unless it becomes useful later.
- **UI variants:** defined in `src/lc0/berserkVariants.ts`.
  - `emscripten`: current smoked worker path, `/berserk/berserk-emscripten.js` + `/berserk/berserk-emscripten.wasm` + `/berserk/berserk-emscripten.data`.
  - `default`: planned scalar WASI/UCI candidate, expected URL `/berserk/berserk.wasm`; not exposed in normal staged selectors until it smokes.
  - `simd`: planned SIMD WASI candidate, expected URL `/berserk/berserk-simd128.wasm`; not exposed in normal staged selectors until it smokes.
  - `custom`: URL param escape hatch for local experiments via `?berserkJs=`/`?berserkData=` or future `?berserkWasm=` and optional `?berserkNnue=`.
- **Expected NNUE assets:** upstream makefile uses `MAIN_NETWORK = berserk-9b84c340af7e.nn` from `https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn`. The Emscripten smoke preloads this network into `berserk-emscripten.data`; `berserkVariants.ts` still models external `/berserk/berserk-9b84c340af7e.nn` for any future WASI/custom path.
- **Strength knob:** depth in staged UI if promoted; movetime should be available in benchmarks once UCI `go movetime` is verified. Tentative defaults should mirror other alpha-beta engines until measured (`arena` shallow depth, `analysis` deeper depth).
- **Artifact footprint:** current local Emscripten smoke build is small JS glue + small WASM + ~24 MB `.data` NNUE preload. Record exact bytes from `npm run berserk:build-emscripten`/smoke reports and keep all generated `public/berserk/*.{js,wasm,data,nn,nnue}` assets ignored.
- **Feature parity to verify:**
  - UCI handshake: `uci` / `uciok`, `isready` / `readyok`.
  - Move search: `position startpos` and one non-startpos FEN with `go depth N`.
  - Analysis/MultiPV: upstream exposes `MultiPV`; verify info-line parsing before enabling analysis UI.
  - New game/hash reset: verify `ucinewgame`, `setoption name Hash`, and clear-hash behavior or document the fallback.
  - Stop/abort: upstream has UCI `stop` support natively, but browser/WASI behavior is unknown; first adapter may terminate/recreate the worker on abort until graceful stop is proven.
  - Threads/tablebases: upstream exposes `Threads` and `SyzygyPath`; browser intake should begin single-threaded and tablebases disabled.
  - NNUE/model loading: decide embedded vs external; missing asset path must be visible in UI before promotion.
- **Speed snapshot:** none yet. Use the Reckless/Viridithas rotated-FEN protocol after smoke, with cold/warm separation and engine-reported nodes/NPS.
- **Validation:** `npm run berserk:build-emscripten` then `npm run berserk:smoke-emscripten` verifies `uci`, `isready`, `ucinewgame`, startpos search, and one non-startpos FEN search in Node. `berserk-smoke.html` exercises the reusable `BerserkEngine` worker adapter in the browser; lifecycle smoke now covers repeated searches, MultiPV parsing, abort-by-worker-restart recovery, and missing-asset failure. Arena/analysis selector smokes and first sanity comparisons against Stockfish full single/Reckless SIMD are recorded in `docs/berserk_browser_benchmarks.md`. The reusable C/C++ intake process is captured in `docs/browser_c_engine_porting.md`.
- **Open risks:** graceful in-search stop without losing the resident worker, pthread/SIMD follow-up, GPL source/archive obligations, deeper benchmark comparison against Stockfish/Reckless/Viridithas, and whether Berserk offers a browser-speed advantage after startup is amortized.

### PlentyChess family

- **Family id / UI label:** proposed `plentychess` / `PlentyChess`.
- **Integration status:** experimental opt-in. Emscripten build, Node UCI smoke, reusable worker adapter, browser lifecycle smoke, first shallow benchmark, and arena/analysis selector smoke pass.
- **Source/version anchor:** initial reconnaissance used `https://github.com/Yoshie2000/PlentyChess.git` at commit `58d8ba2505ae2b49f48dd410d214a457d15c12c6` on the default branch. Current source reports `VERSION = "7.0.66"` and `network.txt = 0134-2r24-s0`. Latest GitHub release observed during intake was `b-v7.0.0` at commit `e2060ab4b3021babb7d74af3cbe908154c03b2fd`, but its `network.txt` value `0119` did not correspond to an immediately available `PlentyNetworks` asset under the Makefile's expected URL during this reconnaissance.
- **License/distribution:** GPL-3.0. Generated JS/WASM and embedded/processed NNUE assets must stay ignored until corresponding-source/archive policy covers PlentyChess as well as Berserk.
- **Runtime adapter:** current proof uses the standard C/C++ recipe in `docs/browser_c_engine_porting.md`: Emscripten, modular JS, exported `command(const char*)`, single-thread synchronous search, and tablebases inert. `src/lc0/plentychessEngine.ts` wraps it as a `BrowserUciEngine` worker adapter.
- **Expected NNUE assets:** default-branch Makefile downloads `https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin`. Reconnaissance observed a 57,557,991-byte raw net with sha256 `550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061`; upstream `tools/process_net` produced a local `processed.bin` of about 60 MB with sha256 `691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4`.
- **Porting notes:**
  - Native UCI loop blocked on `std::getline(std::cin, ...)`; `patches/plentychess-emscripten.patch` adds an Emscripten `command()` entry point like Berserk/Stockfish.js.
  - `ThreadPool::resize(1)` and `TT.clear()` spawned `std::thread`s; the patch adds a `PLENTY_SYNC_SEARCH` path that creates the single worker directly, runs `Worker::startSearching()` inline, makes `waitForSearchFinished()` trivial, and clears TT with one `memset`.
  - Fathom/Syzygy is included but browser tablebases are not shipped. The proof build compiles with `TB_NO_THREADS` and relies on empty tablebase state.
  - Upstream `incbin` assembler directives do not work for wasm, so the proof build preloads `processed.bin` into `.data` and reads it through Emscripten's filesystem at startup.
  - SIMD assumptions are stronger than Berserk. The proof build currently uses Emscripten's wasm SIMD lowering through `-DARCH_X86 -msimd128 -mssse3`; this is acceptable for proof smoke but still needs browser lifecycle and benchmark validation.
- **Strength knob:** depth first; movetime later once UCI smoke is stable. Keep experimental even if fast because lifecycle and artifact policy are unknown.
- **Artifact footprint:** first local proof build emitted `plentychess-emscripten.js` 70,886 bytes, `plentychess-emscripten.wasm` 389,983 bytes, and `plentychess-emscripten.data` 63,023,936 bytes. Generated files remain ignored under `public/plentychess/`.
- **Validation:** `npm run plentychess:build-emscripten` then `npm run plentychess:smoke-emscripten` verifies `uci`, `isready`, `ucinewgame`, startpos search (`bestmove c2c4`), one non-startpos FEN (`bestmove e1g1`), and post-search `readyok` in Node. `plentychess-smoke.html` verifies browser worker prewarm, repeated searches, MultiPV, abort/recovery, and missing-asset failure surfacing. `reckless-benchmark.html` includes a PlentyChess checkbox; first depth-7 rotated-FEN benchmark measured ~9.35 ms/search, ~4,068 nodes/search, and ~718k engine-reported NPS. Arena selector smoke passed on K-v-K vs Stockfish Lite; analysis selector smoke rendered depth-8 startpos PVs. Details are in `docs/plentychess_browser_port.md`.
- **Open risks:** source pin choice (`main` vs release), missing/stale release-network mapping, single-thread Emscripten patch size, tablebase guards, SIMD/scalar fallback correctness, large GPL artifact packaging, and whether the engine is meaningfully distinct from the existing Stockfish/Reckless speed-strength lane in browser form.

### Viridithas family

- **Family id / UI label:** `viridithas` / `Viridithas`.
- **Integration status:** experimental. Arena and analysis can use it, but it should not be presented as production-ready.
- **Source/version anchor:** upstream `cosmobobak/viridithas` pinned to commit `20d7402065cae084715183e019fdd18089e2dfac`; build script downloads v106 `atlantis-b800.nnue.zst`; local patch is `patches/viridithas-wasip1.patch`.
- **License/distribution:** treat generated WASM and downloaded NNUE as third-party engine assets; generated `public/viridithas/*.wasm` is ignored.
- **Runtime adapter:** patched Rust `wasm32-wasip1` UCI.
  - One-shot argv mode.
  - Benchmark-only batch one-process mode.
  - Persistent shared-stdin WASI worker in isolated browsers.
- **UI variants:**
  - `default`: `Viridithas scalar experimental`, `/viridithas/viridithas.wasm`.
  - `simd`: `Viridithas SIMD experimental`, `/viridithas/viridithas-simd128.wasm`.
  - `custom`: URL param escape hatch via `?viridithasWasm=`.
- **Strength knob:** depth in UI and benchmarks; movetime benchmark plumbing exists.
- **Artifact footprint:** scalar and SIMD WASM artifacts are ~55 MB each with compressed network embedded.
- **Feature parity:**
  - Move search: yes for sequential searches and shallow arena games.
  - Analysis: yes through Stockfish/Reckless-style UCI info formatting.
  - Persistent process: works for sequential searches when SAB/cross-origin isolation are available.
  - Stop/abort: currently terminates the worker rather than graceful in-search `stop`; native search-time stdin polling is disabled in wasm search, so robust interactive stop remains open.
  - Threads: native threadpool is replaced/flattened for wasm browser use.
  - NNUE caching: native temp-file/mmap cache is bypassed on wasm; decompressed network is leaked for process lifetime.
- **Speed snapshot:** see `docs/viridithas_browser_benchmarks.md`.
  - Focused SIMD run showed engine-reported NPS gains around 3.4x-12.0x over scalar depending on budget.
  - Versus Reckless, Viridithas one-shot was much slower by wall time and NPS; batch/persistent probes show startup/decompression amortization is important.
- **Validation:** `npm run viridithas:smoke`, browser persistent smoke JSON, arena smoke games, benchmark JSON artifacts.
- **Open work:** graceful stop/abort, larger correctness gauntlet, repeated persistent benchmarks, direct browser API feasibility only if Viridithas remains interesting after SIMD/persistence tuning.

## Onboarding process for a new engine family

Use this checklist before adding a family to the staged selectors.

1. **Create the card first.** Add a section to this catalog with upstream repo, exact commit/release/model/net, license, expected artifact URLs, and initial status set to experimental.
2. **Pin and reproduce source.** Add a build script or documented package source that can recreate the browser artifact from a pinned commit/version. Record local patch files and environment variables.
3. **Choose the browser adapter deliberately.** Prefer the smallest viable path:
   - Existing browser JS/WASM worker if upstream ships one.
   - WASI/UCI one-shot for first smoke.
   - Persistent WASI only when startup/model load dominates and SAB isolation is available.
   - Direct browser API only when UCI overhead/control limitations are proven blockers.
4. **Define variants in one place.** Add variant metadata with `key`, `label`, artifact URL(s), backend kind, note, asset check behavior, and default/fallback rule. Keep UI labels and URL params stable.
5. **Add strength semantics.** Decide whether the UI knob means visits, depth, movetime, nodes, or skill. Clamp ranges and document default values.
6. **Implement lifecycle contract.** Each adapter should implement `src/lc0/browserUciEngine.ts` and follow `docs/browser_uci_adapter_contract.md`: `prewarm()`, `search()`/`bestMove()`, `analyze()`, `newGame()`, stop/abort behavior, `dispose()`, runtime status/labels, asset checks, and missing-asset UI state.
7. **Add asset policy.** Decide whether blobs are committed, symlinked, generated locally, or published by deployment. For GPL/AGPL engines, document source/archive obligations next to the build script.
8. **Smoke before UI promotion.** Minimum: startpos bestmove, one non-startpos bestmove, abort/dispose behavior, asset-missing UI path, and one shallow arena or analysis run.
9. **Benchmark with a shared protocol.** Use rotated multi-FEN suites, cold + warm separation, persistent hash clearing/new-game reset, depth and movetime budgets where applicable, raw JSON output, and browser runtime metadata.
10. **Gate promotion.** Do not move from experimental to default until the card has: repeatable build, clean asset status, UI smoke, fixture/arena correctness where relevant, speed/size numbers, lifecycle cleanup, and explicit stop/abort behavior.
11. **Keep the card fresh.** When an upstream release lands, add a dated note: old version, new version, changed assets, migration risk, benchmark delta, and whether the UI default changed.

## Catalog-to-UI source of truth

The staged selectors now consume the shared typed catalog in `src/lc0/engineCatalog.ts` for:

- family ids, labels, UI order, and doc anchors;
- static Lc0/Stockfish variant lists and BT4 gating metadata;
- arena vs analysis strength metadata/defaults;
- shared row labels for Lc0 and Stockfish;
- the `EngineFamily` / `EngineRow` types used by arena and analysis.

Reckless, Viridithas, and the pre-UI Berserk intake keep their dynamic artifact metadata in `recklessVariants.ts`, `viridithasVariants.ts`, and `berserkVariants.ts` because those modules own URL-param custom variants, asset checks, and runtime-specific defaults. If a future engine uses similar generated artifacts, add a dedicated variant module and then surface its family/static selector facts through `engineCatalog.ts`.

Any UI variant/default change should update both `src/lc0/engineCatalog.ts` and this card in the same commit. `tests/engine_catalog.test.mjs` pins the current selector order, static variants, family guard, and arena/analysis strength defaults.
