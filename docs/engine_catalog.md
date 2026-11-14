# Browser engine catalog and onboarding process

Non-Centipawn cards checked against the current tree: 2026-11-12. Centipawn updated for the 2026-11-13 s4fix EMA bundle.

This is the working catalog for engine families exposed by `/app/arena/` and `/app/analysis/`. It records what we adapted, which artifacts/variants the UI can select, how complete the browser integration is, and where to resume optimization later.

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
| Lc0 | `small`, `t3`, `bt4` | `small` stable; large nets explicit/gated | ORT/WebGPU/WASM worker + explicit research runtimes | Small `t1-256x10-distilled-swa-2432500`; t3 `distill-swa-2767500`; BT4 `policytune-332` | Small b1 f16 QDQ8; t3 b8 f16 ~163 MB; BT4 b8 f16 QDQ8 ~178 MB | Apple-only split-K selection is model-specific; see `lc0SmallModelSelection.ts` and dated outcome records. |
| Centipawn | `bt4-auto`, `bt4-ort` | ONNX Runtime Web only | Centipawn SquareFormer evaluator/search path | s4fix V EMA, eight layers | Float32 dynamic-batch model, 6.2 MB | Persisted compiled selectors fall back to ORT; legacy `tiny` deep-links remain accepted. |
| Stockfish | `lite`, `full` | Lite single-thread default opponent | NPM `stockfish` JS/WASM UCI worker | `stockfish@19.0.1` | Lite WASM ~1.8 MB; full WASM ~94 MB | Mature UCI baseline; strongest/large full variant is optional. |
| Reckless | `simd`, `relaxed-simd`, `full`, `browser-api`, `browser-api-simd`, `browser-api-simd-external`, `wasi-simd-external`, `lite` | Relaxed SIMD if available; fixed SIMD fallback; scalar final fallback | Patched Rust `wasm32-wasip1` UCI; optional direct browser API | Upstream `codedeliveryservice/Reckless` commit `0010617448bd` + local patches | Integrated full WASM ~62 MB; external API WASM ~1.2 MB + NNUE ~60 MB | Best current non-Stockfish browser engine candidate; the default speed ladder is relaxed SIMD > fixed SIMD > scalar WASI/UCI. |
| Viridithas | `relaxed-simd`, `simd`, `default` | Relaxed SIMD when the browser validates it; SIMD, then scalar | Patched Rust `wasm32-wasip1` UCI, one-shot/persistent/batch | Upstream `cosmobobak/viridithas` tag `v20.0.0` commit `0631113e` + v109 `sandhi-s2-b200.nnue.zst` + local patches | WASM ~53 MB with compressed NNUE embedded | Persistent cooperative cancellation is implemented with a watchdog; one-shot abort terminates. The family remains experimental. |
| Berserk | `emscripten`, `emscripten-simd`, `emscripten-relaxed` (WASI planned only) | Relaxed → SIMD → scalar; family experimental | Patched single-thread Emscripten UCI worker; WASI still unpromoted | Upstream `jhonnold/berserk` tag `14` commit `8ae895a6151695be4a50d4fb65b0c131659c513a` + network `berserk-9b84c340af7e.nn` | JS glue + ~0.13–0.14 MB WASM per tier + one shared ~24 MB preload; tracked/deployable with manifest and corresponding source | Shipped relaxed tier uses `-O2 -flto`; see the dated SIMD and compiler-flag records. |
| PlentyChess | `emscripten`, `emscripten-sse41`, `emscripten-relaxed` | Relaxed → SSE4.1 → base; family experimental | Patched single-thread Emscripten UCI worker | `Yoshie2000/PlentyChess` tag `b-v8.0.0`, commit `50ad9f1b523063cbbbf599fe0547f42870fe2906` + network `0178r.bin` | JS ~71 KB + WASM ~392 KB + shared processed NNUE ~53 MB raw; release artifacts use LFS | See `docs/plentychess_upgrade_v8.md` for current native/WASM parity, strength, browser NPS, and compiler-flag gates. |
| Stormphrax | `emscripten`, `emscripten-relaxed` | Relaxed → baseline SIMD; family experimental | Patched single-thread Emscripten UCI worker with wasm SIMD | `Ciekce/Stormphrax` tag `v8.0.0`, commit `582965517ed2032d41a6b4cd6c2e66b1b934e2ad` + `undertown.nnue` | JS ~71 KB + WASM ~697 KB + data 55.9 MB raw | Node UCI smoke passes; available in Play, Arena, and Analysis with depth/movetime and MultiPV. |

## Asset prep/check fast path

Use one local check before browser analysis/arena smokes when optional engine assets may be missing:

```bash
npm run engine-artifacts:check-browser -- --allow-missing
```

The check reports every same-origin browser asset used by the staged BT4/Reckless/Viridithas/Berserk/PlentyChess selectors and prints the prep/build command for each missing family. To make missing assets fail a release/local packaging gate, omit `--allow-missing`:

```bash
npm run engine-artifacts:check-browser
```

Common prep commands:

```bash
npm run lc0:prepare-model-assets
npm run reckless:build-release && npm run reckless:build-lite-wasi
npm run viridithas:build-wasi && npm run viridithas:build-simd-wasi && npm run viridithas:build-relaxed-simd-wasi
npm run berserk:build-emscripten && npm run berserk:build-simd-emscripten && npm run berserk:build-relaxed-simd-emscripten
npm run plentychess:build-emscripten && npm run plentychess:build-sse41-emscripten && npm run plentychess:build-relaxed-simd-emscripten
npm run stormphrax:build-emscripten
```

Generated large/GPL/AGPL blobs remain local unless the release follows `docs/engine_artifact_distribution.md`; the browser UI must continue to treat missing assets as a selectable/runtime status, not as a silent failure.

## Family cards

### Lc0 family

- **Family id / UI label:** `lc0` / `Lc0`.
- **Integration status:** stable for Small; t3/BT4 are explicit large-net choices with capability/download warnings. Big nets have bounded WASM fallback levels; they are not WebGPU-only.
- **Source/version anchors:**
  - Small model: `t1-256x10-distilled-swa-2432500`.
  - Stable browser pack: `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json`.
  - Small ORT default: `/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx`; Apple WebGPU sessions may select its released split-K sibling.
  - t3: `/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx`; current source selects its smolgen split-K sibling on Apple WebGPU (publication status is in `docs/bignet_smolgen_splitk_2026-11.md`).
  - BT4 model: `/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx` (`BT4-it332`). The older b1/b4/b8 unquantized exports remain manifest history, not the current default.
  - Model manifest: `public/models/lc0/manifest.json` with bytes and sha256 values.
- **Runtime adapter:**
  - Small/default: `searchWorker.ts` with ORT and the stable `lc0web-wgsl-encoder-ort-heads` path when requested.
  - Experimental LC0 runtime lanes include WGSL heads, WASM input encoder, WASM/GPU legal priors, deferred readback, TVM-packed kernels, and pipeline-depth experiments. The arena UI exposes the input backend, encoder kernel, and legal-prior backend as explicit LC0 opt-ins; their defaults remain JS input, hand WGSL encoder, and JS legal priors.
  - t3/BT4: lazy dedicated `Bt4WorkerSearcher`, disposable worker per searcher, WebGPU preferred with CPU fallback. `src/lc0/bt4Engine.ts` owns model identities, batch sizes and reduced WASM Play ladders.
- **UI variants:**
  - `small`: label `Lc0`, strength unit `visits`.
  - `t3`: label `Lc0 t3-512`, batch-8 f16 distill export.
  - `bt4`: label `Lc0 BT4-it332`, strength unit `visits`, capability-gated. Arena uses the policytune-332 batch-8 f16 QDQ8 export with tree reuse and batch-8 leaf evaluation, not unquantized batch 4.
  - Play also offers Leela Queen Odds v2 for odds games; it is not a general analysis net.
- **Strength knob:** visits in Analysis/Play; Arena supports fixed visits or equal movetime.
- **Artifact footprint:**
  - f32 batch1 ONNX: 80,895,900 bytes.
  - f16 lc0web pack: metadata 296,585 bytes + shards 40,418,450 bytes.
  - Historical BT4-it332 unquantized f16 ONNX: 370,635,164 bytes. Current b8 QDQ8 warning is ~178 MB; t3 is ~163 MB. Use the manifest for exact per-artifact bytes.
  - Helper WASM: `lc0_input_encoder.wasm` ~6.5 KB; `lc0_legal_priors.wasm` ~16 KB.
- **Feature parity:**
  - Move generation/search: yes for normal browser play/search.
  - Analysis: yes; MultiPV support via search result formatting.
  - Abort: supported through worker cancellation for LC0 search paths; BT4 has explicit `cancel()` and worker disposal.
  - Threads: WebGPU/ORT worker model, not UCI thread controls.
  - Model caching: Cache API/local symlink flow exists; BT4 warns because of size.
- **Speed snapshot:** the following is dated June custom-runtime evidence, not current ORT performance; see `docs/lc0web_custom_inference_checkpoint.md`. Released Small split-K and later ORT capture/IO outcomes are in `docs/lc0_small_splitk_2026-11.md` and `docs/ort_invoke_overhead_2026-11.md`. Representative final local 32 visits/batch 4: stable ORT heads ~65 visits/s; experimental WGSL heads ~110 visits/s. The 2026-06-05 recovered-state autoresearch lane favored the explicit opt-in `hybrid-wgsl-heads` + WASM input + `mixed-tvm-ffn` + JS legal priors + batch 4/depth 1 path at about `6.1 ms/eval` on the fixed suite after benchmark-harness cleanup; this is not a default and must be compared only against similarly cleaned controls.
- **Validation:** `npm run lc0:browser-ci-smoke`, LC0 fixture parity tests, model manifest sha256 checks, and browser search matrix artifacts.
- **Historical custom-runtime work ledger (2026-06):** shared catalog metadata is now centralized. Keep experimental runtime lanes opt-in until their gates pass; the following rejected schedules and measurements retain their original scope. After the naive WebGPU int8 FFN lane regressed throughput despite passing drift, the next high-ROI GPU lane is smolgen f16 tiling/generation. Historical opt-in evidence kept `mixed-tvm-ffn-smolgen-project`, now with 64-column smolgen project plus 8-lane smolgen dense1 with 2-step row-loop unroll, which confirmed at `4.863 ms/eval` on the cleaned mixed-TVM JS-legal 32-position fixed suite with 9-fixture drift passing; a short 2026-06-08 WGSL-head reconfirm was slower than nearby `mixed-tvm-ffn` and failed top-prior drift, so the current documented WGSL-head b4 lane remains `mixed-tvm-ffn` until smolgen-project gets a new cleaned full pass. Simple tiled smolgen-dense1, combined project+dense1, dense1+swish+LN1 fusion, row-split dense1 reduction, discarded dense1 shape/unroll/fma variants including the 3-step row-loop unroll that failed drift, paired-output/dense2-unroll variants, compress/head scalar unrolls/workgroup-size tweaks/packed-f16 buffers/linear-specialized shaders/value dense2+softmax fusion, hand QKV/output-projection larger K tiles, generated TVM `pick_lane` select rewrite, attention-score channel unroll, parallel score+softmax fusion, attention-value loop unrolls/workgroup reshapes, transposed-weight smolgen compress/dense1/dense2, paired-output smolgen project, smolgen dense2/compress workgroup-shape/reduction tweaks, mapped-policy 128-wide groups, 16/32/128/256-column project tiles, manual project-loop unrolling, project weight-index strength reduction, project final-barrier elision, shared-input smolgen dense2/compress, LN2 swish caching, project+TVM outproj/QKV, split-FFN TVM mixes, GPU/WASM legal priors, JS/WGSL input, and project+attention-score fusion attempts did not improve or failed drift, so future smolgen work needs a different generated/fused schedule with isolated parity first.

### Centipawn family

- **Family id / UI label:** `centipawn` / `Centipawn`.
- **Integration status:** served `s4fix-v-ema`; see `docs/model_provenance/s4fix_v_ema.md` for the contract and strength limits.
- **Runtime adapter:** ONNX Runtime Web only. Historical compiled graphs cannot run this model; persisted compiled selectors resolve to ORT with a fallback warning.
- **UI variants:** `bt4-auto`, `bt4-ort`. The stored `bt4-custom` key is still handled safely.
- **Naming note:** previously called "Tiny Leela". The legacy family key `tiny` is still accepted on input (deep-links such as `?engine=tiny`, seat specs, and the `tinyModel`/`tinyMeta`/`tinyManifest` overrides) via `canonicalEngineFamily`, and the hybrid manifest loader still accepts the legacy `tiny-leela.squareformer-tvm-hybrid.v1` schema. The shared NN-runtime infra (`tinyLeela*` timing/env/localStorage) keeps its name for now.

### Stockfish family

- **Family id / UI label:** `sf` / `Stockfish`.
- **Integration status:** stable baseline/opponent.
- **Source/version anchor:** NPM package `stockfish@19.0.1` from `package-lock.json`; `v19.0.1` / `0d5dd794`. See `stockfish_19_upgrade.md` for native parity and network provenance.
- **License/distribution:** GPL-3.0 package. Treat vendored/symlinked assets as third-party distribution artifacts.
- **Runtime adapter:** JS worker UCI protocol via `StockfishEngine`.
- **UI variants:**
  - `lite`: `Stockfish Lite`, URL `/stockfish/stockfish-19-lite-single.js` by default, or `/stockfish/stockfish-19-lite-single-relaxed.js` when the relaxed-SIMD probe passes.
  - `full`: `Stockfish`, URL `/stockfish/stockfish-19-single.js` in current arena/analysis wiring.
  - Threaded flavors exist (`lite-threaded`, `threaded`) and require same-origin cross-origin isolation. Relaxed full/threaded variants remain reproducible local candidates, but are not public selector artifacts yet.
- **Strength knob:** depth by default; `movetimeMs`, `skillLevel`, and `threads` are supported in `StockfishOptions`. Arena supports depth or equal movetime; Analysis uses depth.
- **Artifact footprint:**
  - Lite single WASM: ~1.76 MiB; npm JS ~28 KB, unminified relaxed JS ~159 KB.
  - Lite threaded WASM: ~1.62 MiB plus ~33 KB JS wrapper.
  - Full single/threaded WASM: ~94.5 MiB plus wrapper.
- **Feature parity:**
  - Move search: yes.
  - Analysis/MultiPV: yes through parsed UCI `info ... multipv` lines.
  - Stop/abort: worker can be terminated/recreated on error; UCI stop handling is mature enough for UI use.
  - New game/hash reset: standard UCI path.
  - Threads: supported by Stockfish, browser availability depends on SAB/cross-origin isolation and flavor.
- **Speed snapshot:** SF19 Chromium149 depth12 lite relaxed: 1.086M median aggregate wall NPS vs 0.863M baseline (1.273× paired median). Across-release NPS decreases; native strength and lite footprint improve. See `stockfish_19_upgrade.md` for spreads, startup, all tiers and full/lite strength. The older 2.34M/2.01M snapshot is historical SF18 evidence only.
- **Validation:** exact native bench signatures full453169/lite632754; all flag cells pass. Chromium passes 600/600 browser-vs-native depth12 comparisons; npm and custom lite/full pthread variants pass Threads2 smoke. Focused adapter tests include SF19 silent-startup failure rejection/retry.
- **Open work:** finish Safari fallback checks and decide whether full/threaded relaxed variants belong in staged selectors or remain local/config-only.

### Reckless family

- **Family id / UI label:** `reckless` / `Reckless`.
- **Integration status:** the relaxed SIMD > fixed SIMD > scalar WASI/UCI ladder is the strongest current browser-native candidate; browser API and lite variants remain experimental.
- **Source/version anchor:** upstream `codedeliveryservice/Reckless` commit `0010617448bd` in local build dirs, with local WASI/browser patches. Full artifacts are v60; lite candidate uses v53 `v53-0ba42a8c.nnue` when available.
- **License/distribution:** upstream AGPL-3.0. Generated WASM/NNUE blobs are intentionally ignored; distribution must include corresponding source archives.
- **Runtime adapter:**
  - WASI/UCI one-shot fallback.
  - Persistent shared-stdin WASI worker when `SharedArrayBuffer` and cross-origin isolation are available.
  - Direct browser API variants bypass UCI text for structured calls, but remain experimental.
- **UI variants:**
  - `relaxed-simd`: `Reckless Full Relaxed SIMD`, `/reckless/reckless-relaxed-simd128.wasm`; first-choice default when Relaxed SIMD validates and the asset is present.
  - `simd`: `Reckless Full SIMD`, `/reckless/reckless-simd128.wasm`; fixed-SIMD fallback when Relaxed SIMD is unsupported or its default asset is missing.
  - `full`: `Reckless Full scalar fallback`, `/reckless/reckless.wasm`; final fallback when neither SIMD default is usable.
  - `wasi-simd-external`: `/reckless/reckless-simd128-external.wasm` + `/reckless/reckless-v60-7f587dfb.nnue`; explicit non-default persistent WASI/UCI prototype.
  - `browser-api`: `/reckless/reckless-browser-api.wasm`.
  - `browser-api-simd`: `/reckless/reckless-browser-api-simd128.wasm`.
  - `browser-api-simd-external`: `/reckless/reckless-browser-api-simd128-external.wasm` + `/reckless/reckless-v60-7f587dfb.nnue`.
  - `lite`: `/reckless/reckless-v53-l1-512.wasm`, optional experimental smaller/weaker build.
  - `custom`: URL param escape hatch via `?recklessWasm=`.
- **Strength knob:** depth in Analysis/Play; Arena also supports equal movetime.
- **Artifact footprint:**
  - Full scalar/SIMD/browser-api integrated artifacts: ~62 MB each, because the NNUE is embedded.
  - External-NNUE browser API SIMD: ~1.2 MB WASM + ~60 MB cacheable NNUE.
  - `npm run reckless:build-release` is the canonical release build. The required default-ladder assets are `reckless-relaxed-simd128.wasm`, `reckless-simd128.wasm`, and `reckless.wasm`, with matching generated source archives.
- **Feature parity:**
  - Move search: yes.
  - Analysis/MultiPV: yes via UCI-like info lines/direct API formatting.
  - Persistent process: yes in isolated browsers.
  - Hash/new-game reset: clear-hash/new-game path corrected and benchmarked.
  - Stop/abort: persistent abort queues `stop` and allows 100 ms for natural completion, drains it without publishing cancelled output, and rejects with `AbortError`. Otherwise it terminates/recreates the worker. The synchronous UCI loop cannot reliably read stop during search.
  - Syzygy/tablebases: disabled for browser/WASI builds.
- **Speed snapshot:** see `docs/reckless_browser_benchmarks.md`.
  - Corrected depth 7/8/9 run: SIMD WASI/UCI had ~2.6-2.7x engine-reported NPS over scalar at deeper depths.
  - Browser API SIMD did not beat SIMD WASI/UCI in corrected runs; keep as experimental/control-plane groundwork.
- **Validation:** `lab/reckless-benchmark.html`, Reckless variant tests, one-shot/persistent browser benchmark JSON artifacts, asset HEAD checks in UI.
- **Open work:** improve graceful stop/persistent process control; keep browser API only if it earns latency or control-plane wins; finish deployment packaging for generated artifacts/source archives.

### Berserk family

- **Family id / UI label:** `berserk` / `Berserk`.
- **Integration status:** experimental opt-in. The Emscripten worker adapter passed Node and browser smokes and is available in staged arena/analysis selectors, but not as a default engine.
- **Source/version anchor:** upstream `jhonnold/berserk` release tag `14`, commit `8ae895a6151695be4a50d4fb65b0c131659c513a`; default branch HEAD observed at `27212a24c16d9e5f9bc9180a75264c1c632808bb` during intake. The browser port pins tag `14` unless a newer release is deliberately selected.
- **License/distribution:** GPL-3.0. Generated JS/WASM/data and copied NNUE assets must be treated as redistributable GPL engine artifacts with corresponding source/build instructions. Do not publish generated blobs unless the release satisfies `docs/engine_artifact_distribution.md`.
- **Runtime adapter:** `BerserkEngine` extends the shared `EmscriptenUciEngine`, using `patches/berserk-emscripten.patch` and the scalar/SIMD/relaxed build scripts, exporting `command()`, `isReady()`, and `isSearching()` like Stockfish.js. It disables tablebases and uses synchronous single-thread search to avoid pthread/SAB requirements. WASI remains unpromoted unless it becomes useful later.
- **UI variants:** defined in `src/lc0/berserkVariants.ts`.
  - `emscripten`: scalar fallback, `/berserk/berserk-emscripten.{js,wasm}`.
  - `emscripten-simd`: SSE4.1-shaped SIMD, `/berserk/berserk-emscripten-simd128.{js,wasm}`.
  - `emscripten-relaxed`: feature-detected first choice, `/berserk/berserk-emscripten-relaxed-simd128.{js,wasm}`. The released relaxed tier uses `-O2 -flto`; scalar/SIMD flag defaults are unchanged.
  - All three share `/berserk/berserk-emscripten.data`.
  - `default`: planned scalar WASI/UCI candidate, expected URL `/berserk/berserk.wasm`; not exposed in normal staged selectors until it smokes.
  - `simd`: planned SIMD WASI candidate, expected URL `/berserk/berserk-simd128.wasm`; not exposed in normal staged selectors until it smokes.
  - `custom`: URL param escape hatch for local experiments via `?berserkJs=`/`?berserkData=` or future `?berserkWasm=` and optional `?berserkNnue=`.
- **Expected NNUE assets:** upstream makefile uses `MAIN_NETWORK = berserk-9b84c340af7e.nn` from `https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn`. The Emscripten smoke preloads this network into `berserk-emscripten.data`; `berserkVariants.ts` still models external `/berserk/berserk-9b84c340af7e.nn` for any future WASI/custom path.
- **Strength knob:** Arena depth or equal movetime; Analysis depth. Catalog defaults are depth 4 / 12 for Arena / Analysis.
- **Artifact footprint:** three tracked Emscripten JS/WASM tiers share one ~24 MB `.data` NNUE preload. The release manifest and matching corresponding-source archive are published beside them; regenerate the set together after any engine, patch, build, or network change.
- **Feature parity:** handshake, repeated search/new-game, MultiPV/info parsing, missing-asset reporting and worker-restart abort recovery are implemented. Hash/Threads options are deduplicated because Hash changes clear TT. Browser builds remain single-threaded with tablebases disabled; NNUE is preloaded in the shared `.data`.
- **Speed snapshot:** dated first-port browser data is in `docs/berserk_browser_benchmarks.md`; Node SIMD/relaxed comparisons are in `docs/cpu_engines_simd_audit.md`. The later relaxed-tier LTO gate gained ~29% paired browser wall NPS on M4 at depths 11/12; see `docs/cpu_engine_build_flags_matrix_2026-10.md`. These are separate controls, not one cumulative universal speedup.
- **Validation:** `npm run berserk:build-emscripten` then `npm run berserk:smoke-emscripten` verifies `uci`, `isready`, `ucinewgame`, startpos search, and one non-startpos FEN search in Node. `lab/berserk-smoke.html` exercises the reusable `BerserkEngine` worker adapter in the browser; lifecycle smoke now covers repeated searches, MultiPV parsing, abort-by-worker-restart recovery, and missing-asset failure. Arena/analysis selector smokes and first sanity comparisons against Stockfish full single/Reckless SIMD are recorded in `docs/berserk_browser_benchmarks.md`. The reusable C/C++ intake process is captured in `docs/browser_c_engine_porting.md`.
- **Open risks:** graceful in-search stop without losing the resident worker, portability of measured compiler gains, source/archive obligations, and broader strength/latency evidence. SIMD is delivered; additional CPU threading is not an active promotion recommendation.

### PlentyChess family

- **Family id / UI label:** `plentychess` / `PlentyChess`.
- **Integration status:** experimental opt-in. Emscripten build, Node UCI smoke, reusable worker adapter, browser lifecycle smoke, first shallow benchmark, and arena/analysis selector smoke pass.
- **Current release:** 8.0.0 (`b-v8.0.0`, `50ad9f1b`), network `0178r.bin`; see `docs/plentychess_upgrade_v8.md`. Current variants are `emscripten-relaxed` → `emscripten-sse41` → `emscripten`, feature/asset-gated and sharing one processed-NNUE `.data`. Arena supports depth/equal movetime, Analysis depth. `PlentyChessEngine` uses the shared Emscripten adapter. The following intake notes, sizes and pending-task wording describe the original 7.0.66 port, not the current release.
- **Source/version anchor:** initial reconnaissance used `https://github.com/Yoshie2000/PlentyChess.git` at commit `58d8ba2505ae2b49f48dd410d214a457d15c12c6` on the default branch. Current source reports `VERSION = "7.0.66"` and `network.txt = 0134-2r24-s0`. Latest GitHub release observed during intake was `b-v7.0.0` at commit `e2060ab4b3021babb7d74af3cbe908154c03b2fd`, but its `network.txt` value `0119` did not correspond to an immediately available `PlentyNetworks` asset under the Makefile's expected URL during this reconnaissance.
- **License/distribution:** GPL-3.0. Generated JS/WASM and embedded/processed NNUE assets must stay ignored until a release manifest and matching source archive satisfy `docs/engine_artifact_distribution.md`.
- **Runtime adapter:** current proof uses the standard C/C++ recipe in `docs/browser_c_engine_porting.md`: Emscripten, modular JS, exported `command(const char*)`, single-thread synchronous search, and tablebases inert. `src/lc0/plentychessEngine.ts` wraps it as a `BrowserUciEngine` worker adapter.
- **Expected NNUE assets:** default-branch Makefile downloads `https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin`. Reconnaissance observed a 57,557,991-byte raw net with sha256 `550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061`; upstream `tools/process_net` produced a local `processed.bin` of about 60 MB with sha256 `691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4`.
- **Porting notes:**
  - Native UCI loop blocked on `std::getline(std::cin, ...)`; `patches/plentychess-emscripten.patch` adds an Emscripten `command()` entry point like Berserk/Stockfish.js.
  - `ThreadPool::resize(1)` and `TT.clear()` spawned `std::thread`s; the patch adds a `PLENTY_SYNC_SEARCH` path that creates the single worker directly, runs `Worker::startSearching()` inline, makes `waitForSearchFinished()` trivial, and clears TT with one `memset`.
  - Fathom/Syzygy is included but browser tablebases are not shipped. The proof build compiles with `TB_NO_THREADS` and relies on empty tablebase state.
  - Upstream `incbin` assembler directives do not work for wasm, so the proof build preloads `processed.bin` into `.data` and reads it through Emscripten's filesystem at startup.
  - SIMD assumptions are stronger than Berserk. The proof build currently uses Emscripten's wasm SIMD lowering through `-DARCH_X86 -msimd128 -mssse3`; this is acceptable for proof smoke but still needs browser lifecycle and benchmark validation.
- **Strength knob:** depth first; movetime later once UCI smoke is stable. Keep experimental even if fast because lifecycle and artifact policy are unknown.
- **Artifact footprint:** first local proof build emitted `plentychess-emscripten.js` 70,886 bytes, `plentychess-emscripten.wasm` 389,983 bytes, and `plentychess-emscripten.data` 63,023,936 bytes. `npm run plentychess:artifact-manifest` estimates total transfer at 34,180,543 bytes with gzip-9 or 32,587,527 bytes with brotli-11 if the host compresses/precompresses all sidecars, including `.data`. Generated files remain ignored under `public/plentychess/`.
- **Validation:** `npm run plentychess:build-emscripten` then `npm run plentychess:smoke-emscripten` verifies `uci`, `isready`, `ucinewgame`, startpos search (`bestmove c2c4`), one non-startpos FEN (`bestmove e1g1`), and post-search `readyok` in Node. `lab/plentychess-smoke.html` verifies browser worker prewarm, repeated searches, MultiPV, abort/recovery, and missing-asset failure surfacing. `lab/reckless-benchmark.html` includes a PlentyChess checkbox; first depth-7 rotated-FEN benchmark measured ~9.35 ms/search, ~4,068 nodes/search, and ~718k engine-reported NPS. Arena selector smoke passed on K-v-K vs Stockfish Lite; analysis selector smoke rendered depth-8 startpos PVs. Details are in `docs/plentychess_browser_port.md`.
- **Open risks:** source pin choice (`main` vs release), missing/stale release-network mapping, single-thread Emscripten patch size, tablebase guards, SIMD/scalar fallback correctness, large GPL artifact packaging, and whether the engine is meaningfully distinct from the existing Stockfish/Reckless speed-strength lane in browser form.

### Stormphrax family

- **Family id / UI label:** `stormphrax` / `Stormphrax`.
- **Integration status:** experimental opt-in in Play, Arena, and Analysis.
- **Source/version anchor:** `Ciekce/Stormphrax` tag `v8.0.0`, commit `582965517ed2032d41a6b4cd6c2e66b1b934e2ad`; self-trained release network `undertown.nnue`, 55,914,848 bytes, SHA-256 `04d651e078b7c7334709dbd772d40a23c0a5480e93e19521a03020c7d633f2cf`.
- **License/distribution:** GPL-3.0-or-later. Public JS/WASM/data requires the matching source archive and artifact manifest.
- **Runtime adapter:** patched single-thread synchronous Emscripten UCI worker. A new 128-bit wasm SIMD backend maps the engine's vector operations to Emscripten SSE intrinsics; threat-input geometry uses a correctness-first scalar browser path.
- **Variants:** `emscripten-relaxed` is preferred when supported, with `emscripten` baseline SIMD fallback. Both share one `.data`. Same-origin custom sidecar overrides use `stormphraxJs`, `stormphraxWasm`, and `stormphraxData`.
- **Strength knob:** depth; movetime is supported by the adapter.
- **Feature parity:** UCI handshake/readiness, repeated new games, start/non-start FEN search, MultiPV analysis, parsed info/PV/WDL output, and worker-restart abort recovery. Threads remain fixed at one; Syzygy is not loaded.
- **Artifact footprint:** initial build emits JS 70,611 bytes, WASM 697,245 bytes, and data 55,914,848 bytes.
- **Validation:** `npm run stormphrax:build-emscripten`, `npm run stormphrax:smoke-emscripten`, and browser selector smokes. See `docs/stormphrax_browser_port.md`.
- **Open work:** broader browser/native evidence. Shared Emscripten adapter and browser/deploy registries are implemented; see `docs/engine_integration_architecture.md`.

### Viridithas family

- **Family id / UI label:** `viridithas` / `Viridithas`.
- **Integration status:** experimental. Arena and analysis can use it, but it should not be presented as production-ready.
- **Source/version anchor:** upstream `cosmobobak/viridithas` tag `v20.0.0`, commit `0631113e2215714c54d397cce9a8caebc5e6f8ee`; build script downloads v109 `sandhi-s2-b200.nnue.zst`; local patch is `patches/viridithas-wasip1.patch`. The previous pin `20d74020` was a 2026-06-03 pre-release that already called itself v20.0.0 and embedded `atlantis`.
- **License/distribution:** engine source is MIT; the embedded `sandhi-s2-b200` network is CC0-1.0. The WASM binaries, manifest, and corresponding-source archive are tracked with Git LFS.
- **Runtime adapter:** patched Rust `wasm32-wasip1` UCI.
  - One-shot argv mode.
  - Benchmark-only batch one-process mode.
  - Persistent shared-stdin WASI worker in isolated browsers.
- **UI variants:**
  - `relaxed-simd`: `Viridithas Relaxed SIMD`, `/viridithas/viridithas-relaxed-simd128.wasm` (feature-detected default when Relaxed SIMD validates).
  - `simd`: `Viridithas SIMD experimental`, `/viridithas/viridithas-simd128.wasm` (default when SIMD validates and Relaxed SIMD does not).
  - `default`: `Viridithas scalar experimental`, `/viridithas/viridithas.wasm` (explicit compatibility fallback via `?viridithas=default`).
  - `custom`: URL param escape hatch via `?viridithasWasm=`.
- **Strength knob:** depth in UI and benchmarks; movetime benchmark plumbing exists.
- **Artifact footprint:** scalar, SIMD, and relaxed WASM artifacts are about 53.2 MiB each with the compressed network embedded.
- **Feature parity:**
  - Move search: yes for sequential searches and shallow arena games.
  - Analysis: yes through Stockfish/Reckless-style UCI info formatting.
  - Persistent process: works for sequential searches when SAB/cross-origin isolation are available.
  - Stop/abort: isolated persistent mode uses a separate shared stop word, queues `stop`, drains the old `bestmove` and rejects the cancelled request without publishing output. A 100 ms watchdog terminates unresponsive/older/custom modules and cold startup. One-shot abort terminates. See `docs/cpu_coop_cancel_2026-10.md`.
  - Threads: native threadpool is replaced/flattened for wasm browser use.
  - NNUE caching: native temp-file/mmap cache is bypassed on wasm; decompressed network is leaked for process lifetime.
- **Speed snapshot:** see `docs/viridithas_browser_benchmarks.md`.
  - Focused SIMD run showed engine-reported NPS gains around 3.4x-12.0x over scalar depending on budget.
  - Versus Reckless, Viridithas one-shot was much slower by wall time and NPS; batch/persistent probes show startup/decompression amortization is important.
- **Validation:** `npm run viridithas:smoke`, browser persistent smoke JSON, arena smoke games, benchmark JSON artifacts.
- **Open work:** broader correctness/strength and repeated persistent benchmarks. Cooperative cancellation is implemented, with watchdog limits; a direct API still needs a measured reason.

## Next CPU engine inclusion backlog

The next best browser CPU engine inclusions, after the already-integrated Berserk and Viridithas paths, are:

1. **Koivisto** — top intake target. Independent, strong C++ UCI engine and a useful non-Stockfish comparison point. Start with an Emscripten single-thread UCI worker proof, then decide whether a persistent worker or SIMD follow-up is worth the artifact/runtime complexity. Source candidate: `https://github.com/Luecx/Koivisto`.
2. **Patricia** — second target because it adds a differentiated playing style rather than just more strength coverage. Intake goal is an aggressive/personality engine card for Arena/Analysis, with depth/movetime behavior benchmarked separately from Elo claims. Source candidate: `https://github.com/Adam-Kulju/Patricia`.
3. **Caissa** — third target. Strong from-scratch C++ UCI engine with custom neural evaluation; valuable if its network/model assets can be packaged cleanly for browser distribution. First pass should measure network size, initialization cost, and whether embedded vs sidecar assets are viable. Source candidate: `https://github.com/Witek902/Caissa`.
4. **RubiChess** — fourth target. Mature C++ UCI engine and likely straightforward Emscripten intake; include after the more differentiated Koivisto/Patricia/Caissa candidates unless its build turns out materially easier. Source candidate: `https://github.com/Matthies/RubiChess`.

Do **not** prioritize AsmFish for browser inclusion unless a special-purpose experiment needs it: its assembly-first implementation is a poor fit for WASM and overlaps the existing Stockfish.js baseline. Stockfish-family derivatives such as ShashChess, Crystal, and SugaR can be revisited later as named analysis/personality variants, but they are lower priority than the independent CPU engines above.

Each backlog item still needs the normal onboarding card before selector/UI work: exact upstream pin, license and network provenance, expected artifact URLs, build command, source/archive policy, smoke plan, and benchmark protocol.

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
7. **Add asset policy.** Decide whether blobs are committed, symlinked, generated locally, or published by deployment. For GPL/AGPL engines, follow `docs/engine_artifact_distribution.md` and document source/archive obligations next to the build script.
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

Reckless, Viridithas, Berserk, PlentyChess, and Stormphrax keep their dynamic artifact metadata in `recklessVariants.ts`, `viridithasVariants.ts`, `berserkVariants.ts`, `plentychessVariants.ts`, and `stormphraxVariants.ts` because those modules own URL-param custom variants, asset checks, and runtime-specific defaults. If a future engine uses similar generated artifacts, add a dedicated variant module and then surface its family/static selector facts through `engineCatalog.ts`.

Any UI variant/default change should update both `src/lc0/engineCatalog.ts` and this card in the same commit. `tests/engine_catalog.test.mjs` pins the current selector order, static variants, family guard, and arena/analysis strength defaults.
