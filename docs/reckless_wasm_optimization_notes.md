# Reckless WASM optimization notes

## SIMD build status

The promoted default SIMD WASI/UCI artifact is built with:

```sh
npm run reckless:build-production
```

This builds the scalar fallback and the default SIMD artifact, then emits matching patched-source archives for license compliance. Internally, the SIMD build sets `RUSTFLAGS='-C target-feature=+simd128'`, enables the local `RECKLESS_WASM_SIMD_NNUE=1` patch, and writes the ignored artifact `public/reckless/reckless-simd128.wasm`. Arena and Analysis select this variant by default when the browser validates WebAssembly SIMD support; the scalar `public/reckless/reckless.wasm` remains the fallback for unsupported browsers or missing implicit SIMD assets.

Earlier measurements used a first-pass auto-vectorized-only `+simd128` artifact where upstream Reckless still selected scalar NNUE source paths for `wasm32-wasip1`. The current build now patches Reckless to select its vectorized NNUE path for `wasm32 + simd128` and adds a wasm32 SIMD module using `core::arch::wasm32` intrinsics.

### Smoke results

Node/browser-wasi-shim one-shot depth-5 startpos smoke, 1 cold + 10 warm runs:

| artifact | warm avg | warm min | warm max | avg NPS |
| --- | ---: | ---: | ---: | ---: |
| `reckless.wasm` | 10.467 ms | 9.598 ms | 11.425 ms | 169,435 |
| `reckless-simd128.wasm` | 8.801 ms | 8.183 ms | 10.674 ms | 231,220 |

Headless browser one-shot depth-5 startpos smoke, 1 cold + 3 warm runs:

| artifact | cold | warm avg | avg NPS |
| --- | ---: | ---: | ---: |
| `reckless.wasm` | 113.4 ms | 13.7 ms | 241,415 |
| `reckless-simd128.wasm` | 83.1 ms | 12.4 ms | 422,768 |

These shallow smokes were only early sanity checks. The later rotated-FEN depth 7/8/9 benchmark and fixed-depth parity validation are the basis for promoting SIMD WASI/UCI as the production default.

## Remaining optimization tracks

- **Dedicated WASM SIMD NNUE backend**: now implemented and promoted as the default production Reckless variant when WebAssembly SIMD is supported. The build script patches Reckless' NNUE module selection for `wasm32 + simd128` and adds a `core::arch::wasm32` SIMD module covering accumulator add/sub, `activate_ft`, sparse `propagate_l1`, `propagate_l2`, and `propagate_l3`; see [`reckless_wasm_simd_inspection.md`](./reckless_wasm_simd_inspection.md). A rotated-FEN browser benchmark showed clear persistent depth 8/9 wins versus scalar WASI/UCI, and a follow-up fixed-depth parity validation matched scalar exactly across the 20-position suite at depths 7/8/9; see [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md). The 2026-06-09 kernel-fix pass de-scalarized the `dpbusd` emulation (+19% Node NPS at depth 12), vectorized `find_nnz` over the existing `nnz_table`, fixed the latent `nnz_bitmask` shape bug, and switched relaxed builds to an exact relaxed integer dot (+24% vs old kernels); see [`reckless_simd_kernel_fixes.md`](./reckless_simd_kernel_fixes.md).
- **Threaded search**: still not implemented. The current browser/WASI path forces `Threads=1`. `@bjorn3/browser_wasi_shim` does not provide native Rust pthread-style browser execution for this build, so practical browser threading likely needs either a custom Worker search-split design or a different WASM runtime strategy. Feasibility notes are in [`reckless_threaded_wasm_feasibility.md`](./reckless_threaded_wasm_feasibility.md); the Berserk Emscripten adapter landed since those notes and reopens a real-pthreads route, staged in [`reckless_wasm_next_exploration_notes.md`](./reckless_wasm_next_exploration_notes.md).
- **wasm-opt**: tried on scalar and SIMD artifacts; see [`reckless_wasm_opt_experiment.md`](./reckless_wasm_opt_experiment.md). Binaryen `-O3`/`-O4` trims only ~0.15% raw size and ~0.01% gzip size because embedded NNUE data dominates, and a shallow browser cold one-shot smoke did not show a clear NPS win.
- **Graceful persistent cancellation**: partially improved. The persistent adapter now sends `stop` and waits briefly before terminating the worker, so near-complete searches can return `bestmove` and keep the process reusable while the caller still receives `AbortError`. This is still not a true engine-side cancellation path: the patched wasm32 UCI loop is single-threaded and cannot reliably read `stop` while search is executing. A robust path still needs an engine-side shared abort/control flag checked from the search, or a browser-native API that exposes cancellation directly.
- **Browser-native API**: now implemented experimentally behind `Reckless Full browser API experimental` / `SIMD experimental` and `npm run reckless:build-browser-api`; see [`reckless_browser_native_api_plan.md`](./reckless_browser_native_api_plan.md). It still uses `wasm32-wasip1` imports for clocks because `wasm32-unknown-unknown` panics in `std::time::Instant`, but it bypasses `_start`, argv/stdin/stdout, UCI parsing, and UCI output formatting. Its `new_game` path now clears correction-history tables like UCI `ucinewgame`; corrected clear-hash runs matched scalar/SIMD WASI parity, while browser API SIMD remained slower than SIMD WASI/UCI at depths 7/8/9. Keep WASI/UCI as the default and treat the browser API mainly as a control/cancellation/structured-result path until the facade is faster.
- **Smaller/lazier NNUE asset path**: full Reckless artifacts are ~99.5% data payload; see [`reckless_nnue_asset_size_plan.md`](./reckless_nnue_asset_size_plan.md). The browser API SIMD external-NNUE build emits a ~1.26 MB WASM plus a separate ~63.27 MB `.nnue` payload, reports NNUE load progress in the UI, and matched embedded browser API SIMD exactly in 1260/1260 fixed-depth rotated-FEN pairs at depths 7/8/9. It remains experimental because embedded SIMD WASI/UCI is still the faster default production path.

## Adapter overhead notes

The persistent WASI adapter already keeps one process alive and skips repeated stable `setoption` commands. It now also skips a repeated identical `position ...` command in persistent mode. This is a small UCI parsing/stdin reduction for analysis refreshes and benchmark loops that re-search the same FEN with a different budget; it does not address the larger one-shot costs from WASI process startup, worker messaging, and stdout parsing.

Arena and analysis pages now call `RecklessEngine.prewarm()` after creating a Reckless instance. In cross-origin-isolated browsers, this starts the persistent worker/process and runs `uci`/`isready` before the first real search, hiding the 80-115 ms cold worker/wasm/UCI startup penalty when the user later asks for a move or analysis. Non-isolated browsers still fall back to one-shot mode.

The worker caches one `WebAssembly.Module` promise per artifact URL. It now attempts `WebAssembly.compileStreaming(response.clone())` first and falls back to `arrayBuffer()` + `WebAssembly.compile()` for servers with an incompatible MIME type. Failed cache entries are evicted so transient fetch/compile failures do not poison the worker for the rest of the page lifetime.

The worker now filters stdout before storing or posting it back to the main thread: it keeps only `uciok`, `readyok`, `bestmove`, and `info ... pv ...` lines. Stderr remains unfiltered. This preserves the data needed for readiness, bestmove, PV, score, nodes, and NPS while avoiding a high-volume stream of option/current-move diagnostics during browser searches.

Search command batches now set Reckless' `Minimal` UCI option to `true`. Reckless still emits the final PV/score/nodes/NPS line that the adapter parses, but it avoids intermediate full-report `info ... pv ...` chatter during longer searches. The persistent adapter caches this stable option just like Hash/Threads/MultiPV.

Persistent aborts now attempt a short graceful reuse path: enqueue `stop`, wait up to 100 ms for the current search to naturally emit `bestmove`, reject the caller with `AbortError`, and keep the worker alive if that happens. If no completion arrives during the grace window, the adapter falls back to worker termination as before.

The benchmark harness now has a default-on clear-hash reset. It sends `ucinewgame` + `isready` before each timed run and starts the timer afterward, avoiding persistent repeated-position TT pollution without mixing reset latency into the search wall-clock metric. It also includes a 20-position rotated-FEN suite preset, rotates warm passes across the full suite before repeating a FEN, and now records engine-reported nodes alongside wall ms and NPS.

The next major adapter-overhead reduction remains a browser-native API with direct calls for initialize, set FEN, search, and result retrieval. The staged feasibility plan is in [`reckless_browser_native_api_plan.md`](./reckless_browser_native_api_plan.md).
