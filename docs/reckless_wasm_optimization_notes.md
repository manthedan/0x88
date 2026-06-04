# Reckless WASM optimization notes

## Quick SIMD candidate

A first-pass SIMD candidate was built with:

```sh
npm run reckless:build-simd-wasi
```

This sets `RUSTFLAGS='-C target-feature=+simd128'` and writes the ignored artifact `public/reckless/reckless-simd128.wasm`.

This is **not** a dedicated `core::arch::wasm32` NNUE backend. Upstream Reckless still selects its scalar Rust NNUE source paths for `wasm32-wasip1`; the flag only lets rustc/LLVM emit WebAssembly SIMD where auto-vectorization or library lowering can use it.

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

The SIMD flag is therefore worth keeping as an experimental build/benchmark target, but the browser smoke is short and noisy; do not treat this as final product posture.

## Remaining optimization tracks

- **Dedicated WASM SIMD NNUE backend**: still not implemented. Standalone probes now show large exact-parity speedups for both `i16` accumulator update and `activate_ft` clipped pair-product output preparation; see [`reckless_wasm_nnue_kernel_probe.md`](./reckless_wasm_nnue_kernel_probe.md) and [`reckless_wasm_nnue_activate_probe.md`](./reckless_wasm_nnue_activate_probe.md). The next deeper step is a `wasm32 + simd128` `nnue::simd` module using `core::arch::wasm32` intrinsics for these hot kernels plus sparse/dense output propagation. This should be benchmarked against both scalar and `-C target-feature=+simd128` auto-vectorized builds.
- **Threaded search**: still not implemented. The current browser/WASI path forces `Threads=1`. `@bjorn3/browser_wasi_shim` does not provide native Rust pthread-style browser execution for this build, so practical browser threading likely needs either a custom Worker search-split design or a different WASM runtime strategy. Feasibility notes are in [`reckless_threaded_wasm_feasibility.md`](./reckless_threaded_wasm_feasibility.md).
- **wasm-opt**: tried on scalar and SIMD artifacts; see [`reckless_wasm_opt_experiment.md`](./reckless_wasm_opt_experiment.md). Binaryen `-O3`/`-O4` trims only ~0.15% raw size and ~0.01% gzip size because embedded NNUE data dominates, and a shallow browser cold one-shot smoke did not show a clear NPS win.
- **Graceful persistent cancellation**: partially improved. The persistent adapter now sends `stop` and waits briefly before terminating the worker, so near-complete searches can return `bestmove` and keep the process reusable while the caller still receives `AbortError`. This is still not a true engine-side cancellation path: the patched wasm32 UCI loop is single-threaded and cannot reliably read `stop` while search is executing. A robust path still needs an engine-side shared abort/control flag checked from the search, or a browser-native API that exposes cancellation directly.
- **Browser-native API**: still not implemented. A lower-overhead path would avoid argv/stdin/stdout and WASI process startup by exposing direct WASM functions for engine initialization, option setting, position loading, search, bestmove, and cancellation. This likely means a separate `cdylib`/`wasm32-unknown-unknown` style target or a Reckless library facade rather than the current UCI binary shim.
- **Smaller/lazier NNUE asset path**: full Reckless artifacts are ~99.5% data payload; see [`reckless_nnue_asset_size_plan.md`](./reckless_nnue_asset_size_plan.md). Smaller first-use UX needs Lite/smaller nets or external NNUE loading rather than code-only optimization.

## Adapter overhead notes

The persistent WASI adapter already keeps one process alive and skips repeated stable `setoption` commands. It now also skips a repeated identical `position ...` command in persistent mode. This is a small UCI parsing/stdin reduction for analysis refreshes and benchmark loops that re-search the same FEN with a different budget; it does not address the larger one-shot costs from WASI process startup, worker messaging, and stdout parsing.

Arena and analysis pages now call `RecklessEngine.prewarm()` after creating a Reckless instance. In cross-origin-isolated browsers, this starts the persistent worker/process and runs `uci`/`isready` before the first real search, hiding the 80-115 ms cold worker/wasm/UCI startup penalty when the user later asks for a move or analysis. Non-isolated browsers still fall back to one-shot mode.

The worker caches one `WebAssembly.Module` promise per artifact URL. It now attempts `WebAssembly.compileStreaming(response.clone())` first and falls back to `arrayBuffer()` + `WebAssembly.compile()` for servers with an incompatible MIME type. Failed cache entries are evicted so transient fetch/compile failures do not poison the worker for the rest of the page lifetime.

The worker now filters stdout before storing or posting it back to the main thread: it keeps only `uciok`, `readyok`, `bestmove`, and `info ... pv ...` lines. Stderr remains unfiltered. This preserves the data needed for readiness, bestmove, PV, score, nodes, and NPS while avoiding a high-volume stream of option/current-move diagnostics during browser searches.

Search command batches now set Reckless' `Minimal` UCI option to `true`. Reckless still emits the final PV/score/nodes/NPS line that the adapter parses, but it avoids intermediate full-report `info ... pv ...` chatter during longer searches. The persistent adapter caches this stable option just like Hash/Threads/MultiPV.

Persistent aborts now attempt a short graceful reuse path: enqueue `stop`, wait up to 100 ms for the current search to naturally emit `bestmove`, reject the caller with `AbortError`, and keep the worker alive if that happens. If no completion arrives during the grace window, the adapter falls back to worker termination as before.

The benchmark harness now has a default-on clear-hash reset. It sends `ucinewgame` + `isready` before each timed run and starts the timer afterward, avoiding persistent repeated-position TT pollution without mixing reset latency into the search wall-clock metric. It also includes a 20-position rotated-FEN suite preset, rotates warm passes across the full suite before repeating a FEN, and now records engine-reported nodes alongside wall ms and NPS.

The next major adapter-overhead reduction remains a browser-native API with direct calls for initialize, set FEN, search, and result retrieval. The staged feasibility plan is in [`reckless_browser_native_api_plan.md`](./reckless_browser_native_api_plan.md).
