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

- **Dedicated WASM SIMD NNUE backend**: still not implemented. The next deeper step is a `wasm32 + simd128` `nnue::simd` module using `core::arch::wasm32` intrinsics for the hot `i16` accumulation, pack/saturate, dot-product, and float reduction operations. This should be benchmarked against both scalar and `-C target-feature=+simd128` auto-vectorized builds.
- **Threaded search**: still not implemented. The current browser/WASI path forces `Threads=1`. `@bjorn3/browser_wasi_shim` does not provide native Rust pthread-style browser execution for this build, so practical browser threading likely needs either a custom Worker search-split design or a different WASM runtime strategy.
- **Graceful persistent cancellation**: still not implemented. The persistent process cannot currently receive `stop` while search is running because the patched wasm32 UCI loop is single-threaded. A robust path needs an engine-side shared abort/control flag checked from the search, or a browser-native API that exposes cancellation directly.
- **Browser-native API**: still not implemented. A lower-overhead path would avoid argv/stdin/stdout and WASI process startup by exposing direct WASM functions for engine initialization, option setting, position loading, search, bestmove, and cancellation. This likely means a separate `cdylib`/`wasm32-unknown-unknown` style target or a Reckless library facade rather than the current UCI binary shim.

## Adapter overhead notes

The persistent WASI adapter already keeps one process alive and skips repeated stable `setoption` commands. It now also skips a repeated identical `position ...` command in persistent mode. This is a small UCI parsing/stdin reduction for analysis refreshes and benchmark loops that re-search the same FEN with a different budget; it does not address the larger one-shot costs from WASI process startup, worker messaging, and stdout parsing.

Arena and analysis pages now call `RecklessEngine.prewarm()` after creating a Reckless instance. In cross-origin-isolated browsers, this starts the persistent worker/process and runs `uci`/`isready` before the first real search, hiding the 80-115 ms cold worker/wasm/UCI startup penalty when the user later asks for a move or analysis. Non-isolated browsers still fall back to one-shot mode.

The next major adapter-overhead reduction remains a browser-native API with direct calls for initialize, set FEN, search, and result retrieval.
