# Reckless browser/WASI integration

Reckless does not currently ship a browser-ready JavaScript/WASM worker. Upstream releases are native UCI binaries (Linux/Windows/macOS, generic/AVX2/AVX512) and upstream source is a Rust UCI engine with native threading and optional Syzygy/Fathom support.

This branch adds an optional browser path:

- Build target: `wasm32-wasip1`, `--no-default-features` to skip Syzygy/Fathom.
- Runtime: `@bjorn3/browser_wasi_shim` inside a dedicated Web Worker.
- Protocol: isolated browsers use a persistent UCI process fed through a small `SharedArrayBuffer` stdin ring; non-isolated browsers fall back to one-shot UCI argv commands (`setoption`, `position`, `go`) per search. The one-shot adapter skips redundant `uci`/`isready`/`quit` commands because the patched CLI mode exits automatically after the final argv command.
- UI: optional Reckless engines in `lc0-arena.html` and `lc0-analysis.html`, with Full/Lite variant selectors and runtime status labels.
- Benchmarking: `/reckless-benchmark.html` compares Full vs Lite and persistent vs one-shot browser paths across one or more positions and depth/movetime budgets. It emits raw rows, warm-run summaries, CSV, and a JSON report with browser runtime metadata.

## Build

```sh
npm run reckless:build-wasi
```

This writes `public/reckless/reckless.wasm`. The WASM is ignored by git because it is large and AGPL-licensed upstream.

SIMD candidate build:

```sh
npm run reckless:build-simd-wasi
```

This writes `public/reckless/reckless-simd128.wasm` with `-C target-feature=+simd128`. It still uses Reckless' scalar Rust NNUE source paths, but allows LLVM/rustc to emit WebAssembly SIMD operations where it can. A quick Node/browser-wasi-shim depth-5 startpos smoke on this machine measured warm one-shot average time improving from about 10.5 ms to 8.8 ms (~16%). Treat this as benchmark-first and experimental until browser matrix results are saved.

Lite candidate build:

```sh
npm run reckless:build-lite-wasi
```

This writes `public/reckless/reckless-v53-l1-512.wasm` when `.local_engines/reckless-nets/v53-0ba42a8c.nnue` is available.

Optional environment variables:

- `RECKLESS_REPO` — clone URL, default `https://github.com/codedeliveryservice/Reckless.git`.
- `RECKLESS_REF` — branch/tag, default `main`.
- `RECKLESS_BUILD_DIR` — temp/source checkout, default `.local_engines/reckless-wasi-src`.
- `RECKLESS_WASM_OUT` — output path, default `public/reckless/reckless.wasm`.
- `RECKLESS_EVALFILE` — optional NNUE file to embed instead of upstream's default downloaded network.
- `RECKLESS_L1_SIZE` — optional `src/nnue.rs` `L1_SIZE` override for known-compatible smaller networks, e.g. `512` for the v53 Lite candidate.

## Browser usage

After building the asset:

```sh
npm run web:client
# open /lc0-arena.html or /lc0-analysis.html
```

Use `?recklessVariant=simd` for the local SIMD candidate, `?recklessVariant=lite` for the local Lite candidate, or `?recklessWasm=/path/to/reckless.wasm` to point at another asset. Arena and Analysis show the selected Reckless asset URL and warn when the selected WASM is missing.

## Current limitations

- No upstream browser artifact was found.
- Persistent mode requires cross-origin isolation / `SharedArrayBuffer`; use the isolated dev/static server for this path.
- Non-isolated browsers still use the one-shot fallback, so Reckless does not retain hash between moves there.
- Aborting terminates the worker and recreates it for the next search. Sending `stop` cannot currently preserve the persistent process because patched wasm32 UCI mode is single-threaded and cannot read stdin while the search is executing.
- Threads are forced to `1`; the WASI path avoids native Rust threads.
- The SIMD candidate is only a `simd128` target-feature build, not a dedicated `core::arch::wasm32` NNUE backend.
- The WASM artifact is large because Reckless embeds its NNUE network.
