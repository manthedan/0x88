# Reckless browser/WASI integration

Reckless does not currently ship a browser-ready JavaScript/WASM worker. Upstream releases are native UCI binaries (Linux/Windows/macOS, generic/AVX2/AVX512) and upstream source is a Rust UCI engine with native threading and optional Syzygy/Fathom support.

This branch adds an optional browser path:

- Build target: `wasm32-wasip1`, `--no-default-features` to skip Syzygy/Fathom.
- Runtime: `@bjorn3/browser_wasi_shim` inside a dedicated Web Worker.
- Protocol: one-shot UCI argv commands (`setoption`, `position`, `go`) per search. The adapter skips redundant `uci`/`isready`/`quit` commands because the patched CLI mode exits automatically after the final argv command.
- UI: optional Reckless engines in `lc0-arena.html` and `lc0-analysis.html`.

## Build

```sh
npm run reckless:build-wasi
```

This writes `public/reckless/reckless.wasm`. The WASM is ignored by git because it is large and AGPL-licensed upstream.

Optional environment variables:

- `RECKLESS_REPO` — clone URL, default `https://github.com/codedeliveryservice/Reckless.git`.
- `RECKLESS_REF` — branch/tag, default `main`.
- `RECKLESS_BUILD_DIR` — temp/source checkout, default `.local_engines/reckless-wasi-src`.
- `RECKLESS_WASM_OUT` — output path, default `public/reckless/reckless.wasm`.

## Browser usage

After building the asset:

```sh
npm run web:client
# open /lc0-arena.html or /lc0-analysis.html
```

Use `?recklessWasm=/path/to/reckless.wasm` to point at another asset.

## Current limitations

- No upstream browser artifact was found.
- The adapter is one-shot, so Reckless does not retain hash between moves.
- Aborting terminates the worker and recreates it for the next search.
- Threads are forced to `1`; the WASI path avoids native Rust threads.
- The WASM artifact is large because Reckless embeds its NNUE network.
