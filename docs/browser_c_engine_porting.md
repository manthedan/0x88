# Browser C/C++ engine porting recipe

This is the standard low-effort intake path for C/C++ UCI engines after the Berserk proof-of-build. The goal is to answer "can this engine be a useful browser resident worker?" without turning every intake into a tuning project.

## Scope and default stance

- Prefer **Emscripten first** for C/C++ engines. It matches the successful Stockfish.js-style browser pattern and avoids forcing engines through WASI stdin/pthread problems before the engine has proven useful.
- Start with a **single-thread synchronous worker build**. Pthreads, shared worker pools, and graceful in-search stop are follow-up work only after the engine earns it.
- Disable tablebases/Syzygy for first smoke unless the upstream engine can tolerate missing tablebase files cleanly.
- Keep generated JS/WASM/data/NNUE blobs ignored until license and corresponding-source packaging are explicit.
- Promote to normal UI selectors only after the reusable adapter smoke passes; keep new engines experimental until benchmark and lifecycle data justify defaults.

## Minimal Emscripten target

A first-pass C/C++ engine should expose a tiny browser API around its existing UCI loop:

- `main()` initializes engine global state but does not block forever on stdin under `__EMSCRIPTEN__`.
- `extern "C" void command(const char*)` feeds a UCI command into the normal UCI command processor.
- Optional diagnostics:
  - `extern "C" bool isReady()` for pthread/proxied-main builds.
  - `extern "C" bool isSearching()` for JS-side command queueing.
- JS glue should be modularized and worker-loadable:
  - `-s MODULARIZE=1`
  - `-s EXPORT_NAME="EngineName"`
  - `-s ENVIRONMENT=web,worker,node`
  - `-s EXPORTED_RUNTIME_METHODS=ccall`
  - `-s EXPORTED_FUNCTIONS=["_main","_command",...]`

Recommended first flags, adjusted per engine size:

- `-O3` or upstream release optimization.
- `-s ALLOW_MEMORY_GROWTH=1`.
- Large enough `INITIAL_MEMORY` and `STACK_SIZE` for NNUE/search recursion.
- `-s EXIT_RUNTIME=0`.
- `-s USE_PTHREADS=0` for the first smoke.

If the engine needs an NNUE file, either preload it into `.data` for the first smoke or provide a simple external fetch/cache path. The first Berserk path uses `--preload-file ...@/...` to eliminate custom NNUE I/O during intake.

## Required smoke checklist

Before catalog/UI promotion:

1. Reproducible build script pins upstream repo/tag/commit and network/model URL/checksum where applicable.
2. Patch file is committed and does not require editing generated output.
3. Node smoke verifies:
   - `uci` -> `uciok`
   - `isready` -> `readyok`
   - `ucinewgame`
   - `position startpos` + `go depth 1` -> `bestmove`
   - at least one non-startpos FEN -> `bestmove`
4. Browser worker smoke verifies the reusable TypeScript adapter, not just raw `eval` glue.
5. Lifecycle smoke verifies:
   - repeated searches
   - repeated `ucinewgame`
   - abort/recovery behavior
   - dispose/worker termination
   - visible missing-asset failure
   - `info` / score / PV parsing
   - MultiPV if analysis UI will expose it
6. A small benchmark captures cold/warm behavior over the rotated FEN suite using the shared browser benchmark harness.
7. Docs update `engine_catalog.md` and, if benchmarked, a per-engine benchmark note.

## Stockfish.js as the reference pattern

Inspection of `nmrugg/stockfish.js` at `4a6804f13ac45fe9e570e928700f93771cf967d2` suggests the mature pattern is mostly careful Emscripten packaging around upstream Stockfish, plus a few browser-specific source guards. It is not a large hand-written WebAssembly rewrite.

Important observed choices:

- Uses upstream Stockfish SIMD abstraction and compiles the WASM target with `-msimd128` in `src/emscripten/wasm-makefile.mk`.
- Enables Stockfish's x86-ish SIMD feature macros for the WASM target (`popcnt`, `sse`, `sse2`, `ssse3`, `sse41`) so existing NNUE SIMD code is selected where Emscripten can lower it to wasm SIMD.
- Provides both pthread and single-thread flavors:
  - pthread flavor uses `-s PROXY_TO_PTHREAD` and `-s USE_PTHREADS=1`.
  - single-thread flavor uses `-D__EMSCRIPTEN_SINGLE_THREADED__`, `-s USE_PTHREADS=0`, and Asyncify for `go` commands.
- Exports `_main`, `_command`, `_isReady`/`_isSearching` and wraps them with JS queueing around `ccall("command", ...)`.
- Embeds networks into generated headers for full/lite variants and ships multiple size/strength flavors.
- Uses browser-specific guards for Syzygy/filesystem/threading/tt behavior, but keeps engine logic recognizably upstream.

Takeaway: for new engines, we should first copy the **control-plane shape** (modular Emscripten build + exported `command()` + worker queue + smoke discipline). We should not try to match Stockfish.js's maturity immediately with custom SIMD, pthread flavoring, lite/full network packaging, and Asyncify/pthread variants unless the engine has already proven valuable.

## Deferred optimization ladder

Only climb this ladder when benchmark data says the engine deserves more time:

1. Confirm release flags, memory, stack, and tablebase-disabled code paths.
2. Try `-msimd128` if upstream code has portable/vector-friendly kernels and correctness smoke remains clean.
3. Reduce artifact size or split model/data if startup/download dominates.
4. Add graceful `stop` preservation instead of terminate/recreate on abort.
5. Add pthread flavor (`USE_PTHREADS`, worker sidecars, SAB/COOP/COEP requirement) and compare NPS/latency against single-thread.
6. Consider direct browser API bypassing UCI text only if UCI overhead is visible in profiling.

For Berserk, stop at step 1 for now: the adapter works, speed is acceptable for an experimental opponent, and Stockfish remains the expected speed/strength anchor.
