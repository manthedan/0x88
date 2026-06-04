# Reckless threaded WASM feasibility notes

## Current browser build status

The current Reckless WASI browser build is intentionally single-threaded:

- `RecklessEngine.searchCommands()` always sends `setoption name Threads value 1`.
- `scripts/build_reckless_wasi.mjs` patches Reckless' thread pool for `target_arch = "wasm32"` so scoped worker jobs execute inline instead of spawning OS threads.
- The generated module exports a normal non-shared memory (`wasm-tools print public/reckless/reckless.wasm` shows `(memory (;0;) 999)`), not a shared memory import/export.
- Cross-origin isolation is already available through `npm run web:isolated:static`, so the browser side can provide `SharedArrayBuffer`; the missing part is a Reckless/WASM runtime design that maps search workers onto browser Workers safely.

## Why this does not map cleanly today

Reckless' native search expects Rust `std::thread`/scoped worker semantics and shared in-process structures such as `SharedContext`, transposition table state, node counters, stop status, and thread-local `ThreadData`. The current WASI shim path can keep one process alive, but it does not provide browser pthreads or a transparent implementation of Rust `std::thread` over Web Workers for this build.

Because the wasm32 patch executes worker closures inline, simply setting `Threads > 1` in UCI would not create browser parallel search. Reversing that patch without a thread runtime would either fail to compile/run or hang on unsupported thread primitives.

## Candidate implementation paths

1. **Browser-native API first, then manual root split**
   - Create one engine instance per Worker.
   - Split root moves across Workers with independent TT/search state.
   - Merge best root result/PV on the main adapter side.
   - Pros: likely easiest to ship incrementally and works with current browser primitives.
   - Cons: no shared TT, duplicated NNUE/wasm memory unless external/shared parameters are implemented.

2. **Shared-memory native-style threading**
   - Build a target/runtime with shared wasm memory, atomics, and browser Worker startup glue.
   - Preserve Reckless' shared `SharedContext`/TT model more closely.
   - Pros: best long-term analysis throughput if it works.
   - Cons: highest engineering risk; requires replacing the current WASI shim assumptions and auditing all synchronization for browser Worker constraints.

3. **Hybrid split with shared NNUE asset/module cache**
   - Keep per-Worker engine state but share fetched NNUE bytes and compiled `WebAssembly.Module`.
   - Pros: avoids the largest cold-load duplication once external NNUE loading exists.
   - Cons: still has duplicated mutable search state and result-merging complexity.

## Recommendation

Do not attempt native-style threaded WASI search as the next step. The practical order is:

1. Land the browser-native API facade/worker so searches are direct calls, not UCI text.
2. Add graceful cancellation via a direct shared/atomic stop flag.
3. Add external/shared NNUE asset loading to avoid duplicated 64 MB data payloads per Worker.
4. Prototype root-split parallel search across two Workers and benchmark analysis-mode throughput.
5. Only revisit true shared-memory threading if root-split Workers are insufficient.

This keeps the existing WASI/UCI path as a reliable single-thread fallback while making threaded experiments explicit and measurable.
