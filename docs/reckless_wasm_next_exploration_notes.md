# Reckless WASM: next exploration tracks

Companion to [`reckless_simd_kernel_fixes.md`](./reckless_simd_kernel_fixes.md).
These are the two remaining structural opportunities identified by the
2026-06 audit, with enough design detail to start either one cold.

## 1. Threats/PSQ accumulator SIMD via swizzle

### Why

The native hot-path profile
([`reckless_hot_path_profile.md`](./reckless_hot_path_profile.md)) shows
`nnue::accumulator::threats::scalar::push_threats_single` at 4.5% and
`PstAccumulator::refresh` at 4.0% of active samples. On wasm these stay
scalar: `src/nnue/accumulator/threats/vectorized.rs` only has `avx512` and
`avx2` submodules, so the `RAY_PERMUTATIONS` path never compiles for
`wasm32`. Expected upside is roughly 5-8% engine NPS — worthwhile but
clearly second to the L1 work that already landed.

### Design sketch

The avx512 path permutes 64-byte ray tables with `vpermb`. wasm has no
64-lane permute, but a 64-byte table permute decomposes into 16-byte
`i8x16_swizzle` blocks:

- For each 16-byte output chunk, the source index vector selects from one of
  four 16-byte source chunks. Compute `idx - 0`, `idx - 16`, `idx - 32`,
  `idx - 48` (saturating u8 subtract), swizzle each source chunk, and OR the
  results. Standard `i8x16_swizzle` zeroes out-of-range indexes, so the
  per-chunk masks come for free: 4 swizzles + 3 ORs + 4 subtracts per output
  chunk, 16 swizzles total per 64-byte permutation.
- Under `relaxed-simd`, `i8x16_relaxed_swizzle` has the same in-range
  behavior; out-of-range lanes are implementation-defined, so the saturating
  subtract trick (which forces out-of-range to >= 0x80... only for standard
  swizzle semantics) must stay — treat relaxed swizzle as a drop-in only if
  the subtracted indexes are proven in-range, otherwise keep standard
  swizzle. Do not relax this kernel first; measure the standard version.

Validation: the accumulator add/sub probe pattern
(`reckless:probe-nnue-accum-simd`) extends naturally — generate a standalone
probe with the scalar `push_threats_single` shape and the swizzle version,
require exact parity over randomized boards' threat deltas, then wire into
the build-script patch behind `RECKLESS_WASM_SIMD_NNUE=1` and re-run the
rotated-suite parity protocol.

## 2. Browser threading via the Berserk Emscripten route

### Why the old conclusion is stale

[`reckless_threaded_wasm_feasibility.md`](./reckless_threaded_wasm_feasibility.md)
ruled out native-style threading because `@bjorn3/browser_wasi_shim`
provides no pthread runtime for `wasm32-wasip1`, and recommended root-split
workers instead. Since then, this branch's lineage landed a working
**Berserk Emscripten** build (`patches/berserk-emscripten.patch`,
`scripts/berserk_emscripten_smoke.mjs`, worker adapter): Emscripten provides
real pthreads over Web Workers + SharedArrayBuffer, and the arena/analysis
pages already run cross-origin isolated. That removes the main blocker the
feasibility doc identified, without a custom runtime.

### Staged plan

1. **Port the WASI patch set to an Emscripten target.** Reckless builds for
   `wasm32-wasip1` with thread shims patched *out*; an Emscripten build
   (`-pthread -sPROXY_TO_PTHREAD -sPTHREAD_POOL_SIZE=n`) would instead keep
   upstream's `std::thread` threadpool, NUMA shim excepted. Rust supports
   Emscripten via `wasm32-unknown-emscripten`; the existing UCI stdin-loop
   patch is replaceable with Emscripten's stdin or a direct call facade.
2. **Smoke single-threaded first**: Emscripten Threads=1 must reproduce
   SIMD WASI parity on the rotated suite before any threading claim.
3. **Then Threads=2/4**: parity is *not* expected (parallel search is
   nondeterministic); validate with the playing-strength/arena path instead
   of fixed-depth pair matching, plus crash/abort soak.
4. **Cancellation for free**: an engine-side atomic stop flag in shared
   memory finally gives real `stop` handling, replacing the 100 ms
   grace-then-terminate persistent abort path.

Risks: Emscripten runtime size and startup vs the lean WASI shim; embedded
64 MB NNUE duplicated per... no — shared wasm memory means one copy across
search threads, which is strictly better than the root-split-worker
alternative (N × 64 MB). Main unknowns are scheduler behavior of Reckless'
lazy-SMP under Workers and the COOP/COEP requirement (already satisfied on
the isolated static host).

### Decision gate

Prototype step 1-2 only; if the Emscripten single-thread artifact is more
than ~10% slower than SIMD WASI/UCI at depths 7-9, threading gains must
clear that tax before replacing anything. Keep WASI/UCI as the production
default throughout, mirroring the browser-API precedent.

## Smaller follow-ups

- Re-run the relaxed-dot artifact on x86_64 Chromium (Yukon/Crabbox
  protocol): V8 lowers the relaxed dot to VPDPBUSD/pmaddubsw there, and the
  Apple Silicon Node signal (+5-10% over the shuffle fix) should widen on
  AVX-VNNI hardware.
- `convert_i8_i16` codegen check (`wasm-tools print | grep -A4 load8x8`) to
  confirm LLVM folds the scalar i64 round-trip into `v128.load64_zero`.
- The browser API facade remains slower than WASI/UCI for raw search; its
  value is control/cancellation/structured results. Revisit only after the
  Emscripten question is settled, since Emscripten would subsume both.
