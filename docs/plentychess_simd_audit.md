# PlentyChess WASM SIMD audit

Branch: `feature/plentychess-simd-audit` (2026-06-10). Companion to
`docs/cpu_engines_simd_audit.md` on `feature/cpu-engines-simd-audit`
(Reckless/Viridithas/Berserk), applying the same lens: does SIMD actually
engage, is anything silently scalar, and is the relaxed integer dot provably
exact.

## Findings

1. **The integer L1 path was healthy.** `-DARCH_X86 -msimd128 -mssse3`
   compiles the engine's own SSSE3 `dpbusdEpi32{,x2}` (maddubs pattern)
   through Emscripten's SSE emulation — PlentyChess was never in Berserk's
   silently-scalar situation.
2. **The entire f32 tail was scalar.** All three f32 layer sections in
   `nnue.cpp` (L1 normalisation/activation, L2 propagation, L3 propagation)
   are gated on `__FMA__ || __AVX2__ || AVX512 || ARCH_ARM` — none of which
   emcc defines — so they ran scalar `std::fma` loops. This was the dominant
   wasm-specific drag.
3. **`convertEpi8Epi16` (accumulator weight extension) used a 5-op
   sign-extension fallback** because its fast path is gated on `__FMA__` as
   a modern-x86 proxy; the single-op `_mm_cvtepi8_epi16` is SSE4.1 and maps
   to one wasm extend instruction.
4. **`vecNNZ` uses signed `cmpgt` against zero** — the same latent pattern
   as Reckless's old `nnz_bitmask` — but it is provably safe here because
   activations max at 127 (see below), so packed i32 group lanes are never
   negative. No change made; noted for upstream awareness.
5. **The relaxed dot range proof carries over verbatim**:
   `INPUT_QUANT = 255`, `INPUT_SHIFT = 9` → max pairwise activation
   `255 * 255 >> 9 = 127`, `packusEpi16` saturates negatives to zero. Same
   invariant as Reckless and Viridithas (all three engines share the
   255/shift-9 quant scheme).

## Changes (regenerated `plentychess-emscripten.patch` + build script)

- `simd.h`: `convertEpi8Epi16` gate accepts `__SSE4_1__`; relaxed
  `dpbusdEpi32{,x2}` via `wasm_i32x4_relaxed_dot_i8x16_i7x16_add` under
  `__wasm_relaxed_simd__`; `fmaddPs` via `wasm_f32x4_relaxed_madd` (lowers
  to a fused FMA on modern hardware, matching the scalar `std::fma`
  fallback elementwise).
- `nnue.cpp`: the three f32 vector gates extended with
  `|| defined(__wasm_relaxed_simd__)`, so the relaxed build runs the
  engine's vectorized f32 tail.
- Build variants: `plentychess:build-sse41-emscripten` (default +
  `-msse4.1`) and `plentychess:build-relaxed-simd-emscripten`
  (`-msse4.1 -mrelaxed-simd`). The default artifact's flags are unchanged.
- The build script now uses a shallow pinned-ref fetch (the full
  clone + `--tags` fetch was flaky over slow links).

## Evidence

Node suite via the new generic `scripts/emscripten_uci_bench.mjs`
(20-position rotated FEN suite; raw record
`.local-dev-artifacts/docs/plentychess_simd_emscripten_node_bench_2026-06-10_depth9-11.json`):

- **Parity: 40/40 exact** (bestmove, score, nodes, PV) for both sse41 and
  relaxed vs the current default at depths 9/11. The relaxed result also
  validates the f32-tail swap empirically: fused relaxed madd matched the
  scalar `std::fma` path in every pair.
- **Speed** (isolated-process medians, 8 positions, depth 12, Apple Silicon
  Node): default 603k NPS → sse41 678k (**+12%**) → relaxed 992k
  (**+64%**).
- Opcode inspection: all artifacts ~2700 SIMD opcodes; the relaxed one
  carries 12 relaxed sites (dot + madd).

Verified live in isolated Chrome: `?plentyChessVariant=emscripten-relaxed`
loads, validates, and analyzes.

## Caveats / next

- **Promoted 2026-06-10** (owner decision): the default ladder is now
  relaxed > sse41 with asset fallback to the base Emscripten build, gated by
  `supportsWasmRelaxedSimd()`. The promotion case is value-exactness (40/40
  fixed-depth parity); x86_64 Chromium numbers remain to be collected
  post-promotion.
- The relaxed f32 tail's exactness vs `std::fma` relies on relaxed madd
  lowering fused; on pre-FMA x86 hardware results may differ in ulps. Fine
  for an experimental variant; revisit before promotion.
- The non-relaxed sse41 build leaves the f32 tail scalar on purpose (an
  unfused vector tail would break exact parity with the shipped default).
  If x86_64 validation confirms the relaxed build, the sse41 middle step
  can be dropped.
