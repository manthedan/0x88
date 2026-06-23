# CPU engines WASM SIMD audit (Viridithas, Berserk, PlentyChess)

Branch: `feature/cpu-engines-simd-audit` (2026-06-09). Follow-up to
[`reckless_simd_kernel_fixes.md`](./reckless_simd_kernel_fixes.md), applying
the same audit lens — does SIMD actually engage, are the kernels staying in
vector registers, and is the relaxed integer dot provably applicable — to the
other CPU engines.

## Summary table

| Engine | Before | Finding | After (Node, exact parity) |
| --- | --- | --- | --- |
| Viridithas | simd128 build existed | wasm module already correct; relaxed dot provable | 40/40 parity; simd 5.4-5.8x scalar; relaxed +14% over simd at depth 10 |
| Berserk | **fully scalar** (no `-msimd128`) | silent-scalar trap; SSE4.1 path compiles via emulation | 40/40 parity; simd128 3.8x scalar; relaxed 4.1x scalar (+8% over simd) at depth 11 |
| PlentyChess | `-msimd128 -mssse3` already | not silently scalar; relaxed dot untried | out of scope (own worktree); recipe below |
| Stockfish | npm upstream build | not ours to rebuild | n/a (threaded story instead — resource broker) |

## Viridithas (Rust, wasm32-wasip1)

**Audit result: the existing `viridithas-wasip1.patch` wasm SIMD module was
already well-built** — unlike Reckless's first pass, it selects a real
`simd_wasm.rs` module, its `dot_u8_i8_to_i32x4` already uses the
`i32x4_dot_i16x8` + even/odd shuffle pattern (no lane-extract scalarization),
and `nonzero_mask_i32` already returns the correct 1-bit-per-group mask. No
bugs found. Two notes:

- `madd_f32` in the wasm SIMD module is unfused while the wasm scalar
  module's uses `mul_add`; empirically irrelevant (exact parity holds), left
  alone.
- The relaxed integer dot was unexplored. The range proof mirrors Reckless:
  `QA = 255`, `FT_SHIFT = 9` → max activation `255 * 255 >> 9 = 127`, and
  layers.rs itself documents the "0..127" invariant. `madd_u8_to_i32` now
  has a `relaxed-simd` variant using `i32x4_relaxed_dot_i8x16_i7x16_add`;
  `INNER_ARCH` reports `wasm-relaxed-simd128` for diagnosability.

Build: `npm run viridithas:build-relaxed-simd-wasi` (new); evidence in
local-dev artifact: `.local-dev-artifacts/docs/viridithas_simd_kernel_node_bench_2026-06-09_depth7-8.json`
via the generic `reckless_wasi_node_bench.mjs` harness (`--wasm` accepts any
WASI UCI artifact):

- Parity: 40/40 exact (bestmove, score, nodes, PV) scalar vs simd vs relaxed
  at depths 7/8.
- Speed: scalar 78-93k NPS; simd128 455-504k (5.4-5.8x); relaxed
  566k vs simd 495k at isolated depth 10 (+14%).
- Opcode inspection: scalar 0 SIMD opcodes; simd128 3358; relaxed carries 40
  relaxed-dot sites.

## Berserk (C, Emscripten)

**Audit result: the Emscripten build was compiling Berserk's scalar NNUE
fallback** — the emcc invocation had no `-msimd128`, so the engine's
AVX512/AVX2/SSE4.1/NEON kernels were all cfg'd out and not even
auto-vectorization ran. This is the same silent-scalar trap the first
Reckless `+simd128` build hit, in stronger form.

Fix shape: Berserk's SSE4.1 path uses only intrinsics covered by Emscripten's
SSE emulation headers (`maddubs`, `madd_epi16`, `hadd_epi32`, `packs`,
`srai`, `max_epi8`, …), so `-msse4.1 -msimd128` compiles the engine's own
vectorized NNUE with zero new kernel code:

- `npm run berserk:build-simd-emscripten` → `berserk-emscripten-simd128.*`
- `npm run berserk:build-relaxed-simd-emscripten` →
  `berserk-emscripten-relaxed-simd128.*`; the patch adds
  `__wasm_relaxed_simd__` variants of `m128_add_dpbusd_epi32{,x2}` using
  `wasm_i32x4_relaxed_dot_i8x16_i7x16_add`. Range proof: `InputCReLU8`
  outputs `max(packs_epi16(...), 0)` — packs saturates to int8, max clears
  negatives, so activations are in `[0, 127]` and the relaxed dot is exact.
  This pays more here than in the Rust engines because Emscripten's
  `maddubs` emulation (exact saturating semantics) is multi-instruction.
- One watched risk: upstream's `dpbusd_epi32x2` adds two `maddubs` results
  with a wrapping `_mm_add_epi16` (Stockfish-style, relies on trained weight
  constraints). Parity says the constraint holds for this net: no divergence
  in 40/40 pairs.

Evidence
(local-dev artifact: `.local-dev-artifacts/docs/berserk_simd_emscripten_node_bench_2026-06-09_depth9-11.json`,
new `scripts/emscripten_uci_bench.mjs` suite harness):

- Parity: 40/40 exact for simd128 and relaxed vs scalar at depths 9/11.
- Speed (depth-11 medians): scalar 365k NPS → simd128 1.38M (**3.8x**) →
  relaxed 1.50M (**4.1x scalar**). Berserk is now the fastest
  single-threaded wasm engine in the lineup.

## PlentyChess

Lives in the `leelaweb-plentychess-worker` worktree; its port already builds
with `-DARCH_X86 -msimd128 -mssse3`, so it is not silently scalar. Untried:
the same relaxed-dot swap for its dpbusd helpers (it is Stockfish-shaped, so
the `[0, 127]` activation argument needs checking against its quant scheme
before assuming exactness). Recipe: locate the `maddubs`-based dpbusd in its
NNUE, prove the activation range, gate a `__wasm_relaxed_simd__` variant,
build with `-mrelaxed-simd`, and run a fixed-depth suite parity check.

## Validation posture (same as Reckless)

All numbers above are Apple Silicon Node/V8, single rep per cell, exact-pair
parity-gated. **Promoted 2026-06-10** (owner decision,
`feature/promote-simd-defaults`): Reckless, Viridithas, and Berserk now
default through feature-detected speed ladders (relaxed > simd > scalar) with
asset fallback; PlentyChess follows on its own worktree branch. The
promotion case is value-exactness — every variant matched its baseline in
40+/40+ fixed-depth pairs, so an unsupported or slow lowering can only cost
speed, not correctness. x86_64 Chromium numbers remain to be collected
post-promotion.

## Catalog/broker note

All three engines stay `maxThreads: 1` in `ENGINE_RESOURCE_PROFILES`. The
NPS gains here compound with the future threading work: a 4x single-thread
Berserk that later joins the resource broker's thread pool inherits both
wins.
