# Reckless WASM SIMD kernel fixes and relaxed integer dot

Branch: `feature/reckless-simd-kernel-fixes` (2026-06-09).

A code audit of the generated `src/nnue/simd/wasm32.rs` shim in
`scripts/build_reckless_wasi.mjs` found one latent bug, one significant
performance defect in the hottest kernel, and one provable-but-unexplored
relaxed SIMD opportunity. All three are addressed on this branch.

## Changes

### 1. `dpbusd` emulation no longer scalarizes (perf fix)

The previous emulation built its per-group sums with eight
`i32x4_extract_lane` calls, scalar adds, and an `i32x4(...)` reconstruction —
forcing values out of SIMD registers inside `propagate_l1`'s inner loop, the
hottest NNUE path (native profile: `Network::evaluate` ~51% of samples). The
replacement keeps everything vectorized:

```rust
let sums = i32x4_add(
    i32x4_shuffle::<0, 2, 4, 6>(pair_lo, pair_hi),
    i32x4_shuffle::<1, 3, 5, 7>(pair_lo, pair_hi),
);
```

This is bit-identical to the old code (exact integer arithmetic, no
saturation: u8×i8 products fit i16, pairwise sums fit i32).

### 2. Relaxed builds use the real hardware dot product

`+relaxed-simd` builds now implement `dpbusd` as a single
`i32x4_relaxed_dot_i8x16_i7x16_add`. The earlier relaxed experiment deferred
this with "not a direct dpbusd replacement without proving activation
ranges"; the range proof is short:

- `activate_ft` outputs `lhs * rhs >> FT_SHIFT` with `lhs` clamped to
  `[0, FT_QUANT]` and `packus` saturating negatives to zero.
- `FT_QUANT = 255`, `FT_SHIFT = 9`, so the max activation is
  `255 * 255 >> 9 = 127`.

Activations therefore always satisfy the i7x16 operand precondition, making
the relaxed dot **exact** (not implementation-defined) on every lowering. The
i16 intermediates of pmaddubsw-style x86 lowerings cannot saturate either
(`2 * 127 * 127 = 32258 < 32767`). On ARM this lowers to a single SDOT; on
AVX-VNNI hardware to VPDPBUSD.

The previously shipped relaxed ops (`f32x4_relaxed_madd`/`relaxed_min`/
`relaxed_max` in the f32 tail) are removed: they were the *only* relaxed ops
in the old artifact, measured no win on x86_64, and regressed on Apple
Silicon. That artifact never exercised the one relaxed op with real upside,
so the earlier "relaxed SIMD is slower" conclusion applied to the artifact,
not to relaxed SIMD.

### 3. `nnz_bitmask` fixed; `find_nnz` vectorized

Old: `u8x16_bitmask(i32x4_gt(x, splat(0)))` — 16 bits (4 per i32 group),
incompatible with every upstream caller, which combines two masks as
`mask0 | mask1 << 4` to index the 256-entry `nnz_table`. It was dead code
only because the wasm `find_nnz` was a scalar group scan that ignored the
table. Now: `i32x4_bitmask(i32x4_ne(x, splat(0)))` — one bit per group,
sign-safe — and `find_nnz` is the NEON-shaped SIMD index extraction over the
existing `nnz_table` (32 bytes → 8-bit mask → precomputed `SparseEntry`).

## Evidence

### Standalone kernel probe (`npm run reckless:probe-nnue-dot-simd`)

Apple Silicon Node, 96 groups × 200k iterations, exact parity for all
kernels (activations in `[0, 127]`, full-range i8 weights):

| kernel | vs scalar | vs previous lane-extract |
| --- | ---: | ---: |
| lane-extract (previous) | 2.66x | 1.00x |
| shuffle (new default) | 4.82x | 1.82x |
| relaxed dot (new relaxed) | 19.67x | 7.41x |

### Full-engine Node validation (`scripts/reckless_wasi_node_bench.mjs`)

20-position rotated FEN suite, one-shot WASI in Node:

- **Parity:** scalar vs fixed SIMD vs relaxed dot: **60/60** exact pairs
  (bestmove, score, nodes, PV) at depths 7/8/9. Old-kernel SIMD vs both new
  artifacts: **16/16** exact at depths 11/12. Identical node counts mean
  identical search trees — every change is value-exact in-engine.
- **Speed** (isolated-process medians, 8 positions, depth 12, Apple Silicon
  Node): old kernels 395k NPS → shuffle fix 469k (**+19%**) → relaxed dot
  489k (**+24%**). Depth-11 same-process run showed +22%/+33%.
  Raw records:
  [`reckless_simd_kernel_fixes_node_bench_2026-06-09_depth7-9.json`](./reckless_simd_kernel_fixes_node_bench_2026-06-09_depth7-9.json),
  [`reckless_simd_kernel_fixes_node_bench_2026-06-09_depth11-12_headtohead.json`](./reckless_simd_kernel_fixes_node_bench_2026-06-09_depth11-12_headtohead.json).

Caveats: Node/V8 on one machine, one-shot mode, single rep per cell. Engine
NPS at shallow depths sits near the 1 ms UCI timer granularity; prefer the
depth 11/12 numbers. Browser promotion still requires the established
`/reckless-benchmark.html` persistent-mode rotated-suite protocol, including
an x86_64 Chromium run (the relaxed dot should help *more* there when V8
lowers to VPDPBUSD, but that is unmeasured).

### Artifact inspection

`npm run reckless:inspect-simd`: the relaxed artifact now contains 12
relaxed-dot sites (`simd extended` family) and no relaxed f32 ops; the
standard artifact's `i32x4.extract_lane` count dropped accordingly. Scalar
artifact remains SIMD-free.

## Promotion posture

**Promoted 2026-06-10** (owner decision, `feature/promote-simd-defaults`):
the default selection ladder is now relaxed dot > simd128 > scalar, gated by
`supportsWasmRelaxedSimd()` / `supportsWasmSimd()` with asset fallback, and
the release pipeline builds all three artifacts plus corresponding-source
archives. Promotion rests on value-exactness (60/60 fixed-depth parity —
worst case on unmeasured hardware is no speedup, never wrongness). x86_64
Chromium browser numbers remain to be collected post-promotion; the relaxed
dot should widen there via VPDPBUSD lowering.

## Not changed, deliberately

- `horizontal_sum` still reduces with lane extracts. It runs once per eval
  (not per group) and any tree-reduction changes f32 summation order;
  parity evidence shows the f32 tail tolerates reordering (scalar
  `propagate_l3` uses sequential fused `mul_add` and still matches), but the
  win is too small to spend parity-risk budget on.
- `convert_i8_i16` still routes 8 weights through a scalar `i64`; LLVM
  appears to fold this into a vector load. Check codegen before touching.

## Remaining exploration tracks

See [`reckless_wasm_next_exploration_notes.md`](./reckless_wasm_next_exploration_notes.md)
for the threats-accumulator swizzle design and the Emscripten-pthreads
threading route.
