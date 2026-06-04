# Reckless WASM NNUE SIMD kernel probe

## 2026-06-04 accumulator add/sub prototype

This is a narrow, standalone WASM SIMD probe for the NNUE accumulator update shape, not a full Reckless backend rewrite. Run it with:

```sh
npm run reckless:probe-nnue-accum-simd
```

What the script does:

- Generates a temporary Rust `wasm32-wasip1` `cdylib` with one scalar kernel and one `#[cfg(target_feature = "simd128")]` / `#[target_feature(enable = "simd128")]` kernel.
- The kernel is exact integer accumulator update logic: `acc[i] = acc[i] + add[i] - sub[i]` over `i16` lanes, matching the type of Reckless NNUE accumulator deltas.
- Builds a scalar wasm artifact and a `-C target-feature=+simd128` artifact.
- Instantiates both artifacts in Node, writes deterministic vectors into wasm memory, requires exact parity, and benchmarks repeated fixed-size updates.
- Reuses `scripts/inspect_wasm_simd.mjs` to confirm that the scalar artifact has no real SIMD opcodes while the SIMD artifact does.

Observed local result:

```json
{
  "target": "wasm32-wasip1",
  "kernel": "i16 accumulator acc += add - sub",
  "length": 1024,
  "iterations": 200000,
  "parity": {
    "mismatches": 0,
    "scalarChecksum": 3186944654,
    "simdChecksum": 3186944654
  },
  "simdOpcodeCounts": {
    "scalar": 0,
    "simd": 18
  },
  "scalar": {
    "elapsedMs": 69.982,
    "lanesPerSecond": 2926465049,
    "checksum": 700488897
  },
  "simd": {
    "elapsedMs": 11.066,
    "lanesPerSecond": 18507697594,
    "checksum": 700488897
  },
  "speedup": 6.324
}
```

Interpretation:

- Exact parity is achievable for the accumulator add/sub kernel because it stays in wrapping integer arithmetic.
- The tiny kernel shows strong standalone speedup, but it is a microbenchmark. Integrating this into Reckless would still need a wasm-specific `nnue::simd` module and full-engine eval/search parity tests.
- The hot-path profile suggests the next kernel with larger whole-engine upside is the NNUE output/clipped-dot path (`Network::evaluate` / `propagate_l1`), but accumulator update is the safest first integration candidate because parity is exact.
