# Reckless WASM SIMD NNUE activation probe

This standalone probe targets Reckless' scalar `activate_ft` shape: clipped `i16` PST/threat sums, pairwise product, right shift, and packed `u8` feature output.

The standalone script and package command were retired after the SIMD backend
was integrated. This document retains the measured result.

Result from 2026-06-04:

```json
{
  "target": "wasm32-wasip1",
  "kernel": "NNUE activate_ft clipped pair-product u8 feature output",
  "length": 384,
  "iterations": 250000,
  "parity": {
    "mismatches": 0,
    "scalarChecksum": 1759800980,
    "simdChecksum": 1759800980
  },
  "simdOpcodeCounts": {
    "scalar": 0,
    "simd": 76
  },
  "scalar": {
    "elapsedMs": 79.292,
    "lanesPerSecond": 1210712918,
    "checksum": 1759800980
  },
  "simd": {
    "elapsedMs": 9.594,
    "lanesPerSecond": 10005993173,
    "checksum": 1759800980
  },
  "speedup": 8.265
}
```

At the 2026-06-04 probe stage, this was not wired into Reckless yet. It
confirmed that this sub-kernel mapped to `core::arch::wasm32` SIMD with exact
parity and a large standalone speedup. The backend was subsequently
integrated; see [`reckless_wasm_simd_inspection.md`](./reckless_wasm_simd_inspection.md)
and `scripts/build_reckless_wasi.mjs`. The 8.265× result above remains a
microbenchmark, not an engine-wide speedup.
