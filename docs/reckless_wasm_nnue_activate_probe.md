# Reckless WASM SIMD NNUE activation probe

This standalone probe targets Reckless' scalar `activate_ft` shape: clipped `i16` PST/threat sums, pairwise product, right shift, and packed `u8` feature output.

Run:

```sh
npm run reckless:probe-nnue-activate-simd
```

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

The probe is not wired into Reckless yet, but it confirms another NNUE sub-kernel maps cleanly to `core::arch::wasm32` SIMD with exact parity and a large standalone speedup. Together with the accumulator probe, this supports focusing a real WASM SIMD backend on feature accumulation plus `activate_ft`/output preparation before touching search control logic.
