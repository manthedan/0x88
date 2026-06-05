# Viridithas browser benchmark notes

## 2026-06-04 rotated-FEN WASI/browser comparison

Raw report: [`viridithas_reckless_browser_benchmark_2026-06-04_rotated_depth6-10_movetime100-500.json`](./viridithas_reckless_browser_benchmark_2026-06-04_rotated_depth6-10_movetime100-500.json)

Run configuration:

- Browser/static server: `http://localhost:5181/reckless-benchmark.html`
- Cross-origin isolated: true; SharedArrayBuffer available: true
- Positions: 20-position rotated Ruy Lopez FEN suite
- Budgets: depths 6/8/10 plus movetimes 100/250/500 ms
- Warm repeats: 3
- Hash: 16 MiB; persistent Reckless hash cleared before each timed run
- Variants selected:
  - Reckless Full scalar fallback
  - Reckless Full SIMD
  - Reckless Full browser API SIMD experimental
  - Viridithas experimental
- Modes selected: persistent and one-shot. Browser API one-shot is skipped by design; Viridithas persistent is skipped by design.

Aggregate warm-run means across the 20 FENs:

| Variant | Mode | Budget | Warm avg ms | Avg nodes | Avg NPS |
|---|---:|---:|---:|---:|---:|
| Reckless Full scalar fallback | one-shot | depth 6 | 68.1 | 2,034 | 277,597 |
| Reckless Full scalar fallback | one-shot | depth 8 | 77.4 | 9,230 | 285,419 |
| Reckless Full scalar fallback | one-shot | depth 10 | 102.0 | 24,009 | 289,281 |
| Reckless Full scalar fallback | one-shot | movetime 100ms | 104.3 | 23,244 | 261,504 |
| Reckless Full scalar fallback | one-shot | movetime 250ms | 246.8 | 66,491 | 278,294 |
| Reckless Full scalar fallback | one-shot | movetime 500ms | 497.3 | 133,529 | 273,254 |
| Reckless Full SIMD | one-shot | depth 6 | 180.1 | 2,034 | 595,820 |
| Reckless Full SIMD | one-shot | depth 8 | 189.7 | 9,230 | 746,784 |
| Reckless Full SIMD | one-shot | depth 10 | 207.0 | 24,009 | 742,650 |
| Reckless Full SIMD | one-shot | movetime 100ms | 220.1 | 63,760 | 736,785 |
| Reckless Full SIMD | one-shot | movetime 250ms | 247.1 | 164,726 | 696,221 |
| Reckless Full SIMD | one-shot | movetime 500ms | 495.4 | 348,978 | 717,293 |
| Reckless Full browser API SIMD experimental | persistent | depth 6 | 5.5 | 2,034 | 611,418 |
| Reckless Full browser API SIMD experimental | persistent | depth 8 | 15.8 | 9,230 | 705,697 |
| Reckless Full browser API SIMD experimental | persistent | depth 10 | 34.4 | 24,009 | 742,974 |
| Reckless Full browser API SIMD experimental | persistent | movetime 100ms | 91.5 | 65,160 | 755,006 |
| Reckless Full browser API SIMD experimental | persistent | movetime 250ms | 236.8 | 180,598 | 764,405 |
| Reckless Full browser API SIMD experimental | persistent | movetime 500ms | 486.9 | 366,113 | 752,857 |
| Viridithas experimental | one-shot | depth 6 | 433.8 | 1,738 | 93,128 |
| Viridithas experimental | one-shot | depth 8 | 477.1 | 5,488 | 83,085 |
| Viridithas experimental | one-shot | depth 10 | 630.9 | 17,450 | 78,980 |
| Viridithas experimental | one-shot | movetime 100ms | 531.2 | 8,177 | 75,418 |
| Viridithas experimental | one-shot | movetime 250ms | 674.5 | 18,876 | 73,332 |
| Viridithas experimental | one-shot | movetime 500ms | 937.3 | 36,453 | 71,959 |

Takeaways:

- Viridithas one-shot now runs to the requested depth/time and reports depth/nodes/NPS/PV.
- Viridithas wall-clock is dominated by one-shot startup/decompression overhead, especially on fixed-depth searches. It is ~6.2x slower than Reckless scalar one-shot at depths 6-10 by wall time.
- Viridithas engine-reported NPS is also much lower than Reckless: about 0.27-0.34x scalar one-shot NPS and about 0.10-0.16x SIMD one-shot NPS on this run.
- As movetime grows, fixed one-shot overhead is amortized: Viridithas is ~1.9x slower than Reckless one-shot/browser-API SIMD at movetime 500ms by wall time, but still only ~10-26% of the Reckless NPS depending on baseline.
- This confirms the next likely Viridithas performance target is the NNUE/search hot path (wasm SIMD) plus, separately, avoiding repeated one-shot startup/decompression if Viridithas becomes a production candidate.

## 2026-06-04 Viridithas scalar vs SIMD experiment

Raw report: [`viridithas_simd_browser_benchmark_2026-06-04_rotated_depth6-10_movetime100-500.json`](./viridithas_simd_browser_benchmark_2026-06-04_rotated_depth6-10_movetime100-500.json)

This run used the same 20-FEN suite, budgets, warm repeats, and one-shot mode, comparing only:

- Viridithas scalar experimental: `/viridithas/viridithas.wasm`
- Viridithas SIMD experimental: `/viridithas/viridithas-simd128.wasm`

The SIMD artifact contains wasm SIMD instructions (`simdOpcodeCount=3362`), while the scalar artifact contains none.

Aggregate warm-run means across the 20 FENs:

| Budget | Scalar warm ms | SIMD warm ms | Wall speedup | Scalar NPS | SIMD NPS | NPS speedup |
|---|---:|---:|---:|---:|---:|---:|
| depth 6 | 398.1 | 674.3 | 0.59x | 97,916 | 331,967 | 3.39x |
| depth 8 | 442.0 | 432.0 | 1.02x | 85,742 | 475,274 | 5.54x |
| depth 10 | 662.8 | 433.9 | 1.53x | 74,426 | 490,226 | 6.59x |
| movetime 100ms | 553.4 | 504.8 | 1.10x | 69,775 | 480,390 | 6.88x |
| movetime 250ms | 1063.3 | 659.4 | 1.61x | 38,445 | 462,755 | 12.04x |
| movetime 500ms | 1323.5 | 965.0 | 1.37x | 36,736 | 413,015 | 11.24x |

Takeaways:

- The wasm SIMD NNUE kernels are effective: engine-reported NPS improved about 3.4x-12.0x in this focused run.
- Fixed one-shot overhead still dominates short searches. The depth-6 SIMD artifact reports much higher NPS but worse wall time because startup/decompression/instantiation cost overwhelms the tiny search.
- At deeper or timed budgets, SIMD also improves wall time, with the largest wall gain here at movetime 250ms (~1.6x faster).
- Even with SIMD, Viridithas remains one-shot-only and still pays a large startup cost on every search; persistent/direct API work would be needed before treating it as a delivery candidate.
