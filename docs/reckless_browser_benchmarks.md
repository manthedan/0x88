# Reckless browser benchmark notes

## Harness notes

Use `/reckless-benchmark.html` from the isolated static server after `npm run build:client`. The page now defaults to `ucinewgame` + `isready` before every timed run. This reset happens outside the timed interval, so the wall-clock measurement remains search-only while persistent-mode repeats avoid repeated-position transposition-table reuse. Disable the checkbox only when intentionally measuring warm TT reuse.

## 2026-06-04 isolated headless browser clean depth 7/8/9 smoke

Command surface: `/reckless-benchmark.html` on the isolated static server (`crossOriginIsolated=true`, `SharedArrayBuffer=true`) after `npm run build:client`, with Full scalar and Full `+simd128` artifacts, persistent and one-shot modes, depth budgets 7/8/9, 20 warm repeats plus one cold run, and the default-on persistent clear-hash reset enabled. Positions: `startpos`, an early Italian-like position, and a middlegame position. The table below aggregates warm averages across all three positions; the raw report has per-position rows.

Raw report: [`reckless_browser_benchmark_2026-06-04_clean_depth7-9.json`](./reckless_browser_benchmark_2026-06-04_clean_depth7-9.json).

| Budget | Mode | Variant | Warm avg ms | Avg NPS | Wall ratio vs scalar | NPS ratio vs scalar |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| depth 7 | persistent | Reckless Full | 15.80 | 339,016 | 1.00x | 1.00x |
| depth 7 | persistent | Reckless Full SIMD experimental | 9.92 | 706,881 | 1.59x faster | 2.09x |
| depth 7 | one-shot | Reckless Full | 29.16 | 291,694 | 1.00x | 1.00x |
| depth 7 | one-shot | Reckless Full SIMD experimental | 45.66 | 611,455 | 0.64x as fast | 2.10x |
| depth 8 | persistent | Reckless Full | 29.49 | 326,822 | 1.00x | 1.00x |
| depth 8 | persistent | Reckless Full SIMD experimental | 21.28 | 673,483 | 1.39x faster | 2.06x |
| depth 8 | one-shot | Reckless Full | 40.33 | 306,242 | 1.00x | 1.00x |
| depth 8 | one-shot | Reckless Full SIMD experimental | 49.67 | 651,880 | 0.81x as fast | 2.13x |
| depth 9 | persistent | Reckless Full | 48.23 | 310,949 | 1.00x | 1.00x |
| depth 9 | persistent | Reckless Full SIMD experimental | 33.74 | 653,842 | 1.43x faster | 2.10x |
| depth 9 | one-shot | Reckless Full | 56.76 | 299,956 | 1.00x | 1.00x |
| depth 9 | one-shot | Reckless Full SIMD experimental | 53.82 | 684,750 | 1.05x faster | 2.28x |

Notes:

- With persistent hash clearing enabled, the previous repeated-startpos TT artifact is gone. Persistent SIMD wall-clock improves by roughly 1.4-1.6x across these depth budgets, while engine-reported NPS improves by roughly 2.1x.
- One-shot wall-clock remains noisy and overhead-heavy. SIMD still reports about 2.1-2.3x higher NPS, but the larger/fresh SIMD module and one-shot WASI startup path can hide or reverse the wall-clock gain at shallower budgets.
- Persistent wall-clock is now a better browser compute comparison than the earlier repeated-position run, but NPS is still the cleaner scalar-vs-SIMD compute metric.

## 2026-06-04 isolated headless browser depth 7/8/9 smoke

Command surface: `/reckless-benchmark.html` on the isolated static server (`crossOriginIsolated=true`, `SharedArrayBuffer=true`) with Full scalar and Full `+simd128` artifacts, persistent and one-shot modes, `startpos`, depth budgets 7/8/9, 20 warm repeats plus one cold run.

Raw report: [`reckless_browser_benchmark_2026-06-04_depth7-9.json`](./reckless_browser_benchmark_2026-06-04_depth7-9.json).

| Variant | Mode | Budget | Runs | Cold ms | Warm avg ms | Warm min ms | Warm max ms | Avg NPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reckless Full | persistent | depth 7 | 21 | 97.3 | 0.97 | 0.29 | 6.15 | 564970 |
| Reckless Full | persistent | depth 8 | 21 | 93.3 | 6.31 | 0.31 | 16.95 | 501820 |
| Reckless Full | persistent | depth 9 | 21 | 107.6 | 11.57 | 0.78 | 28.38 | 411139 |
| Reckless Full | one-shot | depth 7 | 21 | 81.3 | 19.95 | 15.12 | 44.32 | 338622 |
| Reckless Full | one-shot | depth 8 | 21 | 97.9 | 20.05 | 18.88 | 23.39 | 355252 |
| Reckless Full | one-shot | depth 9 | 21 | 114.1 | 40.07 | 38.67 | 45.50 | 317926 |
| Reckless Full SIMD experimental | persistent | depth 7 | 21 | 84.2 | 8.29 | 0.28 | 19.59 | 702575 |
| Reckless Full SIMD experimental | persistent | depth 8 | 21 | 90.6 | 18.30 | 0.99 | 24.12 | 780136 |
| Reckless Full SIMD experimental | persistent | depth 9 | 21 | 104.0 | 23.01 | 5.18 | 27.74 | 721113 |
| Reckless Full SIMD experimental | one-shot | depth 7 | 21 | 111.7 | 25.92 | 23.09 | 29.78 | 665188 |
| Reckless Full SIMD experimental | one-shot | depth 8 | 21 | 101.5 | 28.21 | 24.80 | 34.30 | 745323 |
| Reckless Full SIMD experimental | one-shot | depth 9 | 21 | 114.6 | 31.02 | 26.65 | 39.87 | 690498 |

Notes:

- This run explicitly separates persistent from one-shot in the browser runtime and records both wall-clock and engine-reported NPS.
- One-shot is still dominated by WASI process instantiation plus worker/UCI/stdout overhead at these depths, but NPS shows the `+simd128` artifact doing materially more engine work per second.
- Persistent warm repeats of the exact same FEN/depth are not pure compute measurements: Reckless keeps hash state, so repeated searches can become transposition-table hits. Treat persistent wall time here as adapter/hash-reuse UX posture, not scalar-vs-SIMD compute truth.
- This run predates the harness-level clear-hash checkbox. Current follow-up benchmark design should keep that reset enabled, rotate many positions per warm loop, or use fixed movetime budgets when comparing persistent compute throughput.
