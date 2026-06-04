# Reckless browser benchmark notes

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
- Better follow-up benchmark design: either rotate many positions per warm loop, issue an explicit hash clear/`ucinewgame` between persistent repeats, or use fixed movetime budgets when comparing persistent compute throughput.
