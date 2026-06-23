# Reckless browser benchmark notes

## Harness notes

Use `/lab/reckless-benchmark.html` from the isolated static server after `npm run build:client`. The page now defaults to `ucinewgame` + `isready` before every timed run. This reset happens outside the timed interval, so the wall-clock measurement remains search-only while persistent-mode repeats avoid repeated-position transposition-table reuse. Disable the checkbox only when intentionally measuring warm TT reuse.

The harness also has a `Load 20-position rotated FEN suite` preset. Its run order keeps one engine alive per variant/mode/budget, measures a first pass over every listed position, then rotates each warm pass across the entire suite before repeating any one FEN. Raw and summary outputs track wall-clock ms, engine-reported depth, score/mate, nodes, NPS, best move, PV, runtime label, and cold/first-pass vs warm-pass rows separately. Keep score/PV fields in the JSON/CSV exports when using the page for scalar-vs-SIMD parity validation.

## Current production posture

- `Reckless Full SIMD` (`/reckless/reckless-simd128.wasm`) is the default Reckless variant when WebAssembly SIMD validates successfully.
- `Reckless Full scalar fallback` (`/reckless/reckless.wasm`) remains available for browsers without SIMD support and as an implicit fallback if the default SIMD asset is missing.
- Browser API and external-NNUE variants remain explicit experimental options; they are useful for structured-result/control and cache-lifecycle work, but do not replace SIMD WASI/UCI for production performance yet.
- Merge/deploy readiness depends on the release pipeline running `npm run reckless:build-production`, publishing the ignored generated assets (`reckless.wasm` and `reckless-simd128.wasm`) plus their `*corresponding-source*.tar.gz` archives, and serving `.wasm` as `application/wasm` behind COOP/COEP headers for persistent isolated-worker mode. The checked-in `public/_headers` covers static hosts that honor that file; custom hosts should mirror those headers.

## 2026-06-04 corrected browser API SIMD full benchmark

Command surface: `/lab/reckless-benchmark.html` after rebuilding browser API artifacts with corrected `new_game` history reset. Full scalar WASI/UCI, Full SIMD WASI/UCI, browser API scalar, and browser API SIMD variants; persistent mode only; 20-position rotated suite; depths 7/8/9; 20 warm rotated passes; clear-hash reset enabled. To avoid long-session browser tab loss, each variant was run in a fresh browser session and the extracted rows were combined.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_browser_benchmark_2026-06-04_rotated_fen_depth7-9_api_simd_corrected.json`.

Validation notes:

- Fixed-depth parity passed exactly across all variant pairs checked: scalar WASI/UCI vs SIMD WASI/UCI, scalar WASI/UCI vs browser API scalar, browser API scalar vs browser API SIMD, and SIMD WASI/UCI vs browser API SIMD each matched best move, score/mate fields, and full PV in 1260/1260 cold/warm pairs.
- Browser API SIMD is a real speedup over browser API scalar, but it is not faster than SIMD WASI/UCI in this corrected full run. Keep the browser API variants experimental.
- Browser API scalar is modestly faster than scalar WASI/UCI in this chunked run, but SIMD remains the important production default because SIMD WASI/UCI is faster than both scalar paths.

| Budget | Scalar WASI warm ms | SIMD WASI warm ms | Browser API scalar warm ms | Browser API SIMD warm ms | API SIMD vs SIMD WASI |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth 7 | 25.46 | 10.13 | 21.57 | 11.51 | 0.88x |
| depth 8 | 48.08 | 23.59 | 40.48 | 27.08 | 0.87x |
| depth 9 | 64.82 | 41.82 | 56.33 | 42.75 | 0.98x |

## 2026-06-04 external NNUE browser API SIMD validation

Command surface: `/lab/reckless-benchmark.html?external-nnue-api-simd-validation=1` on the isolated static server after `npm run build:client`, with embedded browser API SIMD and external-NNUE browser API SIMD variants. Persistent/browser-API mode only, 20-position rotated suite, depths 7/8/9, 20 warm rotated passes, and clear-hash reset enabled.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_external_nnue_benchmark_2026-06-04_api_simd_depth7-9.json`. Summary metrics: local-dev artifact: `.local-dev-artifacts/docs/reckless_external_nnue_benchmark_summary_2026-06-04.json`.

Validation notes:

- Fixed-depth parity passed exactly between embedded browser API SIMD and external-NNUE browser API SIMD: 1260/1260 pairs matched best move, score/mate fields, and full PV.
- External NNUE preserves the same node counts. It is a delivery/cache improvement, not a compute improvement.
- This headless run had a few large wall-clock outliers, especially at depth 7, so median and 95%-trimmed averages are more representative than raw warm average for delivery-performance posture.

| Budget | Variant | Cold avg ms | Warm avg ms | Warm median ms | 95%-trimmed warm avg ms | Warm p95 ms | Avg nodes | Avg NPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| depth 7 | Embedded browser API SIMD | 7.05 | 14.23 | 6.75 | 11.37 | 63.25 | 4,536 | 798,664 |
| depth 7 | External-NNUE browser API SIMD | 15.96 | 26.70 | 15.69 | 16.05 | 32.55 | 4,536 | 420,247 |
| depth 8 | Embedded browser API SIMD | 12.67 | 14.31 | 14.26 | 13.31 | 25.37 | 9,230 | 781,586 |
| depth 8 | External-NNUE browser API SIMD | 15.19 | 17.21 | 15.36 | 16.15 | 30.70 | 9,230 | 662,908 |
| depth 9 | Embedded browser API SIMD | 24.90 | 27.79 | 25.49 | 25.68 | 46.87 | 15,614 | 676,950 |
| depth 9 | External-NNUE browser API SIMD | 23.58 | 28.17 | 25.39 | 26.35 | 52.33 | 15,614 | 633,179 |

| Budget | External / embedded warm avg | External / embedded median | External / embedded 95%-trimmed avg |
| --- | ---: | ---: | ---: |
| depth 7 | 1.88x | 2.32x | 1.41x |
| depth 8 | 1.20x | 1.08x | 1.21x |
| depth 9 | 1.01x | 1.00x | 1.03x |

## 2026-06-04 browser API history-reset parity smoke

Command surface: `/lab/reckless-benchmark.html` after rebuilding scalar and SIMD browser API artifacts with the corrected direct-API `new_game` implementation. Full scalar WASI/UCI, Full SIMD WASI/UCI, browser API scalar, and browser API SIMD variants; persistent mode only; 20-position rotated suite; depths 7/8/9; one warm rotated pass; clear-hash reset enabled.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_browser_api_history_reset_smoke_2026-06-04.json`.

Validation notes:

- The browser API `new_game` path now mirrors UCI reset more closely by clearing correction-history tables in addition to thread state and TT. Before this fix, browser API scalar/SIMD matched on the first pass but diverged on warm clear-hash runs, because history tables persisted between positions.
- Fixed-depth parity passed exactly for all checked pairs in this smoke: scalar WASI/UCI vs SIMD WASI/UCI, scalar WASI/UCI vs browser API scalar, and browser API scalar vs browser API SIMD each matched best move, score/mate fields, and full PV in 120/120 cold/warm pairs.
- With clear-hash correctness restored, browser API SIMD was roughly comparable but slightly slower than SIMD WASI/UCI in this one-warm-pass smoke; keep the browser API variants experimental pending a full corrected 20-warm-pass benchmark.

| Budget | SIMD WASI warm ms | Browser API SIMD warm ms |
| --- | ---: | ---: |
| depth 7 | 5.41 | 5.71 |
| depth 8 | 11.06 | 11.38 |
| depth 9 | 18.79 | 19.20 |

## 2026-06-04 scalar-vs-SIMD parity validation

Command surface: `/lab/reckless-benchmark.html` on the isolated static server after `npm run build:client`, with Full scalar WASI/UCI and Full integrated wasm NNUE SIMD WASI/UCI. Persistent mode only, 20-position rotated Ruy Lopez suite, depth budgets 7/8/9, movetime budgets 100/250/500ms, one warm rotated pass, and default-on clear-hash reset.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_simd_parity_validation_2026-06-04_depth7-9_movetime100-500.json`.

Validation notes:

- Fixed-depth parity passed exactly: 120/120 scalar-vs-SIMD pairs matched best move, score/mate fields, and full PV (`20 positions × 3 depths × cold/warm`).
- Fixed-movetime exact parity is not expected because the faster SIMD artifact searches different node/depth frontiers under the same clock budget. The movetime run was used as a crash/sanity check and completed 120/120 scalar-vs-SIMD pairs. Same-best-move counts were 36/40 at 100ms, 38/40 at 250ms, and 38/40 at 500ms.
- This validation used one warm pass, so the wall-time aggregates are sanity numbers rather than the deeper 20-warm-pass performance benchmark. It still showed the expected depth 8/9 SIMD speedup and movetime NPS gains while preserving fixed-depth output parity.

| Budget | Warm scalar ms | Warm SIMD ms | Wall speedup | NPS speedup |
| --- | ---: | ---: | ---: | ---: |
| depth 7 | 15.56 | 18.45 | 0.84x | 2.50x |
| depth 8 | 32.23 | 15.84 | 2.03x | 2.59x |
| depth 9 | 57.81 | 20.65 | 2.80x | 2.80x |
| movetime 100ms | 89.33 | 87.26 | 1.02x | 2.50x |
| movetime 250ms | 239.10 | 236.84 | 1.01x | 2.70x |
| movetime 500ms | 489.43 | 486.69 | 1.01x | 2.64x |

## 2026-06-04 rotated-FEN harness smoke

Command surface: isolated static server, Full scalar artifact, persistent mode, 20-position Ruy Lopez suite, depth 1, one warm rotated pass, default-on clear-hash reset. This is a harness-validation smoke, not a final performance comparison.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_browser_benchmark_2026-06-04_rotated_fen_smoke.json`. The report has 40 raw rows and 20 summary rows, with nodes and NPS captured for every position.

## 2026-06-04 rotated-FEN browser API + integrated SIMD NNUE depth 7/8/9

Command surface: `/lab/reckless-benchmark.html` on the isolated static server (`crossOriginIsolated=true`, `SharedArrayBuffer=true`) after `npm run build:client`, with Full scalar WASI/UCI, Full integrated wasm NNUE SIMD WASI/UCI, and Full browser API artifacts. Persistent mode only, 20-position rotated Ruy Lopez suite, depth budgets 7/8/9, 20 warm rotated passes, and default-on clear-hash reset. The table aggregates all warm rows for each variant/budget: 20 positions × 20 warm passes = 400 warm rows per line.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_browser_benchmark_2026-06-04_rotated_fen_depth7-9_api_simd.json`.

| Budget | Variant | Runtime | Warm avg ms | Warm median ms | Avg nodes | Avg NPS | Wall ratio vs scalar | NPS ratio vs scalar |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| depth 7 | Reckless Full | persistent | 21.24 | 17.63 | 4,536 | 298,118 | 1.00x | 1.00x |
| depth 7 | Reckless Full SIMD experimental | persistent | 18.41 | 14.67 | 4,536 | 435,738 | 1.15x faster | 1.46x |
| depth 7 | Reckless Full browser API experimental | browser API | 24.26 | 20.29 | 5,396 | 251,531 | 0.88x as fast | 0.84x |
| depth 8 | Reckless Full | persistent | 35.59 | 33.01 | 9,230 | 279,838 | 1.00x | 1.00x |
| depth 8 | Reckless Full SIMD experimental | persistent | 15.16 | 15.17 | 9,230 | 722,707 | 2.35x faster | 2.58x |
| depth 8 | Reckless Full browser API experimental | browser API | 43.73 | 38.38 | 9,432 | 234,427 | 0.81x as fast | 0.84x |
| depth 9 | Reckless Full | persistent | 66.22 | 57.59 | 15,614 | 266,728 | 1.00x | 1.00x |
| depth 9 | Reckless Full SIMD experimental | persistent | 23.40 | 21.18 | 15,614 | 724,185 | 2.83x faster | 2.72x |
| depth 9 | Reckless Full browser API experimental | browser API | 82.50 | 70.63 | 15,069 | 206,863 | 0.80x as fast | 0.78x |

Notes:

- Integrated wasm NNUE SIMD is a real browser benefit in this run. It is modest/noisy at depth 7, then clearly faster at depth 8/9, with matching scalar node counts and about 2.6-2.7x higher engine-reported NPS at the deeper budgets.
- The experimental browser API did **not** produce a speedup in this implementation. It bypasses UCI text formatting/parsing, but the current direct facade still runs through the same single-threaded search/NNUE code and `wasm32-wasip1` clock imports, and its measured wall/NPS are slower than persistent WASI/UCI on this suite. Keep it experimental; its main value remains as a cleaner path to future cancellation/control and non-UCI structured results, not a proven latency win yet.
- Keep wall-clock and engine-reported NPS separate. SIMD improves both here, while browser API currently improves neither.

## 2026-06-04 isolated headless browser clean depth 7/8/9 smoke

Command surface: `/lab/reckless-benchmark.html` on the isolated static server (`crossOriginIsolated=true`, `SharedArrayBuffer=true`) after `npm run build:client`, with Full scalar and Full `+simd128` artifacts, persistent and one-shot modes, depth budgets 7/8/9, 20 warm repeats plus one cold run, and the default-on persistent clear-hash reset enabled. Positions: `startpos`, an early Italian-like position, and a middlegame position. The table below aggregates warm averages across all three positions; the raw report has per-position rows.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_browser_benchmark_2026-06-04_clean_depth7-9.json`.

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

Command surface: `/lab/reckless-benchmark.html` on the isolated static server (`crossOriginIsolated=true`, `SharedArrayBuffer=true`) with Full scalar and Full `+simd128` artifacts, persistent and one-shot modes, `startpos`, depth budgets 7/8/9, 20 warm repeats plus one cold run.

Raw report: local-dev artifact: `.local-dev-artifacts/docs/reckless_browser_benchmark_2026-06-04_depth7-9.json`.

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
