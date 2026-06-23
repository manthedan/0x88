# Reckless wasm-opt experiment

Generated with:

```sh
npm run reckless:wasm-opt-experiment
```

The script uses the npm `binaryen` package's `wasm-opt`, validates outputs by compiling them with Node `WebAssembly.compile()`, writes optimized artifacts under ignored `.local_engines/reckless-wasm-opt/`, and records byte/gzip size plus compile timings in local-dev artifact: `.local-dev-artifacts/docs/reckless_wasm_opt_experiment_2026-06-04.json`.

## 2026-06-04 results

| Input | Pass | Raw size | Gzip size | Raw ratio | Gzip ratio | Node compile ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scalar | baseline | 64,578,655 | 43,871,199 | 1.0000 | 1.0000 | 5.45 |
| scalar | `-O3` | 64,476,067 | 43,866,332 | 0.9984 | 0.9999 | 6.74 |
| scalar | `-O4` | 64,476,820 | 43,866,596 | 0.9984 | 0.9999 | 5.61 |
| scalar | `-O3 --enable-simd` | 64,476,076 | 43,866,341 | 0.9984 | 0.9999 | 6.83 |
| simd128 | baseline | 64,578,423 | 43,871,212 | 1.0000 | 1.0000 | 6.15 |
| simd128 | `--enable-simd -O3` | 64,478,876 | 43,865,916 | 0.9985 | 0.9999 | 6.77 |
| simd128 | `--enable-simd -O4` | 64,479,739 | 43,866,615 | 0.9985 | 0.9999 | 7.26 |

Takeaways:

- Binaryen trims only about 0.15-0.16% raw size and about 0.01% gzip size. The embedded NNUE payload dominates artifact size, so wasm-opt does not materially solve cold download size.
- All tested optimized scalar and SIMD artifacts validate.
- Node compile timings are noisy at this size; use browser smoke below for cold one-shot behavior.

## Browser cold one-shot smoke

For a quick browser cold-start/NPS check, the scalar `-O3` and `-O4` artifacts were copied into ignored `/reckless/reckless-wasmopt-O3.wasm` and `/reckless/reckless-wasmopt-O4.wasm`, then run through the existing Reckless WASI worker in an isolated browser, depth 5, startpos. Raw smoke report: local-dev artifact: `.local-dev-artifacts/docs/reckless_wasm_opt_browser_smoke_2026-06-04.json`.

| Variant | Cold wall ms | Nodes | NPS | Bestmove |
| --- | ---: | ---: | ---: | --- |
| baseline | 134.43 | 637 | 115,608 | c2c4 |
| wasm-opt `-O3` | 100.54 | 637 | 103,662 | c2c4 |
| wasm-opt `-O4` | 115.98 | 637 | 102,576 | c2c4 |

This single cold smoke is noisy and too shallow for a product decision, but it shows the optimized artifacts remain functional and do not obviously improve engine-reported NPS. Any follow-up should run the rotated-FEN harness with deeper depth or fixed movetime across multiple cold browser sessions.
