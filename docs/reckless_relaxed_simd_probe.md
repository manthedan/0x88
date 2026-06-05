# Reckless relaxed SIMD probe

Status: experimental branch `feature/reckless-relaxed-simd`.

## Report assessment

The recommendation to compile an additional Relaxed SIMD artifact is directionally useful, but the claimed **2x-3x over standard SIMD128** is not safe to assume for Reckless.

Reasons:

- WebAssembly Relaxed SIMD is a separate feature from baseline `simd128`; unsupported browsers fail validation/compilation, so it must remain a distinct artifact with runtime feature detection and fallback.
- Relaxed SIMD only helps where emitted bytecode uses relaxed operations. Compiling with `+relaxed-simd` alone does not automatically replace the existing integer NNUE kernels with faster relaxed dot products.
- Reckless' hottest WASM NNUE path uses integer packed operations including an emulated unsigned-byte × signed-byte dot product. The available relaxed dot intrinsic is signed/i7-shaped and is not a direct `dpbusd` replacement without proving activation ranges and exact score/search parity.
- Relaxed floating-point operations may change rounding/NaN/signed-zero behaviour. That is acceptable only if fixed-depth parity and practical playing/search behaviour are validated.

## Initial implementation

This branch adds a separate build/selection path instead of changing the default:

- `npm run reckless:build-relaxed-simd-wasi`
- artifact: `public/reckless/reckless-relaxed-simd128.wasm` (ignored/generated)
- UI benchmark checkbox: **Reckless Full Relaxed SIMD experimental**
- URL variant alias: `?recklessVariant=relaxed-simd`
- runtime probe: `supportsWasmRelaxedSimd()` validates a tiny module containing `f32x4.relaxed_madd`

The generated `wasm32` Reckless NNUE SIMD shim currently uses relaxed operations only where the semantics are relatively narrow and easy to inspect:

- `mul_add_f32`: `f32x4_relaxed_madd` when `target_feature = "relaxed-simd"`
- `clamp_f32`: `f32x4_relaxed_min/max` when `target_feature = "relaxed-simd"`

The default remains regular SIMD128 because the existing scalar-vs-SIMD parity and benchmark evidence is stronger.

## Initial build/inspection result

`npm run reckless:build-relaxed-simd-wasi` built successfully on this branch.

Inspection with `node scripts/inspect_wasm_simd.mjs public/reckless/reckless-simd128.wasm public/reckless/reckless-relaxed-simd128.wasm` confirmed the relaxed artifact validates in Node/Chrome and contains relaxed opcodes:

- standard SIMD: 64,579,258 bytes; 1,580 SIMD opcodes
- relaxed SIMD: 64,579,272 bytes; 1,554 SIMD opcodes
- relaxed top ops include `f32x4.relaxed_madd:36`, `f32x4.relaxed_max:21`, and `f32x4.relaxed_min:20`

A very small browser smoke on `http://localhost:5205/reckless-benchmark.html` compared persistent startpos depth 7 with 5 warm repeats. Raw report: [`reckless_relaxed_simd_smoke_2026-06-05_startpos_depth7.json`](./reckless_relaxed_simd_smoke_2026-06-05_startpos_depth7.json).

- standard SIMD warm wall avg: 3.577 ms; avg NPS: 844,531
- relaxed SIMD warm wall avg: 3.643 ms; avg NPS: 887,411
- best move, score, nodes, and PV matched on this tiny smoke

This smoke is not sufficient for promotion; it mainly proves the artifact compiles, validates, loads in an isolated browser, emits relaxed opcodes, and produces a matching shallow startpos result.

## Yukon x86_64 Chromium probe

A remote Crabbox run on `yukon` used HeadlessChrome 148 on Linux x86_64. Note: `yukon` reported an AMD Ryzen 9 3900XT rather than Intel, so this is an x86_64/Chrome data point, not an Intel-specific one.

Raw report: [`reckless_relaxed_simd_yukon_chromium_2026-06-05_depth7-8_warm10.json`](./reckless_relaxed_simd_yukon_chromium_2026-06-05_depth7-8_warm10.json). Summary: [`reckless_relaxed_simd_yukon_chromium_2026-06-05_depth7-8_warm10_summary.json`](./reckless_relaxed_simd_yukon_chromium_2026-06-05_depth7-8_warm10_summary.json).

Configuration:

- Chrome runtime validated Relaxed SIMD and was cross-origin isolated with `SharedArrayBuffer`.
- Persistent mode only.
- 20-position rotated FEN suite.
- Depth 7 and depth 8.
- 10 warm repeats with hash clearing.
- 880 raw rows; 440 exact SIMD-vs-relaxed comparison pairs.

Results:

- Parity: 440/440 exact pairs for best move, depth, score, nodes, and PV.
- Weighted warm wall-clock averages were lower for relaxed SIMD (`depth 7`: 17.10 ms vs 23.30 ms; `depth 8`: 30.55 ms vs 36.24 ms), but this was skewed by large standard-SIMD outliers.
- Paired robust timing did **not** show a clean win: relaxed SIMD median wall ratio was 1.086 at depth 7 and 1.069 at depth 8; relaxed was slower in most paired warm comparisons.
- Engine-reported NPS was also lower for relaxed SIMD on this run.

Interpretation: the x86_64 browser probe supports parity, but not promotion. Relaxed SIMD remains useful as an explicit experimental artifact; the current measured speed signal is mixed/noisy and not a reliable improvement over standard SIMD128.

## Validation requirements before promotion

1. Build standard SIMD and relaxed SIMD artifacts from the same source ref.
2. Inspect relaxed artifact and confirm relaxed opcodes are actually present.
3. Run fixed-depth parity against standard SIMD across the rotated FEN suite, at least depths 7/8/9 and movetime 100/250/500 where feasible.
4. Benchmark persistent mode separately from one-shot mode.
5. Keep wall-clock timing and engine-reported NPS separate.
6. Only consider default promotion if relaxed SIMD is both materially faster and parity-safe; otherwise keep it as an experimental benchmark variant.
