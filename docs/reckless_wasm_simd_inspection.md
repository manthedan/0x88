# Reckless WASM SIMD inspection

## 2026-06-04 scalar vs integrated wasm NNUE SIMD artifacts

The repo includes a lightweight code-section SIMD opcode inspector:

```sh
npm run reckless:inspect-simd
```

The command is safe on a clean checkout: generated `public/reckless/*.wasm` files are ignored, so missing artifacts are reported with build-command hints instead of failing. When artifacts are present, it parses the Wasm code section, walks function bodies enough to skip known immediates, and counts real `0xfd` SIMD-prefixed opcodes rather than scanning data/custom sections.

A follow-up `wasm-tools print` check agreed with the custom inspector at the mnemonic level:

```sh
for f in public/reckless/reckless.wasm public/reckless/reckless-simd128.wasm; do
  count=$(wasm-tools print "$f" | rg -c 'v128|i8x16|i16x8|i32x4|i64x2|f32x4|f64x2' || printf 0)
  echo "$f simd_mnemonic_lines=$count"
done
```

```text
public/reckless/reckless.wasm simd_mnemonic_lines=0
public/reckless/reckless-simd128.wasm simd_mnemonic_lines=1181
```

Result after enabling the dedicated wasm NNUE SIMD backend in `npm run reckless:build-simd-wasi`:

```text
public/reckless/reckless.wasm
  bytes=64578655 codeBytes=284147 simdOpcodeCount=0
  families=none
  topOps=none
public/reckless/reckless-simd128.wasm
  bytes=64579258 codeBytes=284933 simdOpcodeCount=1580
  families=v128:706, i32x4:163, f32x4:144, i64x2:129, i16x8/integer:116, i8x16:112, i16x8:100, i32x4/i64x2/integer:85, f64x2:25
```

Interpretation:

- The scalar `reckless.wasm` contains no SIMD opcodes in the code section.
- The SIMD artifact now contains more SIMD instructions than the previous auto-vectorized-only build (`1580` vs earlier `1161` opcode count locally).
- `scripts/build_reckless_wasi.mjs` now patches Reckless' NNUE module selection for `wasm32 + simd128`, adds `src/nnue/simd/wasm32.rs`, and routes `nnue::forward::vectorized` through `core::arch::wasm32` intrinsics.
- The wasm backend covers accumulator add/sub, `activate_ft`, sparse `propagate_l1` dot products, `propagate_l2`, and `propagate_l3`; `find_nnz` remains a simple scalar wasm implementation for now.
- Browser smoke on the rebuilt artifact completed `startpos depth 1` persistent with the expected best move `d2d4`, 42 nodes, and runtime label `persistent`.
- Rotated-FEN browser benchmark evidence is recorded in [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md): versus scalar WASI/UCI, integrated wasm NNUE SIMD was 1.15x faster at depth 7, 2.35x faster at depth 8, and 2.83x faster at depth 9 by warm wall-clock average, with about 1.46x/2.58x/2.72x higher engine-reported NPS.
- Follow-up fixed-depth parity validation across the 20-position rotated suite at depths 7/8/9 matched scalar exactly for best move, score/mate fields, and full PV in 120/120 cold/warm pairs. Movetime 100/250/500ms completed as a sanity/crash check, but exact output parity is not expected under equal clock budgets because SIMD reaches different search frontiers.
- The experimental external-NNUE browser API SIMD artifact (`reckless-browser-api-simd128-external.wasm`) also uses the integrated SIMD backend: `bytes=1260734`, `codeBytes=238447`, `simdOpcodeCount=1279`. Its lower total size comes from removing the embedded NNUE data, not from removing SIMD code.
