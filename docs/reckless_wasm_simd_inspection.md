# Reckless WASM SIMD inspection

## 2026-06-04 scalar vs `+simd128` artifacts

The local machine did not have `wasm-objdump`/`wasm-tools` on `PATH`, so the repo now includes a lightweight code-section SIMD opcode inspector:

```sh
npm run reckless:inspect-simd
```

The command is safe on a clean checkout: generated `public/reckless/*.wasm` files are ignored, so missing artifacts are reported with build-command hints instead of failing. When artifacts are present, it parses the Wasm code section, walks function bodies enough to skip known immediates, and counts real `0xfd` SIMD-prefixed opcodes rather than scanning data/custom sections.

Result on the current local artifacts:

```text
public/reckless/reckless.wasm
  bytes=64578655 codeBytes=284147 simdOpcodeCount=0
  families=none
  topOps=none
public/reckless/reckless-simd128.wasm
  bytes=64578423 codeBytes=283932 simdOpcodeCount=1161
  families=v128:601, i64x2:129, i8x16:108, i32x4:86, f32x4:76, i32x4/i64x2/integer:45, i16x8:36, i16x8/integer:35, f64x2:25, float compare:20
  topOps=v128.store:225, v128.load:156, v128.const:92, i8x16.shuffle:72, i64x2.extract_lane:64, i32x4.extract_lane:54, v128.and:39, f32x4.add:36, i64x2.replace_lane:33, i64x2.splat:32, i8x16.replace_lane:32, i16x8.add:25, simd.184:20, simd.230:20, simd.67:20, v128.andnot:20, simd.234:20, i32x4.add:17, v128.load32_lane:17, simd.173:13, simd.185:9, f64x2.splat:9, v128.or:8, v128.bitselect:8
```

Interpretation:

- The scalar `reckless.wasm` contains no SIMD opcodes in the code section.
- The `reckless-simd128.wasm` artifact definitely contains Wasm SIMD (`v128`, `i8x16`, `i16x8`, `i32x4`, etc.).
- This does **not** mean Reckless' handwritten native NNUE vector modules are active for wasm: upstream `nnue.rs` still selects scalar NNUE source modules for `wasm32-wasip1` because the vectorized modules are gated on `avx2`/`neon`/`avx512f`.
- Therefore the current win is from LLVM/rustc auto-vectorization and SIMD lowerings, not from a dedicated wasm NNUE backend. A handwritten wasm NNUE kernel still has plausible upside.
