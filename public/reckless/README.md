# Reckless WASI asset

Generated `*.wasm` and `.nnue` artifacts are intentionally not committed. Production/deploy builds that offer Reckless should generate and publish both the scalar fallback and the default SIMD artifact:

```sh
npm run reckless:build-wasi
npm run reckless:build-simd-wasi
```

`reckless:build-wasi` writes the scalar fallback `public/reckless/reckless.wasm`. `reckless:build-simd-wasi` writes the default `public/reckless/reckless-simd128.wasm` using `-C target-feature=+simd128` plus the wasm NNUE SIMD patch. Browser UI code selects SIMD by default when `WebAssembly.validate` confirms SIMD support and falls back to scalar when SIMD is unsupported or the implicit SIMD asset is missing.

Build the browser-native API SIMD artifact with an external/cacheable NNUE asset:

```sh
npm run reckless:build-browser-api-simd-external
```

This writes `public/reckless/reckless-browser-api-simd128-external.wasm` plus `public/reckless/reckless-v60-7f587dfb.nnue`.

Build the Lite candidate locally with:

```sh
npm run reckless:build-lite-wasi
```

The scripts clone or reuse `https://github.com/codedeliveryservice/Reckless`, apply browser/WASI patches for one-shot argv searches, isolated persistent-stdin searches, direct browser API exports, optional SIMD NNUE, and optional external NNUE loading, then build `wasm32-wasip1` without Syzygy tablebases and write into `public/reckless/`.

Reckless is licensed AGPL-3.0. If you distribute the WASM artifact, comply with Reckless' license and provide corresponding source for the patched build.
