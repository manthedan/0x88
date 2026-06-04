# Reckless WASI asset

Generated `*.wasm` artifacts are intentionally not committed. Build the full/current asset locally with:

```sh
npm run reckless:build-wasi
```

Build the SIMD candidate locally with:

```sh
npm run reckless:build-simd-wasi
```

This writes `public/reckless/reckless-simd128.wasm` using `-C target-feature=+simd128`.

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
