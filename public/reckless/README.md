# Reckless WASI asset

Generated `*.wasm`, `.nnue`, and corresponding-source archives are intentionally not committed. Production/deploy builds that offer Reckless should generate and publish the scalar fallback, the default SIMD artifact, the experimental relaxed-SIMD artifact exposed in the UI when supported, plus their patched source archives:

```sh
npm run reckless:build-production
```

This writes:

- `public/reckless/reckless.wasm`
- `public/reckless/reckless-simd128.wasm`
- `public/reckless/reckless-relaxed-simd128.wasm`
- `public/reckless/reckless-scalar-corresponding-source.tar.gz`
- `public/reckless/reckless-simd128-corresponding-source.tar.gz`
- `public/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz`

`reckless.wasm` is the scalar fallback. `reckless-simd128.wasm` is the default artifact and uses `-C target-feature=+simd128` plus the wasm NNUE SIMD patch. `reckless-relaxed-simd128.wasm` adds `+relaxed-simd` for browsers that validate the relaxed-SIMD probe; it remains experimental and is never the default. Browser UI code selects SIMD by default when `WebAssembly.validate` confirms SIMD support and falls back to scalar when SIMD is unsupported or the implicit SIMD asset is missing.

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

Reckless is licensed AGPL-3.0 by its upstream authors. This project claims no ownership of Reckless. If you distribute the WASM artifacts, comply with Reckless' license and provide the generated corresponding source archives for the patched builds.
