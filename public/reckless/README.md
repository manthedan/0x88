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

Build the Lite candidate locally with:

```sh
npm run reckless:build-lite-wasi
```

The script clones `https://github.com/codedeliveryservice/Reckless`, applies browser/WASI patches for one-shot argv searches and isolated persistent-stdin searches, optionally applies a known-compatible NNUE shape override, builds `wasm32-wasip1` without Syzygy tablebases, and writes into `public/reckless/`.

Reckless is licensed AGPL-3.0. If you distribute the WASM artifact, comply with Reckless' license and provide corresponding source for the patched build.
