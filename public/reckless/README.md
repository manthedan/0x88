# Reckless WASI asset

`reckless.wasm` is intentionally not committed. Build it locally with:

```sh
npm run reckless:build-wasi
```

The script clones `https://github.com/codedeliveryservice/Reckless`, applies a browser/WASI one-shot patch, builds `wasm32-wasip1` without Syzygy tablebases, and writes `public/reckless/reckless.wasm`.

Reckless is licensed AGPL-3.0. If you distribute the WASM artifact, comply with Reckless' license and provide corresponding source for the patched build.
