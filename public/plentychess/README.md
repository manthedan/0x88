# PlentyChess browser artifacts

This directory is reserved for generated local PlentyChess Emscripten artifacts.

Expected local proof-build outputs:

- `plentychess-emscripten.js`
- `plentychess-emscripten.wasm`
- `plentychess-emscripten.data`

Build with:

```sh
npm run plentychess:build-emscripten
npm run plentychess:smoke-emscripten
```

Use `/plentychess-smoke.html` in the dev server for browser worker lifecycle smoke.

The generated blobs are ignored. PlentyChess is GPL-3.0, so do not distribute the generated JS/WASM/data artifacts until the project has an explicit corresponding-source/archive policy for this engine and its processed NNUE asset.
