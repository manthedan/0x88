# Stormphrax browser artifacts

Stormphrax 8.0.0 is built as a single-thread Emscripten UCI worker with WebAssembly SIMD. The `undertown.nnue` release network is preloaded into the `.data` sidecar.

Expected artifacts:

- `stormphrax-emscripten.js`
- `stormphrax-emscripten.wasm`
- `stormphrax-emscripten.data`
- `stormphrax-emscripten-single-thread.manifest.json`
- `stormphrax-emscripten-single-thread-corresponding-source.tar.gz`

Build and validate:

```sh
npm run stormphrax:build-emscripten
npm run stormphrax:smoke-emscripten
npm run stormphrax:source-archive
npm run stormphrax:release-manifest
```

Stormphrax is GPL-3.0-or-later. Public JS/WASM/data distribution must include the matching corresponding-source archive and artifact manifest. Upstream: https://github.com/Ciekce/Stormphrax
