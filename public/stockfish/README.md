# Stockfish.js assets

Browser Stockfish assets are staged from the `stockfish` npm package (`stockfish@18.0.7`, GPL-3.0).

- Lite single-thread URL: `/stockfish/stockfish-18-lite-single.js` + `/stockfish/stockfish-18-lite-single.wasm`
- Lite single-thread relaxed SIMD candidate: `/stockfish/stockfish-18-lite-single-relaxed.js` + `/stockfish/stockfish-18-lite-single-relaxed.wasm` (feature-detected; baseline artifact remains available as fallback)
- Lite threaded URL: `/stockfish/stockfish-18-lite.js` + `/stockfish/stockfish-18-lite.wasm`
- Full single-thread URL: `/stockfish/stockfish-18-single.js` + `/stockfish/stockfish-18-single.wasm`
- Full threaded URL: `/stockfish/stockfish-18.js` + `/stockfish/stockfish-18.wasm`

The lite single-thread baseline and promoted relaxed SIMD candidate files are committed directly for deterministic deploys; baseline non-lite-single variants are symlinks into `node_modules/stockfish/bin` and are included by the Netlify/Vite build when dependencies are installed. Full single-thread, lite pthread, and full pthread relaxed SIMD artifacts remain reproducible local candidates via `scripts/build_stockfish_relaxed_simd.mjs`, but are not public release artifacts yet.

GPL corresponding source and artifact provenance:

```sh
npm run stockfish:build-relaxed-simd:lite-single
npm run stockfish:build-relaxed-simd:single
npm run stockfish:build-relaxed-simd:lite-threaded
npm run stockfish:build-relaxed-simd:threaded
npm run stockfish:source-archive
npm run stockfish:release-manifest
```

The public release archive is `/stockfish/stockfish-18.0.7-corresponding-source.tar.gz`, and the release manifest is `/stockfish/stockfish-18.0.7.manifest.json`.
