# Stockfish.js assets

Browser Stockfish assets are staged from the `stockfish` npm package (`stockfish@18.0.7`, GPL-3.0).

- Lite single-thread URL: `/stockfish/stockfish-18-lite-single.js` + `/stockfish/stockfish-18-lite-single.wasm`
- Lite threaded URL: `/stockfish/stockfish-18-lite.js` + `/stockfish/stockfish-18-lite.wasm`
- Full single-thread URL: `/stockfish/stockfish-18-single.js` + `/stockfish/stockfish-18-single.wasm`
- Full threaded URL: `/stockfish/stockfish-18.js` + `/stockfish/stockfish-18.wasm`

The lite single-thread files are committed directly for deterministic deploys; the other variants are symlinks into `node_modules/stockfish/bin` and are included by the Netlify/Vite build when dependencies are installed.

GPL corresponding source and artifact provenance:

```sh
npm run stockfish:source-archive
npm run stockfish:release-manifest
```

The public release archive is `/stockfish/stockfish-18.0.7-corresponding-source.tar.gz`, and the release manifest is `/stockfish/stockfish-18.0.7.manifest.json`.
