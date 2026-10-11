# Stockfish.js 19 browser artifacts

Pinned npm package: `stockfish@19.0.1`, upstream `v19.0.1` / `0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6`.
Engine code: GPL v3 (upstream file notices permit later versions); npm metadata: GPL-3.0. Non-engine upstream code is MIT.

| Flavor | JS + sibling WASM |
| --- | --- |
| Lite single | `stockfish-19-lite-single` |
| Feature-detected relaxed lite single | `stockfish-19-lite-single-relaxed` |
| Lite pthread | `stockfish-19-lite` |
| Full single | `stockfish-19-single` |
| Full pthread | `stockfish-19` |

The full/pthread baseline pairs are symlinks into this checkout's npm package. `npm ci` restores them. Lite single baseline and relaxed pairs are tracked; full/threaded relaxed candidates stay in ignored local artifacts, not the release manifest.

Threaded glue must stay same-origin in the app shell; its WASM may use the artifact host. Keep COOP/COEP enabled. SF19 uses the Emscripten 6.x same-script `em-pthread` bootstrap, not the SF18 embedded helper-worker shim.

The release source archive is `stockfish-19.0.1-corresponding-source.tar.gz`; the matching manifest is `stockfish-19.0.1.manifest.json`. The archive includes both hash-verified networks: full `nn-1a298aa575a0.nnue` and lite `nn-61e7af4bb97d.nnue`. Lite has a distinct feature architecture; it cannot be checked by loading its net into unmodified official SF19.

```sh
npm ci
npm run stockfish:build-relaxed-simd:lite-single
# Optional local candidates: :single, :lite-threaded, :threaded
npm run stockfish:source-archive
npm run stockfish:release-manifest
```

Builder outputs: `.local-dev-artifacts/stockfish-relaxed/repro/dist/`. Default toolchain: `emscripten/emsdk:6.0.9`, `-O3 -flto=full`; `--opt O2|O3 --lto on|off` enables the audit matrix. Use `STOCKFISH_NNUE_DIR` or the bundled archive nets for offline builds. Full embedding requires more than a 2 GiB Docker VM.

See `docs/stockfish_19_upgrade.md` for parity, strength, speed and release requirements; `docs/stockfish_relaxed_simd_experiment.md` retains the historical SF18 experiment.
