# Berserk browser artifacts

The Berserk Emscripten artifacts in this directory are committed and deployed:
three JS/WASM SIMD tiers share one canonical NNUE preload, and the matching
manifest and corresponding-source archive ship beside them.

## Distribution basis

Berserk tag 14 is GPL-3.0. Its build names
`berserk-9b84c340af7e.nn` as `EVALFILE`, downloads that exact network during a
normal build, verifies its SHA-256, and cannot produce a working engine without
it. This project therefore treats the network as part of the GPL work's
Corresponding Source and includes it in the source archive.

The separate `jhonnold/berserk-networks` repository has no standalone licence
file. Upstream confirmation has been requested; the reasoning and the contained
reversal procedure are recorded in `docs/engine_artifact_distribution.md`.

## Rebuild

```sh
npm run berserk:build-emscripten
npm run berserk:build-simd-emscripten
npm run berserk:build-relaxed-simd-emscripten
```

`scripts/build_berserk_emscripten.mjs` clones upstream `jhonnold/berserk` at tag
`14` / commit `8ae895a6151695be4a50d4fb65b0c131659c513a`, applies
`patches/berserk-emscripten.patch`, disables tablebases, and builds a synchronous
single-thread browser API via exported `command()`. When the network is not
cached in `.local_engines/berserk-nets/`, the script downloads it from
`https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn`.
It verifies the full SHA-256
`9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084` and the
hash prefix embedded in the filename.

Smoke the result with `npm run berserk:smoke-emscripten` (Node) or
`lab/berserk-smoke.html` (browser worker/adapter).

## Published layout

Paths owned by `src/lc0/berserkVariants.ts`:

- JS glue: `/berserk/berserk-emscripten.js`,
  `/berserk/berserk-emscripten-simd128.js`, and
  `/berserk/berserk-emscripten-relaxed-simd128.js`
- per-tier WASM beside each glue file
- one shared preload: `/berserk/berserk-emscripten.data`

The packaged network is byte-identical across SIMD tiers. Every variant resolves
the canonical `.data` through `Module.locateFile`; the build script hashes each
freshly built preload and refuses to publish divergent bytes.

Planned, unbuilt WASI/UCI candidates remain `/berserk/berserk.wasm`,
`/berserk/berserk-simd128.wasm`, and external NNUE
`/berserk/berserk-9b84c340af7e.nn`.

## Release records

```sh
npm run berserk:source-archive
npm run berserk:release-manifest
```

These produce
`berserk-emscripten-single-thread-corresponding-source.tar.gz` and
`berserk-emscripten-single-thread.manifest.json`. The committed versions must be
regenerated together whenever the engine, patch, build recipe, or network
changes.
