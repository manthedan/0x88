# Berserk browser artifacts

**Berserk browser artifacts are intentionally NOT committed and NOT deployed.**

The Berserk NNUE (`berserk-9b84c340af7e.nn`, from `jhonnold/berserk-networks`) has
no resolved license/provenance, so this repository must not redistribute it or
anything that embeds it — including the Emscripten preload `.data` and the
corresponding-source archive. See `docs/engine_artifact_distribution.md`.

`.gitignore` covers `public/berserk/*.js`, `*.wasm`, `*.data`, `*.nn`, and
`*corresponding-source*.tar.gz`, and `.gitattributes` carries no `public/berserk/`
LFS rules. Only this README and the release manifest — provenance records with no
network bytes — stay tracked.

## Build them yourself

```sh
npm run berserk:build-emscripten
npm run berserk:build-simd-emscripten
npm run berserk:build-relaxed-simd-emscripten
```

`scripts/build_berserk_emscripten.mjs` clones upstream `jhonnold/berserk` at tag
`14` / commit `8ae895a6151695be4a50d4fb65b0c131659c513a`, applies
`patches/berserk-emscripten.patch`, disables tablebases, and builds a synchronous
single-thread browser API via exported `command()`. When the network is not
already cached in `.local_engines/berserk-nets/`, the script downloads it from
the upstream release URL
(`https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn`)
and verifies both the full SHA-256
(`9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084`) and the hash
prefix embedded in the filename, deleting the download and failing loudly on any
mismatch.

Smoke the result with `npm run berserk:smoke-emscripten` (Node) or
`lab/berserk-smoke.html` (browser worker/adapter).

## Generated layout

Paths owned by `src/lc0/berserkVariants.ts`:

- Emscripten JS glue: `/berserk/berserk-emscripten.js`,
  `/berserk/berserk-emscripten-simd128.js`,
  `/berserk/berserk-emscripten-relaxed-simd128.js`
- Emscripten WASM: the matching `.wasm` beside each glue file
- Emscripten preload data containing the NNUE:
  **`/berserk/berserk-emscripten.data` only** — the packaged network is
  byte-identical across SIMD tiers, so every variant resolves this one file
  through `Module.locateFile`. The build script hashes each freshly built
  `.data` against it and fails rather than publishing diverging bytes.

Planned (not built) WASI/UCI candidates: `/berserk/berserk.wasm`,
`/berserk/berserk-simd128.wasm`, external NNUE `/berserk/berserk-9b84c340af7e.nn`.

## Behavior when the artifacts are absent

`DEPLOYED_BERSERK_PATHS` in `src/lc0/berserkVariants.ts` is empty, so a public
origin resolves every Berserk tier to `missing` without spending probe requests,
and the variant note plus any load error carry `BERSERK_ARTIFACT_BUILD_HINT`
("build locally with `npm run berserk:build-emscripten`") instead of looking like
a broken deploy. A localhost origin still probes normally, so a local build
lights the engine up.

## Release packaging helpers

```sh
npm run berserk:source-archive
npm run berserk:release-manifest
```

These write release files beside the generated artifacts:
`berserk-emscripten-single-thread-corresponding-source.tar.gz` and
`berserk-emscripten-single-thread.manifest.json`. The source archive bundles the
network, so it is ignored and must not be published until the network license is
confirmed.
