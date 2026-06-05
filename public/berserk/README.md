# Berserk browser artifacts

Berserk browser artifacts are planned local/generated assets and are intentionally not committed yet.

Planned URLs owned by `src/lc0/berserkVariants.ts`:

- Scalar WASI/UCI candidate: `/berserk/berserk.wasm`
- SIMD WASI/UCI candidate: `/berserk/berserk-simd128.wasm`
- External NNUE candidate: `/berserk/berserk-9b84c340af7e.nn`
- Source network URL: `https://github.com/jhonnold/berserk-networks/releases/download/networks/berserk-9b84c340af7e.nn`

The first build script should pin upstream `jhonnold/berserk` tag `14` / commit `8ae895a6151695be4a50d4fb65b0c131659c513a`, apply any browser/WASI patches, and write generated blobs here. If distributed, publish the corresponding GPL source/archive next to the artifacts.
