# Viridithas WASI assets

Viridithas browser/WASI builds are staged here for deterministic deploys.

- Scalar URL: `/viridithas/viridithas.wasm`
- SIMD URL: `/viridithas/viridithas-simd128.wasm`
- Build locally: `npm run viridithas:build-wasi` and `npm run viridithas:build-simd-wasi`
- Source/provenance archive: `npm run viridithas:source-archive`
- Release manifest: `npm run viridithas:release-manifest`

The committed release archive is `/viridithas/viridithas-wasip1-corresponding-source.tar.gz`, and the release manifest is `/viridithas/viridithas-wasip1.manifest.json`.

The build script downloads Viridithas' matching compressed NNUE network, applies `patches/viridithas-wasip1.patch`, and builds a `wasm32-wasip1` artifact.
