# Viridithas WASI assets

Generated Viridithas browser/WASI builds go here during local experiments.

- Default URL: `/viridithas/viridithas.wasm`
- Build locally: `npm run viridithas:build-wasi`
- Generated `*.wasm` files are intentionally ignored and are not committed.

The build script downloads Viridithas' matching compressed NNUE network, applies `patches/viridithas-wasip1.patch`, and builds a `wasm32-wasip1` artifact.
