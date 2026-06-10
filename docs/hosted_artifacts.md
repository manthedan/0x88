# Hosted browser engines and model artifacts

Last updated: 2026-06-10

This is the authoritative, human-readable inventory for browser engine/model
artifacts that the app knows how to serve. It complements the machine-readable
`public/artifact-index.json` and the per-engine manifests under `public/*`.

## Status terms

- **Tracked/deployable**: committed under `public/`, or committed as a symlink
  that the release build must resolve. These are part of a normal repository
  checkout/build when their dependency/source targets exist.
- **Tracked manifest only**: metadata is committed, but the large blob is not.
  A deploy that wants to serve the blob must provide it separately.
- **Ignored/local**: generated or symlinked for local smoke testing, but hidden
  by `.gitignore` or local exclude rules. Do not assume a public deployment has
  it.
- **Research-only**: exposed only behind explicit opt-in or smoke tooling; not a
  stable/default runtime.

## Current inventory

| Family / UI surface | Current version or source anchor | Served URLs / manifest | Deployment status | Notes |
| --- | --- | --- | --- | --- |
| LC0 Small default | `t1-256x10-distilled-swa-2432500` | `/models/lc0/manifest.json`; ONNX batch 1/4/8/16/32 f16 + batch1 f32; lc0web pack `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json` | Tracked symlinks/pack metadata and shards | Stable LC0 model family. ORT WebGPU/WASM remains the default evaluator path. The custom lc0web pack is opt-in/benchmarked, not a default replacement. |
| LC0 BT4 Arena option | `BT4-1024x15x32h-swa-6147500-policytune-332` (`BT4-it332`) | Current Arena URL: `/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx`; b1/b4/b8 entries in `/models/lc0/manifest.json` | Tracked manifest only; ONNX blobs/symlinks are ignored/local | Browser BT4 uses the batch-4 f16 ONNX export, WebGPU-only worker, leaf batch size 4, pipeline depth 1, and tree reuse. Deploys must stage the large BT4 ONNX artifact explicitly. |
| LC0 legacy BT4 | `BT4-1024x15x32h-swa-6147500` | `/models/lc0/BT4-1024x15x32h-swa-6147500.batch1.f16.onnx` in `/models/lc0/manifest.json` | Tracked manifest only; blob/symlink ignored/local | Kept for historical/research compatibility. It is not the current Arena BT4 target. |
| LC0 TVMJS whole-model WebGPU | Small `t1-256x10-distilled-swa-2432500`; BT4-it332 research artifacts exist locally when staged | `/runtimes/lc0-tvmjs-webgpu/...` | Ignored/local; no tracked runtime artifacts | Research-only. Source/tooling is in the repo, but `public/runtimes/lc0-tvmjs-webgpu/` artifacts are intentionally ignored and not part of the stable runtime registry. |
| Stockfish | `stockfish@18.0.7`; upstream `nmrugg/stockfish.js` commit `32d4b5ae40c01db88219bfbe2b82dbe6dec93832` | `/stockfish/stockfish-18-lite-single.js`, `/stockfish/stockfish-18-lite.js`, `/stockfish/stockfish-18-single.js`, `/stockfish/stockfish-18.js`; manifest `/stockfish/stockfish-18.0.7.manifest.json` | Tracked/deployable | Lite single-thread is the default Stockfish opponent. GPL corresponding source archive is committed at `/stockfish/stockfish-18.0.7-corresponding-source.tar.gz`. |
| Viridithas | `cosmobobak/viridithas` commit `20d7402065cae084715183e019fdd18089e2dfac`; network `atlantis-b800.nnue.zst` v106 | `/viridithas/viridithas.wasm`, `/viridithas/viridithas-simd128.wasm`; manifest `/viridithas/viridithas-wasip1.manifest.json` | Tracked/deployable | Experimental WASI scalar/SIMD variants. Source/provenance archive is committed. |
| Berserk | Upstream tag `14`, commit `8ae895a6151695be4a50d4fb65b0c131659c513a`; network `berserk-9b84c340af7e.nn` | `/berserk/berserk-emscripten.js`, `/berserk/berserk-emscripten.wasm`, `/berserk/berserk-emscripten.data`; manifest `/berserk/berserk-emscripten-single-thread.manifest.json` | Tracked/deployable, experimental | Single-thread Emscripten worker path. Source archive is committed, but distribution policy still calls out network-license/provenance caution. |
| PlentyChess | `Yoshie2000/PlentyChess` commit `58d8ba2505ae2b49f48dd410d214a457d15c12c6`, version `7.0.66`; network `0134-2r24-s0.bin` | `/plentychess/plentychess-emscripten.js`, `/plentychess/plentychess-emscripten.wasm`, `/plentychess/plentychess-emscripten.data`; manifest `/plentychess/plentychess-emscripten-single-thread.manifest.json` | Tracked/deployable, experimental | Single-thread Emscripten worker path with processed NNUE in `.data`. Source archive is committed. |
| Reckless | `codedeliveryservice/Reckless` commit `0010617448bd` + local browser/WASI patches; full v60, lite candidate v53 | `/reckless/reckless.wasm`, `/reckless/reckless-simd128.wasm`, `/reckless/reckless-relaxed-simd128.wasm`, browser-API variants, and optional NNUE sidecars | Ignored/local generated artifacts; README/NOTICE tracked | UI can expose scalar/SIMD/relaxed-SIMD when assets exist. Production must run `npm run reckless:build-production` and publish matching corresponding-source archives. Relaxed SIMD is explicit/experimental and never the default. |

## Machine-readable details

- `public/artifact-index.json` gives a compact deploy/status summary suitable for
  release checks or dashboard rendering.
- `public/models/lc0/manifest.json` records LC0 model URL, byte, and SHA-256
  metadata. Its `mode: "symlink"` means the public path is a staging pointer,
  not proof that the large model blob is committed.
- Engine-specific release manifests with artifact hashes and source-archive
  hashes live beside the deployed engine assets:
  - `/stockfish/stockfish-18.0.7.manifest.json`
  - `/viridithas/viridithas-wasip1.manifest.json`
  - `/berserk/berserk-emscripten-single-thread.manifest.json`
  - `/plentychess/plentychess-emscripten-single-thread.manifest.json`

## Maintenance checklist

When adding or changing a hosted engine/model:

1. Update this file and `public/artifact-index.json` in the same change.
2. Update the engine/model-specific manifest or README.
3. Make deployment status explicit: tracked/deployable vs tracked manifest only
   vs ignored/local.
4. For GPL/AGPL/copyleft engines, publish generated artifacts only with matching
   source archives and manifests per `docs/engine_artifact_distribution.md`.
5. Avoid absolute local paths, usernames, and temporary build paths in public
   manifests.
6. Verify the inventory against git before release:

   ```sh
   git ls-files public/models/lc0 public/stockfish public/viridithas public/berserk public/plentychess public/reckless
   git status --ignored=matching --short public/models/lc0 public/reckless public/runtimes/lc0-tvmjs-webgpu
   ```
