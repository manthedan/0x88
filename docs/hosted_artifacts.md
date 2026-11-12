# Hosted browser engines and model artifacts

Last updated: 2026-11-11

This is the human-readable inventory for browser engine/model artifacts that
the app knows how to serve. The machine-readable record is
`public/artifact-index.json`, which `npm run artifact-index`
(`scripts/write_artifact_index.mjs`) generates from the per-artifact manifests
under `public/`; `tests/artifact_index.test.mjs` fails if the committed index is
stale or names a file that is neither under `public/` nor deliberately
gitignored.

## Status terms

These match the index's `deploymentStatus` values, which are derived from
`.gitignore` and so describe a clean checkout:

- **Tracked/deployable** (`tracked-deployable`): every listed file is committed
  under `public/` (some through Git LFS, and the Stockfish full/pthread pairs as
  symlinks into `node_modules/stockfish` that `npm ci` restores).
- **Tracked manifest, ignored blobs** (`tracked-manifest-ignored-blobs`): the
  manifest (and, for engines, the corresponding-source archives) is committed,
  but the binaries or model blobs are gitignored. Build them, stage them
  (`npm run lc0:prepare-model-assets`, `npm run maia3:stage-assets`), or fetch
  them from the artifact host.
- **Ignored/local** (`ignored-local-artifacts`): nothing listed is committed;
  generated locally for smoke testing. Do not assume a public deployment has it.
- **Research-only**: exposed only behind explicit opt-in or smoke tooling; not a
  stable/default runtime.

Production builds resolve these public paths against the artifact host
(`https://assets.0x88.app`, see `src/lc0/assetUrls.ts` and `netlify.toml`). The
R2 publication history is in `public/releases/`.

## Current inventory

| Family / UI surface | Current version or source anchor | Served URLs / manifest | Deployment status | Notes |
| --- | --- | --- | --- | --- |
| Lc0 Small (default) | `t1-256x10-distilled-swa-2432500` | `/models/lc0/manifest.json`; ONNX batch 1/4/8/16/32 f16, batch 1 and 8 f16 QDQ8, batch 1 f32; lc0web pack `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json` | Tracked manifest, ignored blobs | Analysis, Arena and Play load the batch-1 f16 QDQ8 export by default. The lc0web pack serves the opt-in custom WebGPU path. |
| Lc0 t3-512 distill | `t3-512x15x16h-distill-swa-2767500` | batch 1/4/8/16 f16 entries in `/models/lc0/manifest.json` | Tracked manifest, ignored blobs | The t3 option loads the batch-8 f16 export (`T3_NET` in `src/lc0/bt4Engine.ts`). |
| Lc0 BT4-it332 | `BT4-1024x15x32h-swa-6147500-policytune-332` | `/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx`; b1/b4/b8 f16 entries in `/models/lc0/manifest.json` | Tracked manifest, ignored blobs | Experimental option; loads the batch-8 f16 QDQ8 export (`BT4_NET`). The unquantized f16 exports appear in no committed release manifest. |
| Lc0 legacy BT4 | `BT4-1024x15x32h-swa-6147500` | `/models/lc0/BT4-1024x15x32h-swa-6147500.batch1.f16.onnx` in `/models/lc0/manifest.json` | Tracked manifest, ignored blobs | Historical/research entry; no UI option loads it. |
| Leela Queen Odds | `lqo_v2` | `/models/lc0/lqo_v2.f16.qdq8.onnx`; source export `/models/lc0/lqo_v2.f16.onnx` | Tracked manifest, ignored blobs | Loaded by the Leela Queen Odds option (`LQO_NET`). |
| Centipawn | `bt4_soap_rem_c19000_final` | `/models/bt4_soap_rem_c19000_final.onnx`; metadata `/models/bt4_soap_rem_c19000_final.meta.json`; manifest `/models/centipawn.manifest.json`; provenance `docs/model_provenance/bt4_soap_rem_c19000_final.md` | Tracked/deployable (Git LFS) | ORT model and the fallback for the TVMJS runtime below. BT4/SquareFormer + ThreatGraph TG1 trained from LC0 public data with Stockfish labels; not ChessBench, TCEC, or a generic human-game corpus. |
| Centipawn TVMJS WebGPU | `bt4-soap-rem-c19000-final` f32 `v2-shape-k16` | `/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json` and the files it lists | Tracked/deployable (Git LFS for `.wasm`) | Promoted runtime (`CENTIPAWN_TVMJS_WEBGPU_V2` in `src/nn/runtimeRegistry.ts`). |
| Maia3 | `maia3_simplified` from `CSSLab/maia-platform-frontend` | `/models/maia3/maia3_simplified.qdq8.graph.onnx`, fallbacks `/models/maia3/maia3_simplified.qdq8.onnx` then `/models/maia3/maia3_simplified.onnx`; manifest `/models/maia3/manifest.json`; provenance `docs/model_provenance/maia3.md` | Tracked manifest, ignored blobs | QDQ8 and its graph rewrite (the default since 2026-11-11) are local derivations; see the provenance doc for their recipes and source offer. The QDQ8 file stays published for cached clients. |
| Lc0 TVMJS whole-model WebGPU | Small `t1-256x10-distilled-swa-2432500`; BT4-it332 research artifacts exist locally when staged | `/runtimes/lc0-tvmjs-webgpu/...` | Ignored/local; no tracked runtime artifacts | Research-only. Source/tooling is in the repo, but `public/runtimes/lc0-tvmjs-webgpu/` is gitignored and not part of the stable runtime registry. |
| Stockfish | `stockfish@19.0.1`; upstream `nmrugg/stockfish.js` tag `v19.0.1`, commit `0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6`; networks `nn-1a298aa575a0.nnue` (full), `nn-61e7af4bb97d.nnue` (lite) | `/stockfish/stockfish-19-lite-single.js`, `/stockfish/stockfish-19-lite-single-relaxed.js`, `/stockfish/stockfish-19-lite.js`, `/stockfish/stockfish-19-single.js`, `/stockfish/stockfish-19.js` (each with its `.wasm`); manifest `/stockfish/stockfish-19.0.1.manifest.json` | Tracked/deployable | Lite single-thread is the default Stockfish opponent. GPL corresponding source archive is committed at `/stockfish/stockfish-19.0.1-corresponding-source.tar.gz`. |
| Viridithas | `cosmobobak/viridithas` tag `v20.0.0`, commit `0631113e2215714c54d397cce9a8caebc5e6f8ee`; network `sandhi-s2-b200.nnue.zst` v109 | `/viridithas/viridithas.wasm`, `/viridithas/viridithas-simd128.wasm`, `/viridithas/viridithas-relaxed-simd128.wasm`; manifest `/viridithas/viridithas-wasip1.manifest.json` | Tracked/deployable | Experimental WASI scalar/SIMD/relaxed-SIMD variants. Source/provenance archive is committed. |
| Berserk | Upstream tag `14`, commit `8ae895a6151695be4a50d4fb65b0c131659c513a`; network `berserk-9b84c340af7e.nn` | `/berserk/berserk-emscripten.js`, `.wasm`, `.data`, plus `-simd128` and `-relaxed-simd128` `.js`/`.wasm`; manifest `/berserk/berserk-emscripten-single-thread.manifest.json` | Tracked/deployable, experimental | Single-thread Emscripten builds sharing one preload `.data`. Source archive is committed; the distribution policy records the network-licence reasoning. |
| PlentyChess | `Yoshie2000/PlentyChess` tag `b-v8.0.0`, commit `50ad9f1b523063cbbbf599fe0547f42870fe2906`, version `8.0.0`; network `0178r.bin` | `/plentychess/plentychess-emscripten.js`, `.wasm`, `.data`, plus `-sse41` and `-relaxed-simd128` `.js`/`.wasm`; manifest `/plentychess/plentychess-emscripten-single-thread.manifest.json` | Tracked/deployable, experimental | Single-thread Emscripten builds with the processed NNUE in the shared `.data`. Source archive is committed. |
| Stormphrax | `Ciekce/Stormphrax` `v8.0.0`, commit `582965517ed2032d41a6b4cd6c2e66b1b934e2ad`; network `undertown.nnue` | `/stormphrax/stormphrax-emscripten.js`, `.wasm`, `.data`, plus `-relaxed-simd128` `.js`/`.wasm`; manifest `/stormphrax/stormphrax-emscripten-single-thread.manifest.json` | Tracked/deployable, experimental | Single-thread WebAssembly SIMD and relaxed-SIMD Emscripten builds. Matching GPL source archive is committed. |
| Reckless | `codedeliveryservice/Reckless` commit `0010617448bdef4c8cd7d4f4825b7e42c8bc262a` + local browser/WASI patches; network v60 (`v60-7f587dfb.nnue`), lite candidate v53 built only on request | `/reckless/reckless-relaxed-simd128.wasm`, `/reckless/reckless-simd128.wasm`, `/reckless/reckless.wasm`, browser-API variants, and the external NNUE `/reckless/reckless-v60-7f587dfb.nnue`; manifest `/reckless/reckless-wasip1.manifest.json` | Tracked manifest, ignored blobs; the four corresponding-source archives are committed | The default ladder is relaxed SIMD > fixed SIMD > scalar, with asset fallback at each step. Production must use canonical `npm run reckless:build-release`, include all three default-ladder WASM assets (including relaxed SIMD), generate the release manifest, and publish matching corresponding-source archives. |

## Machine-readable details

- `public/artifact-index.json` gives a compact deploy/status summary suitable for
  release checks or dashboard rendering. Do not edit it by hand; run
  `npm run artifact-index`.
- `public/models/lc0/manifest.json` records Lc0 model URL, byte, and SHA-256
  metadata. Its `mode: "symlink"` means the public path is a staging pointer,
  not proof that the large model blob is committed.
- Engine-specific release manifests with artifact hashes and source-archive
  hashes live beside the deployed engine assets:
  - `/stockfish/stockfish-19.0.1.manifest.json`
  - `/viridithas/viridithas-wasip1.manifest.json`
  - `/berserk/berserk-emscripten-single-thread.manifest.json`
  - `/plentychess/plentychess-emscripten-single-thread.manifest.json`
  - `/stormphrax/stormphrax-emscripten-single-thread.manifest.json`
  - `/reckless/reckless-wasip1.manifest.json`

## Maintenance checklist

When adding or changing a hosted engine/model:

1. Update the engine/model-specific manifest or README.
2. Regenerate the index with `npm run artifact-index` (add a row to
   `scripts/write_artifact_index.mjs` for a new engine or model group), then
   update the table above in the same change.
3. Make deployment status explicit: tracked/deployable vs tracked manifest with
   ignored blobs vs ignored/local.
4. For GPL/AGPL/copyleft engines, publish generated artifacts only with matching
   source archives and manifests per `docs/engine_artifact_distribution.md`.
5. Avoid absolute local paths, usernames, and temporary build paths in public
   manifests.
6. Verify the inventory against git before release:

   ```sh
   npm run artifact-index -- --check
   node --experimental-strip-types --test tests/artifact_index.test.mjs
   git status --ignored=matching --short public/models/lc0 public/models/maia3 public/reckless public/runtimes/lc0-tvmjs-webgpu
   ```
