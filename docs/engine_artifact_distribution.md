# Browser engine artifact distribution policy

This is the project policy for distributing generated browser chess-engine artifacts such as JS glue, WASM modules, preload `.data` files, and required NNUE/network files. It is a practical release checklist, not legal advice.

## Scope

This policy applies before publishing or deploying generated artifacts for third-party engines, including:

- Berserk Emscripten artifacts under `/berserk/`
- PlentyChess Emscripten artifacts under `/plentychess/`
- Viridithas WASI artifacts under `/viridithas/`
- Stockfish.js artifacts under `/stockfish/`
- future C/C++/Rust engine browser ports that bundle or require engine/network assets

Local ignored artifacts may be built and smoke-tested for development. Public distribution is gated by this document.

## Default rule

Do not publish generated engine artifacts unless the same release also publishes a matching corresponding-source archive and an artifact manifest.

A normal release must make the source archive available from the same release page or deployment notes as the generated JS/WASM/data files. Do not rely only on an upstream GitHub URL as the corresponding-source offer; preserve the exact source, patches, build scripts, and asset provenance used for this build.

## Required release contents

For each distributed engine/flavor, publish all of the following:

1. **Generated artifacts**
   - JS glue / worker sidecars
   - WASM modules
   - preload `.data` files
   - required NNUE/network/model files, whether embedded or sidecar

2. **Artifact manifest**
   - engine name and flavor
   - upstream repository URL
   - upstream tag/commit
   - local patch file(s)
   - build script path and command
   - Emscripten/compiler version or container image digest
   - output filenames, byte sizes, SHA-256 hashes, and gzip/brotli transfer-size estimates
   - network/model source URL, license/provenance if known, raw hash, processed hash, and processing command if transformed

3. **Corresponding-source archive**
   - exact upstream source snapshot used for the build
   - local patches
   - build scripts and wrapper/adaptor source
   - source for any bundled/required processing tools
   - instructions sufficient to rebuild the distributed artifacts from the archive plus documented external toolchain
   - license files/notices from upstream projects and network/model providers

4. **Release notes**
   - experimental/stable status
   - browser requirements such as COOP/COEP/SAB/pthreads if relevant
   - known disabled features, e.g. tablebases
   - expected artifact sizes and cache behavior

## Network/model assets

Treat required NNUE/network/model assets as release-critical, even when they are stored in a preload `.data` file rather than as a standalone file.

For each asset:

- record source URL and upstream release/tag when available
- record raw SHA-256 before processing
- record processed SHA-256 after conversion/preloading
- document the exact processing command
- preserve the tool source used for processing in the source archive
- include the asset license/provenance note when known

If the asset license or provenance is unclear, do not publish the generated engine bundle until that is resolved.

## Repository hygiene

- Decide per engine whether artifacts are deploy-tracked or CI-generated; do not leave deployed artifacts hidden by `.gitignore` rules.
- For the current Netlify/Vite deployment, committed/LFS public artifacts are intentional for Berserk, PlentyChess, Viridithas, Stockfish.js, and selected model packs.
- Commit source patches, build scripts, smoke scripts, variant metadata, documentation, source archives, and release manifests for deployed third-party binaries, especially copyleft/GPL binaries.
- Keep generated precompressed `.br`/`.gz` sidecars ignored; they are rebuilt into `dist-client/`.

## Browser UI/deployment gate

A local experimental selector may point at ignored dev-server assets for smoke testing. A public deployment must choose one of these options per engine:

1. publish artifacts and matching corresponding-source archive together; or
2. keep the selector disabled/hidden unless the user supplies custom artifact URLs; or
3. omit the generated artifacts from the deployment entirely.

If the selector is visible in a public deployment but assets are intentionally absent, the UI must fail clearly with an asset-missing diagnostic rather than silently falling back to another engine.

## Current engine release cards

### Berserk Emscripten

- License: GPL-family upstream; treat as GPL-gated for distribution.
- Upstream: `https://github.com/jhonnold/berserk.git`
- Pin: tag `14`, commit `8ae895a6151695be4a50d4fb65b0c131659c513a`
- Patch: `patches/berserk-emscripten.patch`
- Build: `npm run berserk:build-emscripten`
- Smoke: `npm run berserk:smoke-emscripten`, `lab/berserk-smoke.html`
- Network: `berserk-9b84c340af7e.nn`
- Network license/provenance: unresolved; no standalone license file found in `jhonnold/berserk-networks` during intake. Do not publicly distribute the network/data bundle until that is resolved or confirmed as covered by the engine release.
- Distribution status: source-archive/manifest tooling exists, but public distribution remains blocked on network license/provenance confirmation.

### PlentyChess Emscripten

- License: GPL-3.0 upstream.
- Upstream: `https://github.com/Yoshie2000/PlentyChess.git`
- Pin: commit `58d8ba2505ae2b49f48dd410d214a457d15c12c6`
- Version: `7.0.66`
- Patch: `patches/plentychess-emscripten.patch`
- Build: `npm run plentychess:build-emscripten`
- Smoke: `npm run plentychess:smoke-emscripten`, `lab/plentychess-smoke.html`
- Network: `0134-2r24-s0.bin`
- Network license/provenance: `Yoshie2000/PlentyNetworks` GPL-3.0.
- Raw network SHA-256: `550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061`
- Processed network SHA-256: `691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4`
- Processing command: upstream `tools/process_net false`, then preload as `/processed.bin`
- Distribution status: cleared mechanically once generated artifacts, source archive, release manifest, and GPL notices are published together; the ~63 MB `.data` sidecar still needs an explicit product/footprint decision before enabling by default.

### Viridithas WASI

- License: MIT upstream at the pinned commit (`LICENSE` and `Cargo.toml`).
- Upstream: `https://github.com/cosmobobak/viridithas.git`
- Pin: commit `20d7402065cae084715183e019fdd18089e2dfac`
- Patch: `patches/viridithas-wasip1.patch`
- Build: `npm run viridithas:build-wasi` and `npm run viridithas:build-simd-wasi`
- Smoke: `npm run viridithas:smoke`
- Network: `atlantis-b800.nnue.zst`
- Network source: `https://github.com/cosmobobak/viridithas-networks/releases/download/v106/atlantis-b800.nnue.zst`
- Raw network SHA-256: `2d387407b926df4dbda441cdc3e2288fee2e6a2afa8e1bd22262309ec0fb668a`
- Distribution status: source/provenance archive and manifest are committed beside the deployed WASI binaries.

### Stockfish.js

- License: GPL-3.0 upstream/package.
- Upstream: `https://github.com/nmrugg/stockfish.js.git`
- Package: `stockfish@18.0.7`
- Pin: npm `gitHead`/upstream commit `32d4b5ae40c01db88219bfbe2b82dbe6dec93832`
- Build: upstream `npm ci && node build.js --all -f`
- Distribution status: release archive and manifest are committed beside the deployed Stockfish.js assets.

## Manifest helper

Use the checked-in helpers to draft manifests and source archives from local generated artifacts:

```sh
npm run berserk:artifact-manifest
npm run plentychess:artifact-manifest
npm run viridithas:artifact-manifest
npm run stockfish:artifact-manifest
npm run berserk:source-archive
npm run plentychess:source-archive
npm run viridithas:source-archive
npm run stockfish:source-archive
```

The artifact-manifest helper writes ignored JSON under `artifacts/engine-manifests/`, including artifact sizes, SHA-256 hashes, and local gzip/brotli transfer-size estimates. The source-archive helper writes `*corresponding-source.tar.gz` files beside the generated engine artifacts under `public/<engine>/`. Deployed GPL engines should commit these release archives and manifests with the deployed binaries.

For a distribution-ready manifest that records the source archive hash and relative deployment URL, run:

```sh
npm run berserk:release-manifest
npm run plentychess:release-manifest
npm run viridithas:release-manifest
npm run stockfish:release-manifest
```

Pass `ENGINE_ARTIFACT_TOOLCHAIN='...'` or `-- --toolchain '...'` when writing a release manifest if the auto-detected/default toolchain text is not exact enough for the generated artifacts. `npm run build:netlify` uses `scripts/precompress_engine_artifacts.mjs` to emit `.br`/`.gz` sidecars in `dist-client/`; see `docs/netlify_engine_artifacts.md`. A manifest generated without a source archive URL and exact toolchain remains a draft, not a public-distribution clearance.

## Manifest template

```json
{
  "engine": "plentychess",
  "flavor": "emscripten-single-thread",
  "status": "experimental",
  "upstream": {
    "repo": "https://github.com/Yoshie2000/PlentyChess.git",
    "commit": "58d8ba2505ae2b49f48dd410d214a457d15c12c6",
    "license": "GPL-3.0"
  },
  "build": {
    "script": "scripts/build_plentychess_emscripten.mjs",
    "command": "npm run plentychess:build-emscripten",
    "patches": ["patches/plentychess-emscripten.patch"],
    "toolchain": "emscripten version or image digest here"
  },
  "artifacts": [
    { "path": "public/plentychess/plentychess-emscripten.js", "bytes": 0, "sha256": "...", "compression": { "gzip": { "bytes": 0 }, "brotli": { "bytes": 0 } } },
    { "path": "public/plentychess/plentychess-emscripten.wasm", "bytes": 0, "sha256": "...", "compression": { "gzip": { "bytes": 0 }, "brotli": { "bytes": 0 } } },
    { "path": "public/plentychess/plentychess-emscripten.data", "bytes": 0, "sha256": "...", "compression": { "gzip": { "bytes": 0 }, "brotli": { "bytes": 0 } } }
  ],
  "totals": { "bytes": 0, "gzipBytes": 0, "brotliBytes": 0 },
  "assets": [
    {
      "name": "0134-2r24-s0.bin",
      "sourceUrl": "https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin",
      "rawSha256": "550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061",
      "processedPath": "/processed.bin",
      "processedSha256": "691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4",
      "processingCommand": "tools/process_net false"
    }
  ],
  "sourceArchive": {
    "url": "release URL here",
    "sha256": "..."
  }
}
```
