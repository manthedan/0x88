# Browser engine artifact distribution policy

This is the project policy for distributing generated browser chess-engine artifacts such as JS glue, WASM modules, preload `.data` files, and required NNUE/network files. It is a practical release checklist, not legal advice.

## Scope

This policy applies before publishing or deploying generated artifacts for third-party engines, including:

- Berserk Emscripten artifacts under `/berserk/`
- PlentyChess Emscripten artifacts under `/plentychess/`
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
   - output filenames, byte sizes, and SHA-256 hashes
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

- Keep generated engine artifacts ignored in git.
- Commit source patches, build scripts, smoke scripts, variant metadata, and documentation.
- Commit small release manifests only after they refer to an actual published source archive.
- Do not add generated JS/WASM/data/NNUE files to this repository unless a separate decision explicitly changes repo policy.

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
- Smoke: `npm run berserk:smoke-emscripten`, `berserk-smoke.html`
- Network: `berserk-9b84c340af7e.nn`
- Distribution status: not cleared until a matching source archive and manifest are published.

### PlentyChess Emscripten

- License: GPL-3.0 upstream.
- Upstream: `https://github.com/Yoshie2000/PlentyChess.git`
- Pin: commit `58d8ba2505ae2b49f48dd410d214a457d15c12c6`
- Version: `7.0.66`
- Patch: `patches/plentychess-emscripten.patch`
- Build: `npm run plentychess:build-emscripten`
- Smoke: `npm run plentychess:smoke-emscripten`, `plentychess-smoke.html`
- Network: `0134-2r24-s0.bin`
- Raw network SHA-256: `550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061`
- Processed network SHA-256: `691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4`
- Processing command: upstream `tools/process_net false`, then preload as `/processed.bin`
- Distribution status: not cleared until a matching source archive and manifest are published; the ~63 MB `.data` sidecar also needs an explicit product/footprint decision.

## Manifest helper

Use the checked-in helper to draft a manifest from local generated artifacts:

```sh
npm run berserk:artifact-manifest
npm run plentychess:artifact-manifest
```

The helper writes ignored JSON under `artifacts/engine-manifests/`, including artifact sizes and SHA-256 hashes. A generated manifest is still not release-complete until `sourceArchive.url` and `sourceArchive.sha256` are filled with a published matching source archive.

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
    { "path": "public/plentychess/plentychess-emscripten.js", "bytes": 0, "sha256": "..." },
    { "path": "public/plentychess/plentychess-emscripten.wasm", "bytes": 0, "sha256": "..." },
    { "path": "public/plentychess/plentychess-emscripten.data", "bytes": 0, "sha256": "..." }
  ],
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
