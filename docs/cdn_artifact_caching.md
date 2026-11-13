# CDN artifact caching architecture

Last updated: 2026-11-12 (source-checked; no new live CDN measurements).

## Overview

Browser chess engines (WASM binaries, NNUE networks, ONNX models) range from
small glue/WASM files to hundreds of MB for large model exports. Serving them efficiently from a CDN, with correct cache
semantics and compression, is critical for the analysis and arena pages to
function. This document describes the two-plane serving architecture, the
Cloudflare Worker that mediates R2 access, known failure modes, and the
operational playbook for diagnosing artifact serving problems.

## Architecture

```
Browser (0x88.app)
  |
  |  COOP/COEP cross-origin isolated
  |
  v
assets.0x88.app  (Cloudflare R2 + Worker)
  |
  +-- Control plane (mutable, short TTL)
  |     /channels/stable.json     -> release manifest URL
  |     /releases/<date>.<sha>.json -> artifact URL list with hashes
  |
  +-- Data plane (immutable, 1-year cache)
  |     /artifacts/sha256/<sha>/*  -> content-addressed blobs
  |     /viridithas/*.wasm          -> logical alias (short TTL, resolves to artifact)
  |     /berserk/*.wasm             -> logical alias
  |     /reckless/*.wasm            -> logical alias
  |     /stockfish/*.wasm           -> logical alias
  |     /models/lc0/*.onnx          -> logical alias
  |
  +-- App shell (Netlify, not R2)
        index.html                  -> must-revalidate
        /_app/immutable/*           -> 1-year immutable (Vite content-hashed)
```

### Two-plane model

**Control plane:** The channel manifest (`/channels/stable.json`) points to an
immutable release manifest. The release manifest lists every artifact with its
content-addressed URL, byte size, and SHA-256. Channel pointers are mutable
with a short TTL; everything they point to is immutable.

**Data plane:** Content-addressed blobs under `/artifacts/sha256/<full-sha>/`
are write-once and cached for 1 year. Logical aliases like
`/viridithas/viridithas-relaxed-simd128.wasm` resolve through the Worker by
reading the channel manifest, finding the matching release artifact, and
serving the content-addressed blob. Logical aliases have a 5-minute TTL with
stale-while-revalidate to allow rapid channel updates.

### Cross-origin isolation

The app sets `COOP: same-origin` and `COEP: require-corp` for SharedArrayBuffer
support (needed by WASI persistent workers). R2 responses include
`Cross-Origin-Resource-Policy: cross-origin` and `Access-Control-Allow-Origin: *`
so artifacts load under COEP without CORB blocking.

## Cloudflare Worker (artifact-assets-worker.mjs)

The Worker sits in front of R2 and handles:

1. **Logical path resolution:** Maps friendly URLs (`/viridithas/viridithas-relaxed-simd128.wasm`)
   to content-addressed keys by reading channel + release manifests from R2.

2. **Edge caching:** Caches immutable artifact responses in Cloudflare's edge
   cache (`caches.default`). Content-addressed responses have a one-year policy;
   logical responses receive a short client TTL while sharing canonical edge bodies. HEAD metadata and full-body responses are
   cached under separate keys to avoid slicing large blobs in Worker memory.

3. **Range requests:** Delegates to R2's native range reader for `Range: bytes=`
   requests. Range responses are never served from the Worker edge cache, to
   avoid materializing hundreds of MB in Worker memory.

4. **Compression prevention:** Sets `no-transform` on `.wasm`, `.onnx`, and
   `.data` responses to prevent Cloudflare from attempting auto-compression on
   binary content that may already be compressed or could be corrupted by
   transform.

5. **CORS/CORP headers:** Attaches `Access-Control-Allow-Origin`,
   `Cross-Origin-Resource-Policy`, and `Timing-Allow-Origin` to every response.

### What the Worker does NOT do

- It does not compress responses. Brotli/gzip pre-compression is done at publish
  time (see below).
- It validates representation descriptors/metadata, not full-body SHA-256 on
  every request. Body integrity is checked by the publisher/canary and browser
  model cache.
- It does not cache logical-alias responses in the edge cache (only
  content-addressed keys are edge-cached).

## Compression pipeline

### Publish-time pre-compression

`scripts/precompress_engine_artifacts.mjs` generates `.br` and `.gz` sidecars
for local/all-in-one static builds. Brotli quality is 11 through 64 MiB and 5
above it. Directory and extension coverage comes from
`scripts/engine_artifact_registry.mjs`.

The current content-addressed release generator emits identity/Brotli
representations and encoded/decoded metadata. Publish them with
`scripts/publish_hashed_artifacts_to_r2.mjs`; the older
`scripts/r2_brotli_publish_assets.mjs` logical-path helper is not the
release/channel-last publication path.

### CDN behavior

For v2 artifacts the Worker selects the recorded identity or Brotli
representation from `Accept-Encoding`, with `Vary: Accept-Encoding`. Range
requests select identity. Canonical physical representation URLs are distinct
cache keys. `no-transform` prevents a second binary transformation; the Worker
does not recompress the body.

### Historical zero-length HEAD diagnosis: not proof of corruption

The June incident notes reported `Content-Length: 0` for a negotiated
`viridithas-relaxed-simd128.wasm` request (then described as 57 MB), versus a
full identity response. That header observation alone does not establish a
zero-byte body or poisoned cache: Workers may normalize cached synthetic HEAD
responses to length zero.

An engine timeout can still come from missing/truncated bodies, failed compile,
startup or search. Compare the release's decoded `X-Artifact-Content-Length`
and SHA-256 with representation-specific `X-Artifact-Encoded-Length` and hash.
Check an identity Range and, when needed, a tightly bounded full-body integrity
canary. Only diagnose/purge a bad entry after proving a representation/body
mismatch. Query-string cache busting is not a repair for canonical physical
keys. See [`cloudflare_r2_artifact_validation.md`](./cloudflare_r2_artifact_validation.md).

## Browser-side caching

### Cache Storage (model cache)

`src/lc0/modelCache.ts` uses the Cache Storage API to store model artifacts.
New bytes are length/SHA-256 checked before caching. Ordinary warm hits use
trusted IndexedDB verification metadata plus byte length, expected hash,
schema/runtime version and age; missing/stale metadata triggers full rehash.
A mismatch deletes bytes and metadata and re-fetches with `cache: "reload"`
(not `force-cache`). Warm inference still reads the model bytes.

### WASM module cache (worker-side)

The WASI worker (`recklessWasiWorker.ts`) caches compiled `WebAssembly.Module`
objects in a `Map<string, Promise<WebAssembly.Module>>` keyed by URL. This
avoids re-downloading and re-compiling the same WASM on repeated engine
invocations within a page session.

### Service worker

The service worker handles the app shell only. It does not cache engine or
model artifacts. This avoids double-caching with Cache Storage and keeps the
service worker cache small.

## Artifact size reference

| Artifact | Raw size | Notes |
|---|---|---|
| Stockfish 19.0.1 lite single WASM | 1,845,971 bytes baseline | Relaxed lite 1,845,006 bytes |
| Viridithas v20 WASM | ~53.2 MiB each | Scalar, SIMD, relaxed; compressed NNUE embedded |
| Reckless embedded WASM | ~62 MiB each | External prototypes split code + 63,266,880-byte NNUE |
| Berserk WASM + data | ~0.13–0.14 MB + ~25 MB | Three code tiers, one shared NNUE preload |
| PlentyChess v8 WASM + data | ~0.4 MB + ~53 MB | Shared processed NNUE; see upgrade manifest |
| LC0 Small b1 QDQ8 ONNX | 20,558,031 bytes | Apple split-K sibling 20,580,952 bytes |
| LC0 BT4 b8 QDQ8 ONNX | 186,676,343 bytes | Current large-net selection; old f16 export ~370 MB |
| Maia3 graph QDQ8 ONNX | 27,971,924 bytes | Current human model; original f32 fallback 45,683,686 bytes |

These are raw artifact sizes, not transfer size or resident memory. Exact pins
and compression metadata live in per-engine/model manifests.

## Operational playbook

### Diagnosing engine timeouts

1. Check if the WASM URL returns a valid response:
   ```sh
   curl -sI -H 'Accept-Encoding: br,gzip' https://assets.0x88.app/viridithas/viridithas-relaxed-simd128.wasm
   ```
   A zero synthetic HEAD length is inconclusive. Compare decoded/encoded
   artifact headers with the pinned release, then use the canary below.

2. Compare with identity encoding:
   ```sh
   curl -sI -H 'Accept-Encoding: identity' https://assets.0x88.app/viridithas/viridithas-relaxed-simd128.wasm
   ```

3. Check all engine WASM files:
   ```sh
   for wasm in reckless/reckless-relaxed-simd128.wasm berserk/berserk-emscripten-relaxed-simd128.wasm \
               plentychess/plentychess-emscripten-relaxed-simd128.wasm viridithas/viridithas-relaxed-simd128.wasm; do
     echo "=== $wasm ==="
     curl -sI -H 'Accept-Encoding: br,gzip' "https://assets.0x88.app/$wasm" | grep -i 'content-length\|content-encoding'
   done
   ```

4. Use `scripts/validate_artifact_cdn_headers.mjs --release <local-release.json>`
   for immutable release metadata and a small identity Range check. Add
   `--verify-bodies --limit 1` only when a full decoded integrity check is needed.
   See `docs/cloudflare_r2_artifact_validation.md` for complete commands. If a
   body/representation mismatch is proven, purge the affected canonical physical
   URL(s), not just a logical alias. Example API shape (requires authorization):
   ```sh
   curl -X POST "https://api.cloudflare.com/client/v4/zones/<zone>/purge_cache" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     --data '{"files":["https://assets.0x88.app/artifacts/sha256/<decoded-sha>/identity"]}'
   ```

### Publishing new artifacts

1. Build and validate the affected artifact/source/manifest set locally or in CI.
2. Run `scripts/write_artifact_release_manifests.mjs` to generate the release
   and content-addressed bodies under ignored
   `.local-dev-artifacts/artifact-releases/`. Use `--out-dir` only for an
   explicitly managed alternative staging root.
3. Inspect `scripts/publish_hashed_artifacts_to_r2.mjs` without `--execute`.
   After approval, publish immutable blobs and the release first, without a
   channel mutation. Missing/corrupt bodies must block publication.
4. Run the non-mutating release canary in
   `docs/cloudflare_r2_artifact_validation.md`. It checks the exact hosted
   release and its physical representations, not just the current stable alias.
5. After approval and a passing canary, use the same publisher with the matching
   `--channel-manifest`. It validates the release/channel relationship and
   updates the channel last. Purge only the channel if immediate propagation
   is needed. Do not overwrite immutable bodies.

`scripts/netlify_r2_release.mjs` verifies/builds/deploys the pruned Netlify app
shell. Its `--channel-url` bakes a URL into that shell; it does **not** publish
or update the R2 channel.

### Adding a new engine family

1. Add the WASM/JS/data files to `public/<engine>/`.
2. Add inventory/preparation/external-hosting metadata to
   `scripts/engine_artifact_registry.mjs`, rather than parallel directory lists.
3. Include the new logical URLs in the generated release map. Review Worker
   prefix support only if the new path is outside its supported families.
4. Update `public/artifact-index.json` and `docs/hosted_artifacts.md`.
5. Verify headers and Range support after the first deploy.
