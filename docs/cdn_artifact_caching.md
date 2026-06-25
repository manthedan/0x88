# CDN artifact caching architecture

Last updated: 2026-06-25

## Overview

Browser chess engines (WASM binaries, NNUE networks, ONNX models) range from
130 KB to 60+ MB. Serving them efficiently from a CDN, with correct cache
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
   cache (`caches.default`). Content-addressed keys are cached indefinitely;
   logical aliases use a short TTL. HEAD metadata and full-body responses are
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
- It does not validate artifact integrity. SHA-256 validation happens in the
  browser via `modelCache.ts`.
- It does not cache logical-alias responses in the edge cache (only
  content-addressed keys are edge-cached).

## Compression pipeline

### Publish-time pre-compression

`scripts/precompress_engine_artifacts.mjs` generates `.br` and `.gz` sidecars
during the Netlify build. Brotli quality is 11 for files under 64 MB and 5 for
larger files (to avoid timeout on the 57 MB Viridithas and 64 MB Reckless WASMs).

`scripts/r2_brotli_publish_assets.mjs` uploads artifacts to R2 with correct
`Content-Type`, `Content-Encoding`, and `Cache-Control` metadata.

### CDN behavior

Cloudflare auto-negotiates compression based on `Accept-Encoding`. The Worker
sets `no-transform` on binary types to prevent the CDN from re-compressing WASM.
For text assets (JS glue, JSON manifests), the CDN handles brotli negotiation
automatically.

### Known issue: cache poisoning on large WASM files

If the origin (R2 Worker or publish pipeline) returns a 0-byte or truncated
response even once, Cloudflare caches that bad response. This has been observed
on `viridithas-relaxed-simd128.wasm` (57 MB), where `curl -H 'Accept-Encoding:
br,gzip'` returns `Content-Length: 0` while `curl -H 'Accept-Encoding: identity'`
returns the full file.

Symptoms: engine "searching" indefinitely with no error, eventually timing out.
The WASM worker fetches the URL, receives 0 bytes, and either hangs or fails
silently inside `WebAssembly.compile()`.

Mitigation:
- Purge the specific Cloudflare cache entry for the affected URL.
- Use cache-busting query strings or content-addressed URLs on redeploy.
- Monitor `Content-Length` vs `X-Artifact-Content-Length` header mismatches.

## Browser-side caching

### Cache Storage (model cache)

`src/lc0/modelCache.ts` uses the Cache Storage API to store model artifacts.
On cache hit, it validates byte length and SHA-256 hash. A mismatch deletes the
entry and re-fetches with `cache: "reload"` (not `force-cache`).

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
| Stockfish lite single WASM | ~7 MB | Default SF opponent |
| Viridithas WASM (all variants) | ~57 MB | Scalar, SIMD, relaxed SIMD |
| Reckless WASM (all variants) | ~64 MB | Scalar, SIMD, relaxed SIMD |
| Berserk WASM + data | ~10 MB | Emscripten with NNUE |
| PlentyChess WASM + data | ~60 MB | Emscripten with NNUE |
| LC0 small ONNX | ~21 MB | Default LC0 model |
| LC0 BT4 ONNX | ~370 MB | Research/arena only |
| Maia3 ONNX | ~50 MB | Human move model |

## Operational playbook

### Diagnosing engine timeouts

1. Check if the WASM URL returns a valid response:
   ```sh
   curl -sI -H 'Accept-Encoding: br,gzip' https://assets.0x88.app/viridithas/viridithas-relaxed-simd128.wasm
   ```
   If `Content-Length: 0` or the response is truncated, the CDN has a poisoned
   cache entry.

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

4. If a cache entry is poisoned, purge it via the Cloudflare dashboard or API:
   ```sh
   curl -X POST "https://api.cloudflare.com/client/v4/zones/<zone>/purge_cache" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     --data '{"files":["https://assets.0x88.app/viridithas/viridithas-relaxed-simd128.wasm"]}'
   ```

### Publishing new artifacts

1. Build artifacts locally or in CI.
2. Run `scripts/r2_brotli_publish_assets.mjs` to upload with correct metadata.
3. Run `scripts/write_artifact_release_manifests.mjs` to generate release manifests.
4. Run `scripts/netlify_r2_release.mjs` to update the channel pointer.
5. Purge the channel URL for immediate propagation.

### Adding a new engine family

1. Add the WASM/JS/data files to `public/<engine>/`.
2. Add entries to `scripts/r2_brotli_publish_assets.mjs` targets list.
3. Add logical-alias routing in the Cloudflare Worker if using content-addressed
   serving.
4. Update `public/artifact-index.json` and `docs/hosted_artifacts.md`.
5. Verify headers and Range support after the first deploy.
