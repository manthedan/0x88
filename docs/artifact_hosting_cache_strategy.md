# Artifact hosting and cache strategy

Last updated: 2026-06-22

## Recommendation

Use a two-plane release model:

- **Control plane:** tiny mutable channel manifests, for example `/channels/stable.json`.
- **Data plane:** immutable release manifests and content-addressed model/engine files.

A production URL layout should look like:

```text
/channels/stable.json
/releases/2026-06-22.<release-sha>.json
/artifacts/sha256/<full-sha256>/lc0-small.onnx
/artifacts/sha256/<full-sha256>/stockfish.wasm
```

The channel manifest points to an immutable release manifest. The release manifest points to immutable binaries. Deploy blobs first, publish the release manifest second, and update the channel pointer last. Rollback should only repoint the channel manifest; it should not purge or overwrite binaries.

## Current assessment

The repository already has a useful foundation:

- Machine-readable artifact manifests with byte counts and SHA-256 values.
- Separate documentation for deployable, externally hosted, and local/generated artifacts.
- Model-byte validation before writing to Cache Storage.
- A configurable asset base for R2-backed hosting.

The main production risks are around caching semantics and mutable heavy URLs:

1. Stable public URLs must not use one-year `immutable` caching unless they are truly write-once.
2. Forced `.br` rewrites from stable URLs are brittle and should be replaced by CDN/object metadata or Worker-controlled negotiation.
3. Cloudflare should have explicit cache rules for model/engine/release/channel paths.
4. Cache recovery after a hash mismatch must not refetch with `cache: "force-cache"`.
5. Warm-start and cache-size inspection should avoid re-reading and rehashing hundreds of MB on ordinary UI paths.

## Required invariants

- Never overwrite an existing content-hashed key.
- Never publish a channel manifest that points to missing artifacts.
- Never purge immutable blobs as part of a normal release.
- Treat a mismatch at a content-addressed URL as corruption or an origin/CDN invariant violation.
- Keep ORT WebGPU as the stable default runtime; hosting changes must not promote TVMJS or other research paths.
- Keep generated artifacts out of git unless the release policy explicitly changes.

## Headers

For immutable, content-addressed binaries and immutable release manifests:

```toml
[[headers]]
for = "/artifacts/*"
  [headers.values]
  Cache-Control = "public, max-age=31536000, immutable"
  Netlify-CDN-Cache-Control = "public, max-age=31536000"
  CDN-Cache-Control = "public, max-age=31536000"
  Cloudflare-CDN-Cache-Control = "public, max-age=31536000"

[[headers]]
for = "/releases/*"
  [headers.values]
  Cache-Control = "public, max-age=31536000, immutable"
  Netlify-CDN-Cache-Control = "public, max-age=31536000"
  CDN-Cache-Control = "public, max-age=31536000"
  Cloudflare-CDN-Cache-Control = "public, max-age=31536000"
```

For mutable channel pointers:

```toml
[[headers]]
for = "/channels/*"
  [headers.values]
  Cache-Control = "public, max-age=0, no-cache"
  Netlify-CDN-Cache-Control = "public, max-age=60, stale-if-error=86400"
  CDN-Cache-Control = "public, max-age=60, stale-if-error=86400"
  Cloudflare-CDN-Cache-Control = "public, max-age=60, stale-if-error=86400"
```

For HTML and any service-worker script:

```text
Cache-Control: public, max-age=0, must-revalidate
```

Vite-generated hashed JavaScript and CSS can keep a one-year immutable policy.

## Hosting split

Use Netlify for:

- HTML.
- Vite bundles.
- Channel manifests.
- Possibly release manifests.

Use `assets.0x88.app` backed by Cloudflare R2 for:

- ONNX models.
- WASM engines.
- Engine data files.
- Model/runtime shards.

A cross-origin asset host should return:

```text
Access-Control-Allow-Origin: *
Cross-Origin-Resource-Policy: cross-origin
Timing-Allow-Origin: https://0x88.app
Access-Control-Expose-Headers: CF-Cache-Status, Cache-Status, Age, ETag, Content-Length
```

Asset requests should be credential-free. Do not set cookies on the asset hostname.

## Cloudflare cache rules

Add explicit cache eligibility rules for:

```text
/artifacts/*
/models/*
/engines/*
/releases/*
/channels/*
```

Suggested behavior:

- Cache eligibility: eligible for cache.
- Edge TTL: respect origin/CDN cache-control headers.
- Browser TTL: respect origin.
- Cache key: standard host and pathname.
- Do not rely on version query strings.

Monitor object-size limits. Current large models around the 370 MB class fit common Cloudflare cache limits, but future larger models may require sharding.

## Service worker and Range policy

Use the service worker for the app shell, not as a second independent owner of large model files.

- Never precache models.
- Bypass service-worker handling for `/artifacts/` unless the service worker is the single authoritative model cache implementation.
- Do not cache mutable channel pointers with a cache-first strategy.
- Delete old app-shell cache versions during activation.
- Keep model Cache Storage cleanup manifest-driven.
- Pass Range requests through to the network/CDN unless explicitly constructing valid `206 Partial Content` responses.
- Never store a `206` response under the same Cache Storage key as the complete asset.

For resumable downloads, prefer fixed content-hashed shards, for example 16–32 MB each, over arbitrary ranges. Each shard can be independently cached and verified.

## Browser cache implementation changes

### Correctness fix

`src/lc0/modelCache.ts` should not recover from a stale/corrupt Cache Storage entry by refetching with `cache: "force-cache"`.

After a byte-length or SHA-256 mismatch:

```ts
await cache.delete(request);

const response = await fetch(asset.url, {
  cache: "reload",
  signal,
});

// Validate once. If it still fails, report and stop.
```

Content-hashed URLs should make routine stale recovery unnecessary. A repeated mismatch at a hashed URL should fail loudly.

### Warm-start optimization

Current warm loads can read the full cached model and recompute SHA-256. That makes startup CPU work proportional to model size.

Next revision:

- Validate SHA-256 when inserting an asset.
- Store verification metadata keyed by content hash in IndexedDB.
- On ordinary warm loads, check expected byte length and trusted metadata.
- Rehash periodically, after runtime failure, or in diagnostic mode.
- During streamed download, preallocate one `Uint8Array(expectedBytes)` when the expected length is known instead of retaining chunks and concatenating them.

### Storage UI optimization

The UI should not clone every cached response and materialize each body as a blob just to estimate cache size. Record manifest byte sizes and cache metadata in IndexedDB, then render settings/storage UI from that metadata.

## Release sequence

1. Generate hashes and the immutable release manifest in CI.
2. Upload every new artifact under its content-hashed key.
3. Verify size, content type, hash, and a small range request through Cloudflare.
4. Publish the immutable release manifest.
5. Update `/channels/stable.json`.
6. Purge only the channel pointer when immediate propagation matters.
7. Retain old release manifests and blobs for at least one app-release cycle, preferably 30–90 days.
8. Remove obsolete browser cache entries lazily after the new release has loaded successfully.

Rollback should only repoint the channel manifest.

## Observability

Emit one client event per large asset load with:

```text
release_id
asset logical name
short content hash
cache source: memory | Cache Storage | HTTP | network
expected bytes
received bytes
download_ms
hash_ms
cache_read_ms
engine_init_ms
integrity result
service_worker_controlled
storage_persistent
failure category
```

Use Resource Timing as supporting evidence, including transfer size, encoded/decoded sizes, protocol, and service-worker timing when available.

At the CDN layer, monitor:

- Byte-weighted hit ratio for `/artifacts/`.
- `CF-Cache-Status` and `Age`.
- Netlify `Cache-Status`.
- Origin bytes per model hash.
- MISS, BYPASS, EXPIRED, and REVALIDATED rates.
- Range response status and returned byte count.

Alert on:

- Any SHA mismatch.
- The same content-hashed URL returning different ETags or lengths.
- Warm-load network transfer for a recently used model.
- Sudden BYPASS/DYNAMIC growth for model paths.
- Channel adoption taking materially longer than configured TTL.
- Repeated storage eviction.
- Cached range requests returning `200` instead of `206`.

## Production validation checklist

For each representative large asset, exercise headers twice and test ranges/encoding explicitly:

```sh
curl -sSI "$ASSET_URL"
curl -sSI "$ASSET_URL"

curl -sS \
  -H 'Range: bytes=0-1023' \
  -D - -o /dev/null \
  "$ASSET_URL"

curl -sSI -H 'Accept-Encoding: identity' "$WASM_URL"
curl -sSI -H 'Accept-Encoding: br'       "$WASM_URL"
```

Validate at least:

- `Cache-Control` matches path mutability.
- CDN cache headers are present and sane.
- `Content-Length` is available for large immutable assets.
- Range request returns `206` with the expected byte count.
- Brotli/gzip behavior is negotiated, not forced unconditionally.
- Repeated request shows expected CDN cache state.

## Immediate implementation order

1. Hash public model/engine URLs or route them through release manifests.
2. Remove forced Brotli rewrites from the Netlify path.
3. Add Cloudflare cache rules for asset, model, engine, release, and channel paths.
4. Fix `modelCache.ts` mismatch recovery to use `cache: "reload"`.
5. Add IndexedDB verification metadata and avoid full warm-load rehashes by default.
6. Add lightweight asset-load telemetry.
7. Introduce sharded content-addressed model layout for the largest networks.
