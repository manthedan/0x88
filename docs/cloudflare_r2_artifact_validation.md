# Cloudflare/R2 artifact cache validation

Large immutable artifacts should be served from `assets.0x88.app` backed by R2. The app shell may stay on Netlify.

## Cache rules

Create explicit Cloudflare cache rules for the asset hostname and these paths:

```text
/artifacts/*
/models/*
/engines/*
/releases/*
/channels/*
```

Recommended rule behavior:

- Cache eligibility: eligible for cache.
- Edge TTL: respect origin/cache-control headers.
- Browser TTL: respect origin.
- Cache key: host + path, no query-string versioning dependency.
- Do not bypass cache for `.onnx`, `.wasm`, or `.json` just because they are not in Cloudflare's default extension list.

Mutable channel pointers should keep browser revalidation and short edge TTLs. Immutable `/artifacts/sha256/*` objects can use one-year CDN/browser TTLs.

## Required asset-origin headers

The R2/Worker/custom-domain response should include:

```text
Access-Control-Allow-Origin: *
Cross-Origin-Resource-Policy: cross-origin
Timing-Allow-Origin: https://0x88.app
Access-Control-Expose-Headers: CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length
```

Do not set cookies on the asset hostname. App requests should be credential-free.

## Optional Worker front door

If direct R2 custom-domain responses cannot provide all required CORS/CORP/timing headers, deploy the checked-in Worker front door:

```sh
npm run deploy:artifact-worker
```

The Worker config in `cloudflare/artifacts.wrangler.toml` binds `browser-chess-models` as `ARTIFACTS` and serves only `/artifacts/sha256/*` keys. It preserves percent-encoded object keys, supports `GET`/`HEAD`/`OPTIONS`, handles bounded byte ranges through R2 range reads, and caches immutable full-body/HEAD metadata responses without caching errors. Cloudflare Workers may normalize cached synthetic `HEAD` responses to `Content-Length: 0`; the Worker also exposes `X-Artifact-Content-Length` so validation can compare range totals against the original artifact byte length.

## Validation command

Use the repository validator against representative artifacts or a release manifest:

```sh
npm run deploy:validate-cdn-artifacts -- \
  --url https://assets.0x88.app/artifacts/sha256/<sha>/model.onnx

npm run deploy:validate-cdn-artifacts -- \
  --release public/releases/<release-id>.json \
  --limit 5
```

The validator checks:

- first HEAD
- second HEAD
- `Range: bytes=0-1023`
- `Accept-Encoding: identity`
- `Accept-Encoding: br`
- `Content-Length`, `X-Artifact-Content-Length`, `Content-Range`, `ETag`, `Age`, `CF-Cache-Status`, and related cache headers

Range probes should return `206 Partial Content`. A cached range request returning `200` is alert-worthy for large artifacts.
