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
Access-Control-Expose-Headers: CF-Cache-Status, Cache-Status, Age, ETag, Content-Length
```

Do not set cookies on the asset hostname. App requests should be credential-free.

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
- `Content-Length`, `Content-Range`, `ETag`, `Age`, `CF-Cache-Status`, and related cache headers

Range probes should return `206 Partial Content`. A cached range request returning `200` is alert-worthy for large artifacts.
