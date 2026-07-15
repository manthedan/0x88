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

The Worker config in `cloudflare/artifacts.wrangler.toml` binds `browser-chess-models` as `ARTIFACTS` and serves `/artifacts/sha256/*`, `/releases/*.json`, and `/channels/*.json` keys. It preserves percent-encoded object keys, supports `GET`/`HEAD`/`OPTIONS`, handles bounded byte ranges through R2 range reads, and caches immutable artifact full-body/HEAD metadata responses without caching errors. Release manifests are immutable, while mutable channel manifests use revalidation-oriented headers. V2 logical aliases negotiate SHA-only identity and Brotli representation keys, and Range requests always select identity. Cloudflare Workers may normalize cached synthetic `HEAD` responses to `Content-Length: 0`; the Worker also exposes `X-Artifact-Content-Length` so validation can compare range totals against the original artifact byte length.

## Non-mutating live CDN canary

Run this only against a release manifest that is already published. It performs
read-only `HEAD` and small Range requests through the live CDN. It does not
upload objects, deploy the Worker, purge cache, or change a channel:

```sh
export RELEASE_MANIFEST="public/releases/<existing-release-id>.json"

node scripts/validate_artifact_cdn_headers.mjs \
  --release "$RELEASE_MANIFEST" \
  --artifact-base https://assets.0x88.app \
  --limit 5 \
  --range 1024 \
  --json
```

For an already-published logical alias, the exact single-asset canary is:

```sh
node scripts/validate_artifact_cdn_headers.mjs \
  --url https://assets.0x88.app/models/lc0/<existing-model>.onnx \
  --range 1024 \
  --json
```

The v2 release form verifies:

- `Accept-Encoding: identity` selects the SHA-only identity object.
- `Accept-Encoding: br` selects the recorded Brotli representation.
- decoded and encoded SHA-256/length headers match the release manifest.
- a Range request sent with Brotli accepted still returns identity bytes and
  valid `206 Partial Content`.

## Local release and publisher dry-run

Generate a release into a disposable directory, then inspect the publisher plan
without `--execute`:

```sh
export RELEASE_ID="canary-$(date -u +%Y%m%dT%H%M%SZ)"
export RELEASE_ROOT="$(mktemp -d)"

node scripts/write_artifact_release_manifests.mjs \
  --root . \
  --out-dir "$RELEASE_ROOT" \
  --release-id "$RELEASE_ID" \
  --channel canary \
  --generated-at "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"

node scripts/publish_hashed_artifacts_to_r2.mjs \
  --root . \
  --release "$RELEASE_ROOT/releases/$RELEASE_ID.json" \
  --channel-manifest "$RELEASE_ROOT/channels/canary.json" \
  --bucket browser-chess-models
```

The second command is a local plan only. It must report `"execute": false`.
Stop here. Do not add `--execute`, do not run `wrangler deploy`, do not upload
objects, do not purge, and do not repoint `stable` as part of this validation.
The generated release id and channel name are path-safe identifiers, and the
plan must list the release manifest before a channel item whose
`"uploadAction"` is `"update-last"`.

To inspect cleanup safety separately, use the read-only planner and stop before
any delete flags:

```sh
CLOUDFLARE_ACCOUNT_ID="<read-only-account-id>" \
CLOUDFLARE_API_TOKEN="<read-only-token>" \
node scripts/plan_r2_artifact_cleanup.mjs \
  --bucket browser-chess-models \
  --retention-days 90 \
  --json
```

Do not add `--execute`, `--delete-category`, or `--allow-delete-hashed`.

The validator checks:

- first HEAD
- second HEAD
- `Range: bytes=0-1023`
- `Accept-Encoding: identity`
- `Accept-Encoding: br`
- `Content-Length`, `X-Artifact-Content-Length`, `Content-Range`, `ETag`, `Age`, `CF-Cache-Status`, and related cache headers

Range probes should return `206 Partial Content`. A cached range request returning `200` is alert-worthy for large artifacts.
