# Cloudflare R2 model hosting

Use Netlify for the app shell and Cloudflare R2 for large model artifacts. The
browser app can resolve `/models/...` URLs against an external asset origin via
`VITE_LC0_BROWSER_ASSET_BASE_URL`, `VITE_LC0_MODEL_BASE_URL`, or the temporary
query param `?assetBase=`.

## Bucket layout

Mirror the app's `public/` paths in the R2 bucket:

```text
models/lc0/manifest.json
models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx
models/maia3/manifest.json
models/maia3/maia3_simplified.qdq8.onnx
```

If the public R2 custom domain is `https://models.example.com`, the default LC0
small model URL becomes:

```text
https://models.example.com/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
```

## CORS

Set R2 bucket CORS to allow browser fetches from the hosted app origin. Use the
actual Netlify/custom domain before production; `localhost` entries are for
local smoke only.

```json
[
  {
    "AllowedOrigins": [
      "https://YOUR_NETLIFY_SITE.netlify.app",
      "https://YOUR_CUSTOM_APP_DOMAIN",
      "http://localhost:5173",
      "http://127.0.0.1:5173"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range", "Content-Type"],
    "ExposeHeaders": ["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    "MaxAgeSeconds": 86400
  }
]
```

## Upload with AWS CLI

Create an R2 API token with object read/write access to the bucket, then export
the S3-compatible settings locally:

```sh
export AWS_ACCESS_KEY_ID='R2_ACCESS_KEY_ID'
export AWS_SECRET_ACCESS_KEY='R2_SECRET_ACCESS_KEY'
export R2_ACCOUNT_ID='YOUR_CLOUDFLARE_ACCOUNT_ID'
export R2_BUCKET='YOUR_BUCKET_NAME'
export R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
```

Upload the minimal v0 model set:

```sh
aws s3 cp public/models/lc0/manifest.json "s3://${R2_BUCKET}/models/lc0/manifest.json" --endpoint-url "${R2_ENDPOINT}" --content-type application/json --cache-control 'public, max-age=300'
aws s3 cp public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx "s3://${R2_BUCKET}/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx" --endpoint-url "${R2_ENDPOINT}" --content-type application/octet-stream --cache-control 'public, max-age=31536000, immutable'
aws s3 cp public/models/maia3/manifest.json "s3://${R2_BUCKET}/models/maia3/manifest.json" --endpoint-url "${R2_ENDPOINT}" --content-type application/json --cache-control 'public, max-age=300'
aws s3 cp public/models/maia3/maia3_simplified.qdq8.onnx "s3://${R2_BUCKET}/models/maia3/maia3_simplified.qdq8.onnx" --endpoint-url "${R2_ENDPOINT}" --content-type application/octet-stream --cache-control 'public, max-age=31536000, immutable'
```

Upload optional large LC0 models only when the product should expose them:

```sh
aws s3 cp public/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx "s3://${R2_BUCKET}/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx" --endpoint-url "${R2_ENDPOINT}" --content-type application/octet-stream --cache-control 'public, max-age=31536000, immutable'
aws s3 cp public/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx "s3://${R2_BUCKET}/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx" --endpoint-url "${R2_ENDPOINT}" --content-type application/octet-stream --cache-control 'public, max-age=31536000, immutable'
aws s3 cp public/models/lc0/lqo_v2.f16.onnx "s3://${R2_BUCKET}/models/lc0/lqo_v2.f16.onnx" --endpoint-url "${R2_ENDPOINT}" --content-type application/octet-stream --cache-control 'public, max-age=31536000, immutable'
```

## Netlify configuration

Set one build environment variable:

```text
VITE_LC0_BROWSER_ASSET_BASE_URL=https://models.example.com
```

Use the R2-specific Netlify build command so model blobs are pruned from
`dist-client` before artifact precompression:

```text
npm run build:netlify:r2
```

The normal `build:netlify` command still copies local `public/models` into the
deploy output, which is useful for local all-in-one experiments but is too large
for v0 hosting.

For one-off local tests without rebuilding, append:

```text
?assetBase=https://models.example.com
```

Example:

```text
http://127.0.0.1:5173/lc0-play.html?assetBase=https://models.example.com
```

## Smoke checks

Check headers and CORS from the app origin:

```sh
curl -I -H 'Origin: https://YOUR_NETLIFY_SITE.netlify.app' https://models.example.com/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
curl -I -H 'Origin: https://YOUR_NETLIFY_SITE.netlify.app' https://models.example.com/models/maia3/maia3_simplified.qdq8.onnx
```

Expected headers include `access-control-allow-origin`, `content-length`, and a
long `cache-control` for immutable ONNX files.

Then open the app with:

```text
/lc0-play.html?assetBase=https://models.example.com
/lc0-analysis.html?assetBase=https://models.example.com
/lc0-arena.html?assetBase=https://models.example.com
```

Use the same URL as a Netlify preview smoke before promoting the variable to the
production environment.
