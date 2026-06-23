# LC0 Pack Serving Compression Audit

## Scope

Current LC0 lc0web pack:

- `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json`
- `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.000.bin`
- `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.001.bin`
- `/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/weights.002.bin`

This is Pack Footprint / transfer-size work only. It does not change Runtime Configuration, explicit GPU-buffer Execution Footprint, evaluator cacheFootprint, or stable defaults.

## Header check

Vite dev serving does not compress these assets by default, even when the request advertises `Accept-Encoding: br,gzip`:

- `/tmp/lc0_pack_header_model_json_vite_20260608.txt`
- `/tmp/lc0_pack_header_weights_bin_vite_20260608.txt`

Observed local headers included raw `Content-Length` and no `Content-Encoding`:

- `model.lc0web.json`: `Content-Length: 296585`, `Content-Type: application/json`
- `weights.000.bin`: `Content-Length: 16565504`, `Content-Type: application/octet-stream`

The repo's deploy path is Netlify (`netlify.toml`). Stable engine/model URLs are no longer force-rewritten to `.br` sidecars. LC0 pack serving should not copy that old pattern: if a host cannot reliably attach `Content-Encoding: br` to rewritten sidecars, browsers can receive raw Brotli bytes and fail JSON/shard loading.

## Compression estimates

Artifact: `/tmp/lc0_pack_serving_compression_audit_20260608.json`.

| File | Raw bytes | gzip bytes | brotli bytes | brotli ratio |
| --- | ---: | ---: | ---: | ---: |
| `model.lc0web.json` | 296,585 | 26,397 | 17,996 | 0.0607 |
| `weights.000.bin` | 16,565,504 | 15,292,856 | 14,542,191 | 0.8779 |
| `weights.001.bin` | 15,766,736 | 14,563,552 | 13,853,255 | 0.8786 |
| `weights.002.bin` | 8,086,210 | 7,007,779 | 6,642,114 | 0.8214 |
| Total | 40,715,035 | 36,890,584 | 35,055,556 | 0.8610 |

The JSON manifest compresses very well. F16 binary shards are already entropy-dense, but brotli still saves about 5.66 MB across this pack.

## Recommendation

No runtime compression change was made in this audit. Follow-up deployment policy removed the old Netlify forced sidecar rewrite pattern for stable URLs.

Recommended serving policy:

1. Prefer host/CDN automatic compression for the original `.json` and `.bin` URLs when the platform supports it.
2. If using precompressed sidecars for LC0 packs, serve them only through a mechanism that is proven to set `Content-Encoding` correctly for the response, such as CDN object metadata or an edge/function handler.
3. Keep LC0 pack compression evidence under Pack Footprint only; do not mix it with runtime Execution Footprint or evaluator cacheFootprint.
4. For future larger LC0 packs, generate a deployment manifest that records raw/gzip/brotli sizes and the serving mechanism actually used.

Stable defaults remain unchanged.
