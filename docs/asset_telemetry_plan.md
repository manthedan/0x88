# Asset and runtime telemetry plan

Status: proposed

This document defines privacy-preserving operational telemetry for the 0x88 browser application and `assets.0x88.app`. The immediate goal is to measure artifact reliability, cache behavior, runtime fallback, and startup failures during production soak without collecting chess content or identifying users.

## Goals

- Measure model and engine artifact request volume, latency, status, range use, and cache behavior.
- Confirm that immutable artifacts and the mutable stable channel behave as designed over time.
- Measure browser runtime outcomes that cannot be inferred from HTTP requests, especially WebGPU acceptance, WASM fallback, model-cache hits, and initialization failures.
- Establish release-over-release baselines and identify regressions quickly.
- Keep telemetry inexpensive, bounded, schema-versioned, and privacy-preserving.

## Non-goals

- Product analytics, engagement tracking, advertising attribution, or user profiling.
- Recording FENs, PGNs, moves, engine output, search positions, or analysis content.
- Storing raw IP addresses, full user-agent strings, persistent user IDs, or browser fingerprints.
- Replacing retained production-smoke reports or Workers request logs used for targeted debugging.
- Building a permanent raw-event data lake in the first iteration.

## Recommended architecture

### Cloudflare Workers Analytics Engine

Use Workers Analytics Engine as the primary time-series store. Attach one dataset to the existing artifact Worker:

- Worker: `cloudflare/artifact-assets-worker.mjs`
- Wrangler configuration: `cloudflare/artifacts.wrangler.toml`
- Binding: `ASSET_TELEMETRY`
- Dataset: `lc0_asset_telemetry`

```toml
[[analytics_engine_datasets]]
binding = "ASSET_TELEMETRY"
dataset = "lc0_asset_telemetry"
```

The dataset is created automatically on the first write. `writeDataPoint()` is non-blocking and must not be awaited in the response path.

Analytics Engine is appropriate because these records are high-cardinality operational measurements queried as aggregates. It is preferable to D1 for request events and preferable to raw R2 objects for interactive time-series queries.

Current Cloudflare documentation:

- <https://developers.cloudflare.com/analytics/analytics-engine/>
- <https://developers.cloudflare.com/analytics/analytics-engine/get-started/>
- <https://developers.cloudflare.com/analytics/analytics-engine/limits/>

### Two telemetry sources

Use the same dataset for two explicitly separated sources:

1. **Trusted server events** written by the artifact Worker after handling a request.
2. **Untrusted client events** accepted through a constrained telemetry endpoint and written by the Worker after validation.

Every event includes `source=worker` or `source=client`. Queries and alerts must be able to separate them.

### Optional long-term rollups

Analytics Engine currently retains data for three months. This is sufficient for the initial soak and release comparisons. If longer history becomes useful, generate one aggregate report per day and write it to a separate R2 bucket, not the artifact bucket:

```text
browser-chess-telemetry/
  daily/2026/07/11.json
```

Do not store raw client events in R2 by default. Separate telemetry from `browser-chess-models` so access controls, retention, and cleanup cannot affect production artifacts.

## Event schema

Analytics Engine uses ordered `blobs`, `doubles`, and one sampling index. Field order is therefore a versioned contract. Do not insert fields into an existing schema; create a new event schema version instead.

### Worker asset request event

Event name: `asset_request_v1`

Suggested ordered blobs:

| Position | Name | Example | Notes |
| --- | --- | --- | --- |
| `blob1` | event | `asset_request_v1` | Schema discriminator. |
| `blob2` | source | `worker` | Trusted server measurement. |
| `blob3` | family | `stormphrax` | Bounded family name or `shared`. |
| `blob4` | asset_kind | `wasm` | `model`, `wasm`, `js`, `data`, `manifest`, `release`, `channel`, `source`, or `other`. |
| `blob5` | address_kind | `immutable` | `immutable`, `logical`, `release`, or `channel`. |
| `blob6` | request_kind | `full` | `head`, `full`, `range`, or `invalid_range`. |
| `blob7` | cache_status | `hit` | Normalized Worker cache outcome. |
| `blob8` | encoding | `br` | `br`, `gzip`, `identity`, or `unknown`. |
| `blob9` | status_class | `2xx` | Bounded status class. |
| `blob10` | release_id | `2026-07-11.stormphrax-relaxed-simd.3` | `unknown` when not resolved through a release. |
| `blob11` | colo | `SJC` | Coarse Cloudflare colo; optional. |

Suggested ordered doubles:

| Position | Name | Unit |
| --- | --- | --- |
| `double1` | status | HTTP status code. |
| `double2` | response_bytes | Bytes returned for this response. |
| `double3` | object_bytes | Full immutable object size when known. |
| `double4` | elapsed_ms | Worker handling duration. |
| `double5` | range_start | Start offset, or `-1`. |
| `double6` | range_end | End offset, or `-1`. |

Use a stable, non-user sampling index derived from the bounded logical asset ID, such as `family:asset_kind:logical_name`. Never use an IP address, cookie, or browser identifier as the index.

Example write:

```js
env.ASSET_TELEMETRY?.writeDataPoint({
  blobs: [
    'asset_request_v1',
    'worker',
    family,
    assetKind,
    addressKind,
    requestKind,
    cacheStatus,
    encoding,
    statusClass,
    releaseId,
    request.cf?.colo ?? 'unknown',
  ],
  doubles: [status, responseBytes, objectBytes, elapsedMs, rangeStart, rangeEnd],
  indexes: [assetId],
});
```

Telemetry failure must never fail or delay an artifact response.

### Browser runtime event

Event name: `runtime_event_v1`

Suggested ordered blobs:

| Position | Name | Examples |
| --- | --- | --- |
| `blob1` | event | `runtime_event_v1` |
| `blob2` | source | `client` |
| `blob3` | outcome | `runtime_started`, `runtime_fallback`, `model_cache_hit`, `model_cache_miss`, `download_complete`, `verification_failed`, `worker_failed`, `asset_probe_timeout` |
| `blob4` | surface | `play`, `analysis`, `arena`, `diag` |
| `blob5` | family | `centipawn`, `stormphrax`, `lc0` |
| `blob6` | variant | Bounded catalog variant key. |
| `blob7` | requested_provider | `webgpu`, `wasm`, `auto`, `cpu` |
| `blob8` | resolved_provider | `webgpu`, `wasm`, `cpu`, `none` |
| `blob9` | failure_code | Bounded code or `none`; never a raw exception message. |
| `blob10` | browser_family | `chromium`, `firefox`, `safari`, `other`; optional and coarse. |
| `blob11` | release_id | Current stable release ID or `unknown`. |

Suggested ordered doubles:

| Position | Name | Unit |
| --- | --- | --- |
| `double1` | elapsed_ms | Initialization, download, or probe duration. |
| `double2` | bytes | Downloaded/model bytes when known. |
| `double3` | cache_hit | `1` or `0`. |
| `double4` | success | `1` or `0`. |

Client events are operational hints, not authoritative facts. Data validation and dashboards must label them accordingly.

## Client ingestion endpoint

Add a route to the artifact Worker:

```text
POST https://assets.0x88.app/telemetry/v1/events
```

Suggested browser helper:

```text
src/lc0/runtimeTelemetry.ts
```

The helper should use `navigator.sendBeacon()` where suitable, with `fetch(..., { keepalive: true })` as a fallback. Telemetry must never block engine startup, navigation, or disposal.

The Worker endpoint must:

- Accept `POST` and `OPTIONS` only.
- Permit `Access-Control-Allow-Origin: https://0x88.app` only.
- Reject bodies above a small fixed limit, initially 2 KiB.
- Require `Content-Type: application/json`.
- Validate event names, catalog families, variants, providers, surfaces, and failure codes against bounded allowlists.
- Reject free-form dimensions and unknown object keys.
- Clamp numeric fields to sane non-negative limits.
- Write `source=client` itself rather than trusting a supplied source.
- Return `204` after validation.
- Apply a rate limit before increasing the client sampling rate.

Because the endpoint is unauthenticated, clients can fabricate events. Keep server and client measurements separate and never use client telemetry for billing, security decisions, or artifact integrity decisions.

## Sampling

Start conservatively:

- Record all Worker-side failures, invalid ranges, release/channel requests, and cache misses.
- Sample successful immutable artifact requests at 5%.
- Record all client failures and fallbacks.
- Sample successful client startup/cache events at 5%.
- Do not sample the nightly production-smoke synthetic events; tag them with a bounded `synthetic` dimension or suppress them consistently.

Analytics Engine may also sample data internally. Queries must use `_sample_interval` when calculating counts and weighted averages.

Sampling decisions must be random per event or based on a non-user event key. Do not create a persistent browser sampling identifier.

## Privacy and security

Never collect:

- FEN, PGN, moves, positions, search PVs, engine output, or imported-game content.
- IP addresses in custom fields.
- Full user-agent strings.
- Cookies, account IDs, localStorage IDs, cache keys unique to a browser, or stable fingerprints.
- Raw exception messages, stack traces, or arbitrary URLs.
- Query strings from user-facing routes.

Allowed dimensions must be coarse and bounded. Sanitize failure information into codes such as:

```text
network
http_404
http_5xx
probe_timeout
hash_mismatch
webgpu_unavailable
webgpu_init_failed
wasm_init_failed
worker_crash
abort
unknown
```

Cloudflare may process ordinary request metadata as part of operating the Worker, but the application must not copy identifying request data into Analytics Engine.

## Initial dashboards and queries

### Artifact failure rate by family

```sql
SELECT
  intDiv(toUInt32(timestamp), 300) * 300 AS bucket,
  blob3 AS family,
  SUM(if(double1 >= 400, _sample_interval, 0)) /
    SUM(_sample_interval) AS failure_rate
FROM lc0_asset_telemetry
WHERE blob1 = 'asset_request_v1'
  AND timestamp >= NOW() - INTERVAL '1' DAY
GROUP BY bucket, family
ORDER BY bucket, family
```

### Cache-hit ratio by asset kind

```sql
SELECT
  blob4 AS asset_kind,
  SUM(if(blob7 = 'hit', _sample_interval, 0)) /
    SUM(_sample_interval) AS cache_hit_ratio
FROM lc0_asset_telemetry
WHERE blob1 = 'asset_request_v1'
  AND timestamp >= NOW() - INTERVAL '1' DAY
GROUP BY asset_kind
ORDER BY cache_hit_ratio ASC
```

### Runtime fallback rate

```sql
SELECT
  blob5 AS family,
  blob7 AS requested_provider,
  blob8 AS resolved_provider,
  SUM(_sample_interval) AS estimated_events
FROM lc0_asset_telemetry
WHERE blob1 = 'runtime_event_v1'
  AND blob3 = 'runtime_fallback'
  AND timestamp >= NOW() - INTERVAL '7' DAY
GROUP BY family, requested_provider, resolved_provider
ORDER BY estimated_events DESC
```

Additional launch queries:

- P50/P95 Worker latency by artifact kind.
- Full versus range request volume.
- Cache misses by release ID.
- Model cache hit/miss rate by family and provider.
- WebGPU requested-to-resolved transition rates.
- Probe timeout and verification failure trends.
- Requests still resolving through an older stable release.

## Reporting and alerting

### Phase 1

Create a query script, for example:

```text
scripts/query_asset_telemetry.mjs
```

It should call the Analytics Engine SQL API using an API token with Account Analytics Read permission and emit a redacted JSON summary. A scheduled GitHub workflow can retain the daily summary alongside production-smoke reports.

Suggested initial alert candidates after a baseline exists:

- Artifact HTTP failure rate above 1% over 15 minutes.
- Channel or release manifest failures above zero outside deployments.
- P95 immutable artifact latency exceeding the observed baseline by a defined multiplier.
- Runtime startup failure above 2% over one hour.
- WebGPU-to-WASM fallback increasing materially for the same browser-family mix.
- Stable release resolution reporting multiple unexpected release IDs.

Do not set hard thresholds before collecting at least several days of baseline data.

### Phase 2

If the daily reports become cumbersome, connect Grafana to the Analytics Engine SQL API. Keep Worker Observability enabled for targeted debugging. Consider Workers Logpush to a separate R2 bucket only when raw request forensics justify its volume and retention cost.

References:

- <https://developers.cloudflare.com/workers/observability/>
- <https://developers.cloudflare.com/logs/logpush/>
- <https://developers.cloudflare.com/logs/logpush/logpush-job/enable-destinations/r2/>

## Implementation sequence

1. Add the Analytics Engine binding to `cloudflare/artifacts.wrangler.toml`.
2. Define schema constants and a defensive `writeAssetTelemetry()` helper in `cloudflare/artifact-assets-worker.mjs`.
3. Instrument Worker request completion, including errors, ranges, cache status, response bytes, and elapsed time.
4. Add unit tests proving telemetry writes cannot alter responses and that no identifying fields are emitted.
5. Deploy server telemetry and collect a one-week baseline.
6. Add the validated `/telemetry/v1/events` endpoint.
7. Add `src/lc0/runtimeTelemetry.ts` and instrument a small set of runtime outcomes.
8. Add SQL query fixtures and a daily summary workflow.
9. Review cardinality, sampling, failure codes, and usefulness after the soak.
10. Decide whether Grafana, alerting, or daily R2 rollups are warranted.

## Acceptance criteria

The first production increment is complete when:

- The artifact Worker records schema-versioned request events without affecting response behavior or latency materially.
- Dashboards can show request volume, cache behavior, latency, status, and release resolution by bounded asset dimensions.
- No event contains chess content, persistent identifiers, raw user agents, arbitrary URLs, or free-form errors.
- Sampling-adjusted SQL queries are checked into the repository and exercised against the deployed dataset.
- Telemetry write failures are invisible to artifact clients.
- The nightly production journey still passes.
- Documentation identifies the dataset, schema field order, sampling policy, retention, and deletion/rollup policy.
