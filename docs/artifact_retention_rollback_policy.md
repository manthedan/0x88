# Artifact retention and rollback policy

Last updated: 2026-06-22

This policy applies to LC0 browser model/engine artifacts published through content-addressed release manifests and R2-backed artifact hosting.

## Invariants

1. **Never overwrite content-addressed keys.**
   - Keys under `/artifacts/sha256/<sha256>/<file>` are write-once.
   - A release/publish tool must verify that the key hash equals the local file hash before upload.
   - If a key already exists, treat it as immutable. Do not replace it during routine releases.

2. **Rollback only repoints channels.**
   - Rollback means editing `/channels/stable.json` to point at an older immutable release manifest.
   - Do not mutate old release manifests.
   - Do not purge immutable blobs during rollback.

3. **Retain old release manifests and blobs.**
   - Minimum retention: one full app-release cycle.
   - Preferred retention: 30–90 days.
   - Longer retention is acceptable for externally referenced or compliance-sensitive engine/source artifacts.

4. **Publish in dependency order.**
   - Upload all new `/artifacts/sha256/*` blobs first.
   - Validate size/hash/content type/range behavior through the asset hostname.
   - Publish the immutable release manifest.
   - Update the channel pointer last.

5. **Generated assets remain out of git unless policy changes.**
   - Release manifests may be committed or generated in CI.
   - Large model/engine blobs should be published to R2, not committed to the app repo.

## Operational rollback

To roll back from release `bad` to release `good`:

1. Confirm `public/releases/good.json` or the hosted equivalent still exists.
2. Confirm the blobs referenced by `good` are still present under `/artifacts/sha256/*`.
3. Update `/channels/stable.json` to reference `/releases/good.json`.
4. Purge only `/channels/stable.json` if immediate propagation is required.
5. Do not purge or overwrite `/releases/bad.json` or any `/artifacts/sha256/*` keys.

## Cleanup

Artifact garbage collection is a separate maintenance action, never part of routine release or rollback.

A safe cleanup candidate must satisfy all of the following:

- Not referenced by any retained channel manifest.
- Not referenced by any retained release manifest.
- Older than the retention window, preferably 30–90 days.
- Not needed for license/source distribution obligations.
- Not part of a rollback window for the currently deployed app shell.

Use the cleanup planner before deleting anything:

```sh
CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... npm run --silent deploy:r2-cleanup-plan -- --json
```

The planner is dry-run by default. It never deletes `channels/` or `releases/`, never deletes a hashed artifact referenced by any retained release manifest, and protects unreferenced source archives for manual license review. If a dry-run looks safe, delete only explicit low-risk categories, for example:

```sh
CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \
  npm run --silent deploy:r2-cleanup-plan -- \
  --execute \
  --delete-category legacy-unreferenced-metadata
```

Deleting `artifacts/sha256/*` candidates requires both `--delete-category hashed-orphan` and `--allow-delete-hashed`. Do not use those flags until the candidate is outside the retention window and has been manually reviewed.

## Tooling hooks

- `scripts/write_artifact_release_manifests.mjs` verifies local file byte counts and SHA-256 before emitting a release entry.
- `scripts/publish_hashed_artifacts_to_r2.mjs` verifies local bytes, requires the `/artifacts/sha256/<sha>/...` key hash to match the file hash before planning/uploading, and publishes release/channel manifest JSON after blob uploads when `--channel-manifest` is provided.
- `scripts/validate_artifact_cdn_headers.mjs` validates HEAD, repeated HEAD, range, CORS/CORP, timing, no-cookie, cache-status, and encoding behavior.
- `scripts/plan_r2_artifact_cleanup.mjs` lists R2 objects through the Cloudflare API, compares them against retained release manifests, and defaults to a no-delete cleanup plan.

Actual production uploads should use an R2 role that can put new objects but should be treated operationally as write-once for `/artifacts/sha256/*`.
