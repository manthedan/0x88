# Engine integration architecture

This document describes the code and deployment boundaries used when adding or maintaining browser chess-engine families. It complements the user-facing engine catalog and the artifact distribution policy.

## Browser registry boundaries

`src/lc0/engineCatalog.ts` is the browser-facing source of truth for concerns shared across Play, Analysis, and Arena:

- family ids, labels, aliases, and selector order
- runtime kind and resource profiles
- Analysis/Arena strength ranges and defaults
- production (`v0`) variant allowlists and fallbacks
- Play options and strength ladders

Keep runtime-specific behavior in typed modules rather than expanding the catalog into a generic plugin framework:

- `*Variants.ts` modules own URL parsing, feature detection, asset probes, and fallback policy.
- `*Engine.ts` modules own initialization, command/search behavior, cancellation, and disposal.
- `engineProvision.ts` owns shared construction and cache-key derivation for standard browser UCI engines.
- Neural runtimes retain their specialized evaluator and search paths.

## Analysis and Arena lifecycle

Analysis and Arena retain expensive browser UCI workers only while a matching resolved variant remains selected.

`src/lc0/disposableVariantPool.ts` provides the shared lifecycle primitive:

- `getOrCreate(variant)` reuses a resource by its complete runtime cache key.
- `peek(variant)` supports diagnostics without creating a worker.
- `retain(activeVariants)` disposes workers that are no longer selected.
- `disposeAll()` performs page teardown idempotently.

Every pooled resource must implement `dispose()`. Cache keys must include every URL/backend field that changes the executable runtime; a display variant key alone is insufficient.

`src/lc0/browserVariantOption.ts` converts family-specific probe states into consistent selector behavior. It accepts both `ok` and `present` success states, distinguishes optional probes from required generated artifacts, and gives unsupported-browser reasons priority over missing-asset text.

## Build and artifact registry

Node deployment tooling uses `scripts/engine_artifact_registry.mjs`. This registry is intentionally separate from `engineCatalog.ts` so build commands and repository paths are not shipped in the browser bundle.

It centralizes:

- required and optional browser-engine asset groups
- build/preparation commands and documentation anchors
- externally hosted engine directories
- precompression directories and extensions
- external artifact filename classification

The registry is consumed by asset checks, R2 staging, external-asset pruning, release verification, and precompression. Add a new externally hosted engine directory here instead of adding parallel directory lists to individual scripts.

## ONNX Runtime packaging

The application configures ONNX Runtime to load glue and WASM from the staged `/ort/` directory. `vite.config.ts` selects ONNX Runtime's `onnxruntime-web-use-extern-wasm` conditional export while retaining Vite's normal client resolution conditions.

Production staging is intentionally allowlisted in `scripts/ort_runtime_assets.mjs`. The application imports only `onnxruntime-web/webgpu`; in ONNX Runtime 1.27 that entrypoint requests the Asyncify glue/binary pair for both accepted WebGPU sessions and WASM fallback sessions:

- `ort-wasm-simd-threaded.asyncify.mjs`
- `ort-wasm-simd-threaded.asyncify.wasm`

This was verified with a production build running a Centipawn move through an accepted WebGPU provider while recording `/ort/` requests. The package's other JS bundles, source maps, base WASM, JSEP, and JSPI variants are not copied into the deployment shell. `npm run deploy:ort-runtime-assets-check` rejects missing or unexpected runtime variants, including compressed sidecars. `npm run deploy:ort-wasm-dedup-check` separately verifies that the canonical fallback WASM exists and no `ort-wasm*.wasm` files were emitted under `dist-client/_app/`.

Run both checks against the final R2-pruned deployment directory, not only a development build.

## Browser UX policy

Play persists the selected opponent, strength, side, and Maia controls in local storage. Analysis persists manual engine rows separately from named profiles; selecting “manual / default” restores those rows. Analysis deliberately downgrades persisted large Lc0 nets to the small net so a page visit cannot silently trigger a very large download. Arena persists seat families, variants, and strengths; explicit URL parameters still override stored Arena seats. Corrupt, stale, or unavailable storage always falls back to current safe defaults.

Play exposes download/search progress, retryable and actionable engine failures, and selection-specific cautions. Play, Analysis, and Arena share the compact browser-capability panel so WebGPU, shared-memory/threaded WASM, CPU capacity, and model-cache availability are visible before an engine is selected.

## CI artifact checkout

CI uses Git LFS skip-smudge behavior and explicitly pulls only release fixtures read by the test suite. The Centipawn release integrity test currently requires:

```text
public/models/bt4_soap_rem_c19000_final.onnx
```

Do not restore blanket `lfs: true` checkout unless tests genuinely require every browser engine binary and corresponding-source archive. When a test begins reading another LFS body, add that exact path to the selective pull step.

## Adding an engine family

1. Add shared family metadata to `ENGINE_FAMILY_DEFINITIONS`.
2. Add a typed variant module with URL validation, probes, and fallback behavior.
3. Add or reuse a `BrowserUciEngine` implementation with reliable abort and disposal semantics.
4. Add standard construction/cache-key wiring to `engineProvision.ts` when applicable.
5. Use `DisposableVariantPool` on Analysis/Arena rather than adding another family-specific `Map` lifecycle.
6. Add artifact inventory and preparation metadata to `engine_artifact_registry.mjs`.
7. Add targeted-test mappings in `scripts/run_targeted_tests.mjs`.
8. Validate typecheck, the full Node suite, the R2-pruned build, ORT deduplication, and browser smokes.
9. For distributed third-party binaries, follow `docs/engine_artifact_distribution.md` and publish matching immutable artifacts, manifests, notices, and corresponding source.

## Validation commands

```sh
npm run typecheck
npm test
npm audit
npm run build:netlify:r2
npm run deploy:ort-wasm-dedup-check
npm run lc0:analysis-browser-smoke -- --base-url http://127.0.0.1:5181
```

For runtime-sensitive changes, also select the affected engine in Play/Arena/Analysis and wait for an actual move or completed search before deployment.

## Scheduled production smoke

`.github/workflows/production-smoke.yml` runs a nightly browser journey against `https://0x88.app` and can also be started manually with an alternate production origin. The workflow pins its browser harness version and uploads a JSON report for 30 days.

The journey intentionally uses one browser session while navigating between product surfaces so route teardown and remount behavior is exercised:

1. Centipawn completes a first move in Play.
2. Arena exposes the canonical family order with Centipawn last.
3. Analysis exposes the same order and resolves runtime diagnostics.
4. Stormphrax completes a first move in a newly mounted Play page.
5. Browser errors, HTTP 4xx/5xx requests, the app-shell cache policy, and the canonical ORT WASM cache policy are checked.

Run the same gate locally against production with:

```sh
npm run production:browser-smoke -- --out /tmp/production-browser-smoke.json
```
