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

This prevents Vite from emitting identical ORT WASM files into both the main asset graph and evaluator-worker asset graphs. `npm run deploy:ort-wasm-dedup-check` verifies that:

- `dist-client/ort/ort-wasm-simd-threaded.asyncify.wasm` exists, and
- no `ort-wasm*.wasm` files were emitted under `dist-client/_app/`.

Run this check against the final R2-pruned deployment directory, not only a development build.

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
