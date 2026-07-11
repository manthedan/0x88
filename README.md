# 0x88

Browser-first 0x88 chess-engine research prototype.

The current trunk is `main` and is intentionally based on the browser LC0 arena/UI work. The `lc0-webgpu-pivot` branch is a separate active WebGPU research branch and should only be changed with explicit approval.

## Browser entry points

- `/` — landing page.
- `/app/play` — focused play surface.
- `/app/arena` — engine arena UI.
- `/app/analysis` — multi-engine analysis UI.
- `/single-engine` — LC0 policy/eval browser playground.
- `/docs` — browser-product documentation surface.

Smoke/probe/benchmark-only pages live under `lab/` so the app routes stay limited to product-facing entry points.

## Local development

```sh
npm install
npm run web:client
```

For SharedArrayBuffer/threaded WASM experiments, build and serve with isolation headers:

```sh
npm run build:client
npm run web:isolated:static
```

Then open `http://localhost:5181/app/arena`.

## Engine integration and artifacts

- `docs/engine_integration_architecture.md` documents the family registry, typed runtime boundaries, disposable worker pools, artifact tooling, ORT packaging, and engine-onboarding checklist.
- `docs/hosted_artifacts.md` tracks hosted/deployable engine and model versions.
- `docs/engine_artifact_distribution.md` defines the binary, manifest, licensing, and corresponding-source release gate.
- `public/artifact-index.json` provides the compact machine-readable artifact summary.

## Validation

Common frontend checks:

```sh
npm run typecheck
npm test
npm run build:netlify:r2
npm run deploy:ort-wasm-dedup-check
npm run deploy:ort-runtime-assets-check
```

The nightly production journey can also be run manually with `npm run production:browser-smoke`.
