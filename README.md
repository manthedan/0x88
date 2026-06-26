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

## Artifact inventory

Hosted/deployable engine and model versions are tracked in `docs/hosted_artifacts.md`, with a compact machine-readable summary at `public/artifact-index.json`.

## Validation

Common frontend checks:

```sh
npm run typecheck
npm run build:client
```
