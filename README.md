# lc0_webgpu

Browser-first LC0/WebGPU chess-engine research prototype.

The current trunk is `main` and is intentionally based on the browser LC0 arena/UI work. The `lc0-webgpu-pivot` branch is a separate active WebGPU research branch and should only be changed with explicit approval.

## Browser entry points

- `lc0-arena.html` — LC0 vs Stockfish arena UI.
- `lc0-policy-only.html` — LC0 policy/eval browser playground.
- `lc0-analysis.html` — LC0 analysis UI.

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

Then open `http://localhost:5181/lc0-arena.html`.

## Artifact inventory

Hosted/deployable engine and model versions are tracked in `docs/hosted_artifacts.md`, with a compact machine-readable summary at `public/artifact-index.json`.

## Validation

Common frontend checks:

```sh
npm run typecheck
npm run build:client
```
