# Browser Runtime Plan

For the canonical cross-engine Runtime Configuration, benchmark artifact, and footprint schemas, see [Browser Runtime Configuration and Benchmark Schema](browser_runtime_configuration_and_benchmark_schema.md).

The shared engine substrate is backend-neutral:

1. chess rules and move encoding live in `src/chess/`;
2. feature encoding and `Evaluator` live in `src/nn/`;
3. search consumes only `BoardState` plus an `Evaluator`.

Research lanes may add:

- ONNX Runtime Web WASM evaluator;
- WebGPU/FP16 evaluator;
- worker message protocol and model cache;
- progressive loading from micro to balanced model.

Those lanes must not change move encoding or feature fixtures without a new benchmark id.

## Cross-origin isolation / serious browser mode

Threaded WASM requires `SharedArrayBuffer`, which browsers only expose when the page is cross-origin isolated. Local/dev/preview servers should send:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

This unlocks two opt-in high-strength paths:

- threaded Stockfish WASM (`sfFlavor=lite-threaded` or `sfFlavor=threaded`, optionally `sfThreads=N`);
- threaded ORT WASM CPU inference (`ortThreads=auto` or `ortThreads=N`).

Defaults remain conservative: Stockfish lite single-thread and ORT WASM thread count 1. The arena runtime badge reports isolation, `SharedArrayBuffer`, WebGPU availability, ORT provider/threads, and whether threaded Stockfish can be selected.

Vite dev/preview is configured with the isolation headers. For static built assets, use:

```sh
npm run build:client
npm run web:isolated:static
```

The full Stockfish assets are exposed through `public/stockfish/` as symlinks to the installed `stockfish` package so they can be loaded on demand without committing 100MB+ WASM blobs. ORT's threaded WASM sidecars are exposed through `public/ort/` for the same reason; this lets ORT load its pthread worker module when `ortThreads` is greater than 1.

## Runtime audit and fast productization gate

Runtime audit events are intentionally browser-visible and machine-readable. leelaweb emits `console.info('[lc0-browser-runtime-audit]', detail)` plus the `lc0-browser-runtime-audit` `window` event. Payload fields distinguish model identity (`family`, `modelId`, `modelUrl`, `metaUrl`) from runtime configuration (`requestedRuntime`, `resolvedRuntime`, `runtimeConfigId`, `manifestUrl`, `fallbackReason`, `searchBudget`). The LC0 audit panels filter to `family === 'lc0'`; Tiny Leela audit events remain available in the browser event stream without overwriting LC0 runtime details.

The fast gate for runtime-audit/Tiny-LC0 productization is:

```sh
npm run productization:fast-gate
```

By default it runs:

- `npm run typecheck`;
- syntax check for `scripts/lc0_tiny_strict_custom_webgpu_smoke.mjs`;
- targeted runtime/catalog/analysis tests: `tests/engine_catalog.test.mjs`, `tests/lc0_analysis_format.test.mjs`, and `tests/lc0_stable_backend_defaults.test.mjs`;
- a dry-run of the strict Tiny custom WebGPU browser smoke wiring.

Use the real browser/WebGPU smoke only on hosts with current Chrome/WebGPU and the versioned Tiny hybrid bundle present:

```sh
npm run productization:fast-gate -- --strict-browser-smoke
```

That path drives `lc0-analysis.html` and `lc0-arena.html`, installs an audit-event collector, requests Tiny `runtime=custom-webgpu` with strict fallback disabled, and fails on missing `custom-webgpu` resolution or any ORT fallback event. Save evidence with `--out artifacts/productization_fast_gate.json`.

Known exclusions from the fast gate are deliberate: full `npm test` remains noisy because some fixture/artifact-dependent tests are pre-existing productization blockers, and cross-browser/cross-GPU latency/lifecycle repeats remain release-candidate evidence rather than every-commit checks.

See also: [Browser inference research lane: Rust-owned search, optional Rust-owned inference](browser_inference_research_lane_20260522.md). The current opinion there is that ORT-Web remains the production inference baseline, while RTen/tract/Lele/Burn/Candle are bounded research probes behind a Rust evaluator abstraction.
