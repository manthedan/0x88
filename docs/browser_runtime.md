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

## Runtime audit and targeted productization smoke

Runtime audit events are intentionally browser-visible and machine-readable. leelaweb emits `console.info('[lc0-browser-runtime-audit]', detail)` plus the `lc0-browser-runtime-audit` `window` event. Payload fields distinguish model identity (`family`, `modelId`, `modelUrl`, `metaUrl`) from runtime configuration (`requestedRuntime`, `resolvedRuntime`, `runtimeConfigId`, `manifestUrl`, `fallbackReason`, `searchBudget`). The LC0 audit panels filter to `family === 'lc0'`; Tiny Leela audit events remain available in the browser event stream without overwriting LC0 runtime details.

The targeted productization smoke for runtime-audit/Tiny-LC0 work is:

```sh
npm run productization:targeted-smoke
```

`npm run productization:fast-gate` remains as a compatibility alias, but new documentation and release notes should use `productization:targeted-smoke` to avoid implying full shipped-path parity coverage.

By default it runs:

- `npm run typecheck`;
- syntax check for `scripts/lc0_tiny_strict_custom_webgpu_smoke.mjs`;
- targeted runtime/catalog/analysis tests: `tests/engine_catalog.test.mjs`, `tests/lc0_analysis_format.test.mjs`, and `tests/lc0_stable_backend_defaults.test.mjs`;
- a dry-run of the strict Tiny custom WebGPU browser smoke wiring.

Use the real browser/WebGPU smoke only on hosts with current Chrome/WebGPU and the versioned Tiny hybrid bundle present:

```sh
npm run productization:targeted-smoke -- --strict-browser-smoke
```

That path drives `lc0-analysis.html` and `lc0-arena.html`, installs an audit-event collector, requests Tiny `runtime=custom-webgpu` with strict fallback disabled, and fails on missing `custom-webgpu` resolution or any ORT fallback event. Save evidence with `--out artifacts/targeted_productization_smoke.json`.

Known exclusions are deliberate: full `npm test` remains noisy because some fixture/artifact-dependent tests are pre-existing productization blockers, and cross-browser/cross-GPU latency/lifecycle repeats remain release-candidate evidence rather than every-commit checks. Treat `npm run productization:targeted-smoke` as a targeted productization smoke, not a full shipped-path parity guarantee.

For shipped LC0 WebGPU parity, run the browser CI smoke on a host where WebGPU is expected and fail closed on WebGPU unavailability:

```sh
npm run lc0:browser-ci-smoke
```

That smoke keeps two separate WGSL-heads comparisons:

- `wgsl-heads-vs-ort-wasm-fixtures`: semantic baseline against ORT WASM, retained so custom WGSL heads stay anchored to the stable fallback path;
- `wgsl-heads-vs-ort-webgpu-fixtures`: shipped-path comparison against ORT WebGPU with `strictWebGpu=1`, which fails if the browser lacks WebGPU, ORT WebGPU cannot be selected, or ORT silently falls back to WASM during the fixture run.

Current policy: ORT-Web remains the production inference baseline. Alternative inference runtimes are bounded research probes behind explicit runtime selection and evidence gates.
