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

See also: [Browser inference research lane: Rust-owned search, optional Rust-owned inference](browser_inference_research_lane_20260522.md). The current opinion there is that ORT-Web remains the production inference baseline, while RTen/tract/Lele/Burn/Candle are bounded research probes behind a Rust evaluator abstraction.
