# Reckless browser-native API plan

## Why

The current browser adapter drives a patched `wasm32-wasip1` Reckless UCI binary through WASI argv/stdin/stdout. Persistent mode hides most cold startup cost, but every search still pays for UCI command parsing, stdout formatting/parsing, worker message fan-out, and a cancellation path that terminates the worker rather than stopping the search cleanly.

A browser-native API should expose engine operations directly and return structured search results instead of formatted UCI lines.

## Current source constraints

- Reckless is currently a binary crate (`src/main.rs`) with private modules and no `lib.rs` facade.
- `main()` performs global initialization:
  - `lookup::initialize()`
  - `nnue::initialize()`
  - then hands argv commands to `uci::message_loop()`
- `uci::message_loop()` owns the important runtime state:
  - `Arc<SharedContext>`
  - `Settings`
  - `ThreadPool`
  - `Board`
- The useful operations are already factored internally:
  - `position(...)`
  - `set_option(...)`
  - `reset(...)`
  - `go(...)`
- Those helpers are private to `uci.rs`; a direct API can live inside the Reckless crate and use them or equivalent code, but the project needs a library/API facade instead of only a binary main.

## Proposed staged implementation

### Stage 1: internal Rust facade

Status: implemented for the local browser build path in `scripts/build_reckless_browser_api.mjs`, following the earlier native probe in [`reckless_browser_api_probe.md`](./reckless_browser_api_probe.md). The build script patches a temporary Reckless library facade and emits ignored artifact `/reckless/reckless-browser-api.wasm`.

Add a `browser_api` module behind a feature flag in the local Reckless patch/build path:

```rust
pub struct BrowserEngine {
    shared: Arc<SharedContext>,
    settings: Settings,
    threads: ThreadPool,
    board: Board,
}
```

Expose methods that mirror the current adapter needs:

- `new(hash_mb: usize) -> BrowserEngine`
- `set_fen(&mut self, fen: &str) -> Result<(), ApiError>`
- `new_game(&mut self)`
- `search_depth(&mut self, depth: u32) -> SearchResult`
- `search_movetime(&mut self, ms: u32) -> SearchResult`
- future: `stop()` / shared abort flag

`SearchResult` should carry structured fields currently parsed from UCI:

- `bestmove: Option<String>`
- `depth`
- `score_cp` / `mate_in`
- `nodes`
- `nps`
- `pv: Vec<String>`
- optional `multipv: Vec<SearchLine>`

### Stage 2: WASM export ABI

Status: implemented experimentally. The artifact still targets `wasm32-wasip1` because `std::time::Instant` panics on `wasm32-unknown-unknown`; the browser worker uses WASI imports for clocks but bypasses `_start`, argv/stdin/stdout, UCI parsing, and UCI output formatting.

Use a small C-style/string-buffer ABI first, avoiding a new JS dependency:

- Export `reckless_api_new(hash_mb) -> u32` engine handle.
- Export `reckless_api_new_with_network(hash_mb, ptr, len) -> u32` for external-NNUE browser API builds.
- Export `reckless_api_set_fen(handle, ptr, len) -> i32`.
- Export `reckless_api_search_depth(handle, depth) -> i32`.
- Export `reckless_api_result_json_ptr(handle) -> *const u8` and `..._len(handle) -> usize`, or write JSON into a caller-provided output buffer.
- Export `reckless_api_free(handle)`.

This can initially still target `wasm32-wasip1` if needed, but it bypasses argv/stdin/stdout and UCI text parsing. A later `wasm32-unknown-unknown`/`cdylib` build can remove remaining WASI shim cost if Reckless dependencies cooperate.

### Stage 3: browser adapter replacement

Status: implemented experimentally in `src/lc0/recklessBrowserApiWorker.ts` and selected through the `Reckless Full browser API experimental` variant. The worker:

1. compiles/instantiates the direct API module,
2. initializes WASI imports for clocks without starting the UCI binary,
3. optionally fetches/cache the external NNUE `ArrayBuffer` and passes it to `reckless_api_new_with_network`,
4. owns one engine handle,
5. sends compact request/response messages,
6. returns structured search results directly to `RecklessEngine`.

The existing WASI/UCI adapter remains the default and fallback until parity and performance are proven. The direct API reset path must keep mirroring UCI `ucinewgame`; it now clears thread state, TT, and correction-history tables.

Smoke/performance evidence:

- Node direct-ABI smoke: `startpos depth 4` returned `bestmove=c2c4`, `scoreCp=55`, `nodes=210`, PV `c2c4 g8f6`.
- Browser benchmark smoke on isolated static server: browser API variant, persistent mode label, `startpos depth 1`, one warm repeat completed 2 rows; cold wall `4.70ms`, warm wall `1.74ms`, runtime label `browser API`.
- Rotated-FEN browser benchmark, persistent depth 7/8/9, 20 positions × 20 warm passes: browser API was slower than persistent WASI/UCI (`0.88x`, `0.81x`, and `0.80x` as fast by wall-clock for depths 7/8/9). See [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md).
- Browser API history-reset smoke after fixing correction-history clearing: scalar WASI/UCI, SIMD WASI/UCI, browser API scalar, and browser API SIMD all matched exactly across the 20-position suite at depths 7/8/9 with one warm rotated pass.
- Corrected full browser benchmark, persistent depth 7/8/9, 20 positions × 20 warm passes: browser API SIMD remained slower than SIMD WASI/UCI (`0.88x`, `0.87x`, and `0.98x` as fast by wall-clock for depths 7/8/9), while preserving exact fixed-depth parity. Browser API scalar was modestly faster than scalar WASI/UCI but still slower than SIMD WASI/UCI.
- External-NNUE browser API SIMD validation: `reckless-browser-api-simd128-external.wasm` is 1,260,734 bytes and loads `reckless-v60-7f587dfb.nnue` as a separate 63,266,880-byte asset; depth-4 startpos smoke returned the expected `c2c4`/210-node result, and a full rotated depth 7/8/9 run matched embedded browser API SIMD exactly in 1260/1260 fixed-depth pairs. See [`reckless_nnue_asset_size_plan.md`](./reckless_nnue_asset_size_plan.md) and [`reckless_browser_benchmarks.md`](./reckless_browser_benchmarks.md).

## Verification requirements

Before switching UI defaults:

1. Full-engine parity on a fixed FEN suite: bestmove, score class, depth, nodes presence, and legal PV.
2. MultiPV parity for analysis mode.
3. Browser benchmark comparison with:
   - one-shot WASI/UCI,
   - persistent WASI/UCI,
   - direct API persistent.
4. Abort/cancel behavior test; if direct `stop()` is not ready, document that it still falls back to worker termination.

## Expected payoff / current result

This path attacks the user-visible shallow-search latency that SIMD cannot fix: no UCI command formatting/parsing, no stdout capture/filtering, fewer worker messages, and a clearer route to graceful stop/reuse. It should be benchmarked separately from engine NPS because it mostly targets wall-clock adapter overhead.

Current result: the integrated browser API does **not** beat the SIMD WASI/UCI path yet. It is useful as a structured-result/control-path prototype, and the corrected reset path now has exact fixed-depth parity, but SIMD WASI/UCI remains the faster production default. Future browser API work should focus on removing the remaining WASI shim/clock dependency, reducing facade overhead, and adding true engine-side cancellation before considering it as the default.
