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

Status: native probe scripted in [`reckless_browser_api_probe.md`](./reckless_browser_api_probe.md). It confirms a direct Rust facade can reproduce UCI bestmove/score/nodes/PV for a smoke position without using UCI as the primary data path.

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

Use a small C-style/string-buffer ABI first, avoiding a new JS dependency:

- Export `reckless_api_new(hash_mb) -> u32` engine handle.
- Export `reckless_api_set_fen(handle, ptr, len) -> i32`.
- Export `reckless_api_search_depth(handle, depth) -> i32`.
- Export `reckless_api_result_json_ptr(handle) -> *const u8` and `..._len(handle) -> usize`, or write JSON into a caller-provided output buffer.
- Export `reckless_api_free(handle)`.

This can initially still target `wasm32-wasip1` if needed, but it bypasses argv/stdin/stdout and UCI text parsing. A later `wasm32-unknown-unknown`/`cdylib` build can remove remaining WASI shim cost if Reckless dependencies cooperate.

### Stage 3: browser adapter replacement

Add a new worker next to `recklessWasiWorker.ts` that:

1. compiles/instantiates the direct API module,
2. owns one engine handle,
3. sends compact request/response messages,
4. returns structured search results directly to `RecklessEngine`.

Keep the existing WASI/UCI adapter as fallback until parity and performance are proven.

## Verification requirements

Before switching UI defaults:

1. Full-engine parity on a fixed FEN suite: bestmove, score class, depth, nodes presence, and legal PV.
2. MultiPV parity for analysis mode.
3. Browser benchmark comparison with:
   - one-shot WASI/UCI,
   - persistent WASI/UCI,
   - direct API persistent.
4. Abort/cancel behavior test; if direct `stop()` is not ready, document that it still falls back to worker termination.

## Expected payoff

This path attacks the user-visible shallow-search latency that SIMD cannot fix: no UCI command formatting/parsing, no stdout capture/filtering, fewer worker messages, and a clearer route to graceful stop/reuse. It should be benchmarked separately from engine NPS because it mostly improves wall-clock adapter overhead.
