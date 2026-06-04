# Reckless hot-path profile notes

## 2026-06-04 native proxy profile

A browser/WASM profiler was not available in the local toolchain, so this pass used the patched local Reckless source as a native proxy to separate engine hot paths from the already-measured browser adapter overhead.

Build and sample command shape:

```sh
cd .local_engines/reckless-wasi-src
cargo build --release --no-default-features
printf 'uci\nisready\nposition fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\ngo movetime 10000\nquit\n' | target/release/reckless
# sampled the running PID for 5s with macOS `sample`, 1ms interval
```

The sampled run used `startpos` and reached ~1.3M NPS by depth 23-25. The full sample report was local-only, but the top-of-stack summary was:

| Exclusive top-of-stack item | Samples | Share of active worker samples | Bucket |
| --- | ---: | ---: | --- |
| `reckless::nnue::Network::evaluate` | 2213 | 51.5% | NNUE output/eval |
| `reckless::search::search` | 417 | 9.7% | search control |
| `reckless::nnue::forward::vectorized::propagate_l1` | 228 | 5.3% | NNUE output/eval |
| `reckless::nnue::accumulator::threats::scalar::push_threats_single` | 193 | 4.5% | NNUE accumulator/update |
| `reckless::nnue::accumulator::psq::PstAccumulator::refresh` | 171 | 4.0% | NNUE accumulator/update |
| `reckless::board::movegen::Board::generate_moves` | 166 | 3.9% | movegen |
| `reckless::board::see::Board::see` | 146 | 3.4% | move ordering/search |
| `reckless::movepick::MovePicker::next` | 145 | 3.4% | move ordering/search |
| `reckless::movepick::MovePicker::score_quiet` | 123 | 2.9% | move ordering/search |
| `reckless::search::make_move` | 105 | 2.4% | search/make move |
| `reckless::board::Board::update_threats` | 78 | 1.8% | NNUE-adjacent update |
| `reckless::search::qsearch` | 61 | 1.4% | search control |

Interpretation:

- The native engine profile is NNUE-heavy: `Network::evaluate` alone was ~51% of active worker samples, and named NNUE forward/accumulator/update frames account for roughly another 10-12% exclusive samples.
- Movegen/search/move-ordering still matter, but they are secondary for a SIMD experiment. A small hand-written WASM SIMD prototype should target the NNUE output/clipped dot-product path first, then accumulator add/sub/update if output parity and speed look good.
- The sampled native build uses host CPU codegen; it is not proof that the browser `wasm32-wasip1` artifact has the same exact split. The source inspection still matters: wasm currently selects scalar NNUE modules except for LLVM auto-vectorization under `+simd128`.
- Browser adapter overhead remains a separate wall-clock issue. The depth 7/8/9 browser benchmark shows one-shot warm timings are much higher than persistent warm timings even when engine NPS improves, so a direct browser API could still beat more NNUE SIMD for shallow UX.

Next profiling improvements:

- Use a browser DevTools/Chrome trace of the WASI worker if source maps or useful wasm symbols are available.
- Repeat with rotated FENs or fixed movetime to avoid persistent hash reuse when measuring browser compute throughput.
- If adding Reckless-local instrumentation, count calls/time for `Network::evaluate`, `propagate_l1`, accumulator refresh/update, movegen, and UCI/stdout separately.
