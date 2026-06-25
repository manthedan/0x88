# Stockfish.js relaxed SIMD experiment

Status: experimental; do not promote without a larger browser/device benchmark.

## Build finding

A Stockfish 18 lite single-threaded WASM can be built with WebAssembly relaxed SIMD dot-product opcodes by patching the upstream `stockfish.js` source from `public/stockfish/stockfish-18.0.7-corresponding-source.tar.gz` and building with `emscripten/emsdk:3.1.40`.

Working local builder:

```sh
./scripts/build_stockfish_relaxed_simd.mjs
```

Output:

```text
.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-lite-single-relaxed.js
.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-lite-single-relaxed.wasm
```

The committed builder keeps the generated artifacts under `.local-dev-artifacts` intentionally.

## Important build details

- Emscripten 3.1.7 matches upstream `stockfish.js` expectations, but its `wasm_simd128.h` does not provide `wasm_i32x4_relaxed_dot_i8x16_i7x16_add`.
- Emscripten 3.1.40 provides the relaxed dot intrinsic, but the old Stockfish.js makefile needs patching:
  - add `-mrelaxed-simd` next to `-msimd128`;
  - remove obsolete `-fexperimental-new-pass-manager`;
  - use `--closure 0` for this experiment;
  - keep `-s ASYNCIFY=1` and add `-s ASYNCIFY_IMPORTS=["emscripten_utils_getline_impl"]`;
  - add `-s STACK_SIZE=1048576`.
- Without the larger stack, the 3.1.40 relaxed build can emit `Stack overflow detected` or fail later with runtime traps.
- A Docker-latest / Emscripten 6.0.0 build and a 3.1.40 build without the stack increase both failed runtime smoke tests, even when the relaxed opcodes were present.

## NNUE dot-product patch

The relaxed path replaces Stockfish's SSSE3 `m128_add_dpbusd_epi32` helper with:

```cpp
wasm_i32x4_relaxed_dot_i8x16_i7x16_add(weights, act_i7, sum)
```

The operands are intentionally swapped relative to the SSSE3 helper because Stockfish NNUE activations are clipped to `0..127`, making them safe as the relaxed dot-product `i7x16` operand; weights remain the signed operand.

## Validation evidence

Browser Worker smoke test of the reproducible artifact passed:

- `uci` -> `uciok`
- `isready` -> `readyok`
- `position startpos`, `go depth 1` -> `bestmove d2d4`

SIMD inspector output for the reproducible artifact:

```text
simdOpcodeCount=3877
simd extended:70
simd.275:70
```

Browser comparison against the committed public baseline (`public/stockfish/stockfish-18-lite-single.{js,wasm}`), 5 positions x depths 11 and 13:

```json
{
  "publicBaseline": { "rows": 10, "medianNps": 1760555, "aggregateNps": 2034983 },
  "relaxed3140Stack": { "rows": 10, "medianNps": 2112666, "aggregateNps": 2393629 },
  "speedupAggregate": 1.176240292916452,
  "speedupMedian": 1.2
}
```

Parity for that run:

```json
{ "pairs": 10, "sameBestmove": 10, "sameScore": 10, "samePv": 10 }
```

Full local reports are ignored under `.local-dev-artifacts/stockfish-relaxed/*report*.json`.
