# Stockfish.js relaxed SIMD experiment

Status: lite single-thread relaxed artifact staged as a feature-detected candidate; do not promote as a default replacement without the larger browser/device benchmark below.

## Promotion benchmark matrix

Toolchains to keep in the Stockfish relaxed SIMD benchmark matrix:

- `emscripten/emsdk:3.1.40` as the first known-good baseline toolchain.
- `emscripten/emsdk:5.0.7` because the final asyncify/import/stack fix works on the 5.x generation too. Prefer Docker for repeatability; local macOS builds need extra Makefile host-flag cleanup.
- `emscripten/emsdk:latest` / current 6.x because `6.0.1` passed smoke/parity and was marginally fastest in the small matrix.

Minimum release-candidate matrix before promotion:

- Lite single-threaded: baseline vs relaxed across all toolchains above.
- Full single-threaded: baseline vs relaxed once full-net embedding/download is reproducible.
- Lite/full pthread builds: only after cross-origin-isolation and helper-worker loading are validated.
- Browsers: Chromium/Chrome, Firefox, Safari where relaxed SIMD support and fallback can be verified.
- Positions: opening, tactical middlegame, quiet middlegame, castling-rights edge cases, en-passant edge cases, promotion/endgame cases.
- Depth/time: fixed-depth parity (`10/12/14` or equivalent) plus fixed-movetime NPS stability runs.
- Gates: UCI smoke (`uci`, `isready`, `go depth 1`), `sameBestmove`, `sameScore`, `samePv`, no console/runtime errors, and speedup retained over the current public baseline.

## Build finding

A Stockfish 18 lite single-threaded WASM can be built with WebAssembly relaxed SIMD dot-product opcodes by patching the upstream `stockfish.js` source from `public/stockfish/stockfish-18.0.7-corresponding-source.tar.gz`. The default reproducible build currently uses `emscripten/emsdk:3.1.40`; Emscripten 5.0.7 and 6.0.1 also pass smoke/parity with the same asyncify/import/stack fixes.

Working local builder:

```sh
./scripts/build_stockfish_relaxed_simd.mjs
# or test a newer Docker toolchain:
EMSDK_DOCKER_IMAGE=emscripten/emsdk:latest ./scripts/build_stockfish_relaxed_simd.mjs
```

Output:

```text
.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-lite-single-relaxed.js
.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-lite-single-relaxed.wasm
```

The committed builder keeps fresh generated artifacts under `.local-dev-artifacts`; promotion copies the lite single-thread candidate to `public/stockfish/stockfish-18-lite-single-relaxed.{js,wasm}` as a separate artifact while preserving `stockfish-18-lite-single.{js,wasm}` for fallback.

## Important build details

- Emscripten 3.1.7 matches upstream `stockfish.js` expectations, but its `wasm_simd128.h` does not provide `wasm_i32x4_relaxed_dot_i8x16_i7x16_add`.
- Emscripten 3.1.40+ provides the relaxed dot intrinsic, but the old Stockfish.js makefile needs patching:
  - add `-mrelaxed-simd` next to `-msimd128`;
  - remove obsolete `-fexperimental-new-pass-manager`;
  - use `--closure 0` for this experiment;
  - keep `-s ASYNCIFY=1` and add `-s ASYNCIFY_IMPORTS=["emscripten_utils_getline_impl"]`;
  - add `-s STACK_SIZE=1048576`.
- Without the larger stack, the relaxed build can emit `Stack overflow detected` or fail later with runtime traps.
- Earlier Docker-latest / Emscripten 6.x attempts failed before the stack/import fix was applied. With the final fix set, Emscripten 6.0.1 passes browser Worker smoke and parity.
- Local Homebrew Emscripten 5.0.7 also passes after the same core fixes, though the upstream Makefile additionally needs Darwin host flags removed (`-arch`, `-mdynamic-no-pic`, `-mmacosx-version-min`) when building outside Linux Docker.

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

Toolchain matrix run, same 5 positions x depths 11 and 13:

```json
{
  "baseline": { "rows": 10, "medianNps": 1667894, "aggregateNps": 1988633 },
  "relaxed3140": { "rows": 10, "medianNps": 1980625, "aggregateNps": 2304865, "speedupAggregate": 1.1590197889706144 },
  "relaxed507": { "rows": 10, "medianNps": 1980625, "aggregateNps": 2290468, "speedupAggregate": 1.1517801424395553 },
  "relaxed601": { "rows": 10, "medianNps": 1980625, "aggregateNps": 2332723, "speedupAggregate": 1.1730284069509054 }
}
```

All three relaxed toolchain builds had `10/10` same bestmove, score, and PV against baseline in that matrix. Emscripten 6.0.1 was slightly fastest in this small run, but the spread between relaxed toolchains was small; choose a promotion toolchain after a larger matrix and browser compatibility pass.

Production artifact checks now include the relaxed lite-single files in `scripts/check_browser_engine_assets.mjs`, the R2 Brotli publish list, and `public/stockfish/stockfish-18.0.7.manifest.json`. The corresponding-source archive includes the relaxed builder, app integration code, feature probe, and this experiment note.

Current app integration uses `supportsWasmRelaxedSimd()` from `src/lc0/wasmFeatures.ts` to select `stockfish-18-lite-single-relaxed.js` only when the tiny relaxed-dot `WebAssembly.validate()` probe passes; otherwise it falls back to `stockfish-18-lite-single.js`. Unit coverage in `tests/lc0_stockfish_engine.test.mjs` patches `WebAssembly.validate` to verify both selection paths and keeps the cross-origin blob worker hash behavior intact.

A public-asset browser Worker smoke on the staged relaxed files passed in Chromium via a local static server:

- worker URL: `/stockfish/stockfish-18-lite-single-relaxed.js#<encoded wasm url>`
- `uci` -> `uciok`
- `isready` -> `readyok`
- `position startpos`, `go depth 1` -> `bestmove d2d4`
- agent-browser console/page-error checks after the smoke returned no console messages and no page errors.

Larger Chromium public-asset validation covered opening, tactical, quiet middlegame, castling-rights, en-passant, promotion, and endgame FENs with fixed depths 7/9 plus fixed movetime 120 ms:

```json
{
  "summary": {
    "baseline": { "rows": 21, "medianNps": 1591700, "aggregateNps": 2013275 },
    "relaxed": { "rows": 21, "medianNps": 1804000, "aggregateNps": 2337552 },
    "depthSpeedupAggregate": 1.1024590335416582,
    "movetimeSpeedupAggregate": 1.1612924466738477,
    "speedupAggregate": 1.161069401845252
  },
  "parity": {
    "depthPairs": 14,
    "depthSameBestmove": 14,
    "depthSameScore": 14,
    "depthSamePv": 14,
    "movetimePairs": 7,
    "movetimeSameBestmove": 7
  }
}
```

The larger run also had no console messages or page errors in agent-browser. Firefox and Safari compatibility remain pending before replacing the default baseline artifact.

Full local reports are ignored under `.local-dev-artifacts/stockfish-relaxed/*report*.json`.
