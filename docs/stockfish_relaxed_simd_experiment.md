# Stockfish.js relaxed SIMD experiment

Status: the lite-single relaxed SIMD artifact is promoted as a feature-detected candidate. Full single-threaded, lite pthread, and full pthread relaxed SIMD artifacts remain reproducible local candidates with smoke/parity evidence, but are not public release artifacts yet. Do not promote relaxed SIMD as an unconditional/default replacement without the broader browser/device matrix below.

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
# or explicitly choose a variant:
npm run stockfish:build-relaxed-simd:lite-single
npm run stockfish:build-relaxed-simd:single
npm run stockfish:build-relaxed-simd:lite-threaded
npm run stockfish:build-relaxed-simd:threaded
# or test a newer Docker toolchain:
EMSDK_DOCKER_IMAGE=emscripten/emsdk:latest ./scripts/build_stockfish_relaxed_simd.mjs --variant lite-single
```

Outputs stay under `.local-dev-artifacts/stockfish-relaxed/repro/dist/` and use these candidate names:

```text
stockfish-18-lite-single-relaxed.{js,wasm}
stockfish-18-single-relaxed.{js,wasm}
stockfish-18-lite-relaxed.{js,wasm}
stockfish-18-relaxed.{js,wasm}
```

The committed builder keeps fresh generated artifacts under `.local-dev-artifacts`. The public promotion currently copies only `stockfish-18-lite-single-relaxed.{js,wasm}` to `public/stockfish/` as a separate artifact while preserving the baseline artifact for fallback; full single-threaded, lite pthread, and full pthread relaxed outputs remain local candidates until deeper benchmark and compatibility gates pass.

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
- Full-network variants require `nn-c288c895ea92.nnue` and `nn-37f18f62d772.nnue`. The builder now parses the selected upstream net header, copies from `STOCKFISH_NNUE_DIR`/`STOCKFISH_NNUE_PATH` when available, and otherwise tries the upstream Stockfish network URLs.
- On memory-constrained Docker hosts, use `STOCKFISH_MAKE_JOBS=1` or `2` for full-network variants. `STOCKFISH_RELAXED_DISABLE_LTO=1` removes upstream `-flto=full` for exploratory full/pthread builds when the linker or compiler is killed.
- Apple Silicon Colima defaults (`2` CPUs, `2GiB` memory) were too small for full-network variants and produced Docker exit `137`. Increasing Colima to `6` CPUs, `12GiB` memory, and `80GiB` disk, then using native `linux/arm64` `emscripten/emsdk:latest` (`emcc 6.0.1`) allowed the full single-threaded relaxed artifact to build:

```sh
colima stop
colima start --cpu 6 --memory 12 --disk 80 --runtime docker --arch aarch64

STOCKFISH_NNUE_DIR=.local-dev-artifacts/stockfish-relaxed/nets \
STOCKFISH_MAKE_JOBS=2 \
STOCKFISH_RELAXED_DISABLE_LTO=1 \
EMSDK_DOCKER_IMAGE=emscripten/emsdk:latest \
DOCKER_PLATFORM=linux/arm64 \
node scripts/build_stockfish_relaxed_simd.mjs --variant single
```

## Large variant porting status

- `lite-single`: builds, smokes, benchmarks, and is staged as the feature-detected candidate.
- `single`: builds locally with `emscripten/emsdk:latest`/`emcc 6.0.1` on native `linux/arm64` after increasing Colima memory. Local candidate output: `.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-single-relaxed.{js,wasm}` (`108MiB` wasm, `simd.275:140`). Isolated Chromium smoke reached `uciok`, `readyok`, and `bestmove e2e4`; promotion validation against `stockfish-18-single` had `8/8` same bestmove, score, and PV. Not promoted yet.
- `lite-threaded`: builds locally as `.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-lite-relaxed.{js,wasm}` with relaxed opcodes (`simd.275:70`). A patched helper-worker path now avoids the obsolete upstream `worker-extern-post.js` shim and initializes the embedded Emscripten pthread helper directly. Isolated Chromium smoke reached `uciok`, `readyok`, and `bestmove d2d4` with no helper-worker `startWorker`/`createObjectURL` errors; promotion validation against `stockfish-18-lite` had `8/8` same bestmove, score, and PV. Not promoted yet.
- `threaded`: builds locally with `emscripten/emsdk:3.1.40` under `linux/amd64` after increasing Colima memory. Local candidate output: `.local-dev-artifacts/stockfish-relaxed/repro/dist/stockfish-18-relaxed.{js,wasm}` (`108MiB` wasm, `simd.275:140`). Isolated Chromium smoke reached `uciok`, `readyok`, and `bestmove e2e4` with no helper-worker bootstrap errors; promotion validation against `stockfish-18` had `8/8` same bestmove, score, and PV. Not promoted yet. `emscripten/emsdk:latest`/`6.0.1` compiles the full pthread object/link step only after adding `-pthread`, but its newer output no longer emits the `stockfish.worker.js` file that upstream `stockfish.js` `build.js` expects; the 6.x pthread packaging path needs a dedicated follow-up patch before it can replace 3.1.40 for pthread variants.

## NNUE dot-product patch

The relaxed path replaces Stockfish's SSSE3 `m128_add_dpbusd_epi32` helper with:

```cpp
wasm_i32x4_relaxed_dot_i8x16_i7x16_add(weights, act_i7, sum)
```

The operands are intentionally swapped relative to the SSSE3 helper because Stockfish NNUE activations are clipped to `0..127`, making them safe as the relaxed dot-product `i7x16` operand; weights remain the signed operand.

## Pthread shared-memory status message

The pthread variants can print:

```text
Network replica 1: Local memory. Shared memory not supported by the OS. Local allocation fallback.
```

This message comes from Stockfish's `SystemWideSharedConstant` network-replica allocator (`src/shm.h` / `src/engine.cpp`), not from browser `SharedArrayBuffer` or Emscripten pthread support. In the browser/WASM build, Stockfish's OS-level shared-memory backend is unavailable, so each network replica falls back to local memory. Cross-origin isolation and browser shared linear memory are still required for pthread helper workers; the promotion smoke validates `crossOriginIsolated=true`, `SharedArrayBuffer=function`, `setoption name Threads value 2`, and successful searches. The local-memory fallback remains documented as expected browser behavior rather than a blocker.

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

Production artifact checks now include only the promoted relaxed lite-single files in `scripts/check_browser_engine_assets.mjs`, the R2 Brotli publish list, and `public/stockfish/stockfish-18.0.7.manifest.json`. The corresponding-source archive includes the relaxed builder, app integration code, feature probe, and this experiment note; optional full/threaded relaxed candidate build commands remain documented but are not release artifacts.

Current app integration uses `supportsWasmRelaxedSimd()` from `src/lc0/wasmFeatures.ts` to select the lite-single relaxed artifact only when the tiny relaxed-dot `WebAssembly.validate()` probe passes; otherwise it falls back to the baseline lite-single Stockfish.js artifact. Full and pthread flavors continue to resolve to baseline public artifacts. Pthread flavors remain same-origin only because helper workers derive their URLs from `self.location`. Unit coverage in `tests/lc0_stockfish_engine.test.mjs` patches `WebAssembly.validate` to verify both lite-single selection paths and keeps the cross-origin blob worker hash behavior intact.

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

The larger run also had no console messages or page errors in agent-browser.

Promotion-readiness validation for the additional unpromoted candidate artifacts is recorded in `.local-dev-artifacts/stockfish-relaxed/browser_promotion_public_validation_report.json`. Chromium ran under the isolated static server with `crossOriginIsolated=true` and `SharedArrayBuffer=function`:

```json
{
  "single": { "pairs": 8, "sameBestmove": 8, "sameScore": 8, "samePv": 8, "speedupAggregate": 0.9967 },
  "liteThreaded": { "pairs": 8, "sameBestmove": 8, "sameScore": 8, "samePv": 8, "speedupAggregate": 0.9962 },
  "fullThreaded": { "pairs": 8, "sameBestmove": 8, "sameScore": 8, "samePv": 8, "speedupAggregate": 0.9986 }
}
```

A separate `Threads=2` smoke in the same run reached bestmoves for both baseline and relaxed lite/full pthread artifacts. The shallow depth-5/7 NPS numbers above are dominated by full-NNUE load/warmup and browser timing overhead, so they are only a promotion smoke; use the earlier larger lite-single depth/movetime matrix for public speedup claims and run a deeper full-net matrix before claiming full-net speedups.

A follow-up Yukon Linux matrix copied the promoted lite-single plus unpromoted candidate artifacts to `~/stockfish-relaxed-matrix` and served them with COOP/COEP headers. Reports are mirrored locally under `.local-dev-artifacts/stockfish-relaxed/yukon-matrix/reports/`. Firefox 152 and Playwright Chromium 148 both reported `crossOriginIsolated=true`, `SharedArrayBuffer=function`, and `relaxedProbe=true`. Each browser validated lite-single, full single, lite pthread, and full pthread relaxed artifacts against the corresponding baseline with `8/8` same bestmove, score, and PV. Threaded scaling smoke covered `Threads=2` and `Threads=4` for lite/full pthread baseline and relaxed artifacts. Yukon Firefox aggregate speedups were approximately neutral in shallow depth-5/7 parity runs (`liteSingle 1.0000`, `single 0.9985`, `liteThreaded 1.0000`, `fullThreaded 1.0010`), with median-NPS ratios ranging from `0.9282` to `1.1242`. Yukon Chromium aggregate speedups were also approximately neutral (`liteSingle 1.0000`, `single 1.0035`, `liteThreaded 0.9965`, `fullThreaded 1.0005`), with noisier median-NPS ratios from `0.7587` to `1.1257`. Treat these as cross-browser compatibility and pthread smoke evidence for candidate builds, not promotion or speedup evidence. Safari compatibility remains pending before replacing any default baseline artifact.

Full local reports are ignored under `.local-dev-artifacts/stockfish-relaxed/*report*.json`.
