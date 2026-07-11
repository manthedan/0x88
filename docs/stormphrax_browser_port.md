# Stormphrax browser port

## Pinned release

- Upstream: https://github.com/Ciekce/Stormphrax
- Version/tag: `8.0.0` / `v8.0.0`
- Commit: `582965517ed2032d41a6b4cd6c2e66b1b934e2ad`
- Network: `undertown.nnue`
- Network URL: https://github.com/Ciekce/stormphrax-nets/releases/download/undertown/undertown.nnue
- Network bytes: `55,914,848`
- Network SHA-256: `04d651e078b7c7334709dbd772d40a23c0a5480e93e19521a03020c7d633f2cf`
- License: GPL-3.0-or-later

Stormphrax states that its networks are trained from self-generated data. The 8.0.0 release identifies `undertown` as its release network.

## Browser runtime

The first browser build follows `docs/browser_c_engine_porting.md`:

- Emscripten 6.0.2;
- modular JS factory plus WASM and preloaded `.data` sidecar;
- WebAssembly SIMD through a 128-bit backend mapped to Emscripten SSE intrinsics;
- scalar browser fallback for Stormphrax's threat-input geometry;
- synchronous, single-thread search in a dedicated Web Worker;
- persistent UCI handler with `uci`, `isready`, `ucinewgame`, `MultiPV`, depth, and movetime support;
- abort recovery by terminating and recreating the worker;
- Syzygy disabled by leaving `SyzygyPath` empty.

The upstream network's layout requires no permutation for this 128-bit backend. The build verifies the original network hash before preloading it.

## Build and validation

```sh
npm run stormphrax:build-emscripten
npm run stormphrax:smoke-emscripten
```

The Node smoke covers UCI handshake, readiness, new-game reset, start position, a non-start FEN, info output, and a second readiness barrier. At depth 2 the browser build exactly matched the official Apple M1 8.0.0 binary after equivalent `ucinewgame` resets: startpos `g1f3` (76 nodes, +0.35) and the test FEN `e1g1` (145 nodes, +1.38). The smoke pins both best moves and emitted no stderr.

### Relaxed SIMD promotion

`npm run stormphrax:build-relaxed-simd-emscripten` adds `-mrelaxed-simd` and replaces the NNUE L1 `maddubs`/`madd` sequence with `i32x4.relaxed_dot_i8x16_i7x16_add`. Each vector is checked against the opcode's `[0, 127]` i7 precondition; an out-of-range vector uses the baseline SIMD sequence, preserving exact behavior for every input without relying on a position-specific activation bound. An optional audit build traps on that fallback for coverage experiments.

The artifact contains 12 relaxed dot-product instructions and passes the standard Node smoke. Fixed-depth comparison passed all 20 rotated positions at depths 7, 9, and 11 (60/60) with identical best move, score, nodes, and PV. On Node.js on an Apple M4, aggregate NPS was 648k vs 621k at depth 9 (`1.04x`) and 717k vs 634k at depth 11 (`1.13x`). A Chromium depth-11 run on the same machine produced 540k vs 477k aggregate NPS (`1.13x`) with 20/20 exact browser parity. These measurements are device/runtime-specific.

Relaxed SIMD is the automatic default when the exact relaxed-dot feature probe validates. Browsers without it select the baseline SIMD artifact, and a missing default relaxed asset also falls back to baseline. `stormphraxVariant=relaxed` remains available as an explicit diagnostic override.

Before distribution:

```sh
npm run stormphrax:source-archive
npm run stormphrax:release-manifest
```

The matching source archive and manifest are mandatory for public GPL artifacts.

## Integration status

Stormphrax is an experimental family in Play, Arena, and Analysis. It uses depth as its strength unit and is intentionally single-threaded until a pthread build earns promotion through browser lifecycle and performance testing.

## Standardization assessment

The project has a real standardized **contract and release process**: typed `BrowserUciEngine`, engine cards, variant metadata, pinned builds, checksums, smoke requirements, asset checks, source archives, manifests, and R2 publication rules.

The implementation is not yet a declarative plug-in system. C/C++ adapters and variant modules still repeat substantial boilerplate, and each new family must be threaded manually through Play, Arena, Analysis, diagnostics, deployment allowlists, and documentation. Stormphrax followed the standard process successfully, but its integration confirms that the next useful abstraction is a shared configurable Emscripten UCI worker adapter plus a central family registry consumed by all three UI surfaces and packaging scripts.
