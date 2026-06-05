# PlentyChess browser port notes

Last updated: 2026-06-05

## Status

PlentyChess has a first Emscripten proof-build, Node UCI smoke, reusable browser worker adapter, and browser lifecycle smoke. It is **not** integrated into arena/analysis selectors yet.

Current local commands:

```sh
npm run plentychess:build-emscripten
npm run plentychess:smoke-emscripten
```

Generated artifacts are intentionally ignored under `public/plentychess/`.

Browser worker smoke page:

```text
/plentychess-smoke.html
```

## Source and assets

- Upstream repo: `https://github.com/Yoshie2000/PlentyChess.git`
- Browser proof-build pin: `58d8ba2505ae2b49f48dd410d214a457d15c12c6`
- Reported version at that pin: `7.0.66`
- Network id: `0134-2r24-s0`
- Raw network URL: `https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/0134-2r24-s0.bin`
- Raw network sha256: `550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061`
- Processed network sha256 after upstream `tools/process_net false`: `691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4`

## Build shape

The proof-build follows `docs/browser_c_engine_porting.md`:

- Emscripten modular JS output.
- Single-thread synchronous search path via `PLENTY_SYNC_SEARCH`.
- Exported browser ABI:
  - `_main`
  - `_command`
  - `_isReady`
  - `_isSearching`
- `USE_PTHREADS=0`.
- Fathom compiled with `TB_NO_THREADS`; browser tablebase files are not shipped.
- Processed NNUE is preloaded as `processed.bin` in `.data` because upstream `incbin` assembler directives do not work for wasm.
- First build uses wasm SIMD lowering through Emscripten's x86/SSE-compatible path: `-DARCH_X86 -msimd128 -mssse3`.

## Generated artifact snapshot

From the first successful local build:

| Artifact | Size |
| --- | ---: |
| `public/plentychess/plentychess-emscripten.js` | 70,886 bytes |
| `public/plentychess/plentychess-emscripten.wasm` | 389,983 bytes |
| `public/plentychess/plentychess-emscripten.data` | 63,023,936 bytes |

## Smoke results

### Node

`npm run plentychess:smoke-emscripten` at depth 1 passed:

- `uci` -> `uciok`
- `isready` -> `readyok`
- `ucinewgame`
- `position startpos`, `go depth 1` -> `bestmove c2c4`
- test FEN, `go depth 1` -> `bestmove e1g1`
- post-search `isready` -> `readyok`

Representative final info line:

```text
info depth 1 seldepth 3 score cp 143 multipv 1 nodes 37 tbhits 0 time 1 nps 37000 hashfull 0 pv e1g1
```

### Browser worker lifecycle

`plentychess-smoke.html?depth=1&abortDepth=16` passed locally with:

- startpos: `c2c4`
- non-startpos FEN: `e1g1`
- repeated startpos search: `c2c4`
- MultiPV depth 1: `e1g1`, `b1c3`
- abort: `AbortError`, recovery returned `e1g1`
- missing asset failure surfaced:
  - `Failed to execute 'importScripts' on 'WorkerGlobalScope': The script at 'http://127.0.0.1:5173/plentychess/missing-plentychess-...js' failed to load.`

The smoke page covers:

- prewarm/`uci`/`isready`
- `ucinewgame`
- startpos search
- non-startpos FEN search
- repeated search after `ucinewgame`
- MultiPV depth-1 analysis
- abort/recovery by terminating and recreating the synchronous Emscripten worker
- missing JS asset failure surfacing

## Current limitations

- No arena/analysis selector integration yet.
- Abort uses the same conservative terminate/recreate strategy as other synchronous worker engines.
- GPL-3.0 distribution/corresponding-source policy is required before distributing generated artifacts.
- The `.data` file is large because the processed NNUE is preloaded externally rather than embedded with upstream `incbin`.

## First rotated-FEN benchmark

The shared browser UCI benchmark harness (`reckless-benchmark.html`) now includes a `PlentyChess Emscripten experimental` checkbox. A depth-7, 20-position rotated-FEN run with cold + 1 warm pass completed in persistent worker mode.

Raw ignored artifact:

- `artifacts/plentychess/plentychess-emscripten-depth7-rotated-fen-2026-06-05.json`

Summary:

- rows: 40 raw, 20 summary
- warm average/search: ~9.35 ms
- warm min/max: ~2.37 / 15.14 ms
- mean nodes/search: ~4,068
- mean engine-reported NPS: ~718k

This puts PlentyChess Emscripten roughly in the same shallow-browser-NPS band as full Stockfish single-threaded (~746k) and Reckless SIMD (~794k), with a much larger `.data` sidecar than Berserk.

## Next gates before UI integration

1. Decide whether the ~63 MB `.data` artifact is acceptable for an experimental selector.
2. If yes, add experimental staged selector metadata and UI wiring.
3. Keep generated artifacts ignored and gated by GPL corresponding-source policy.
