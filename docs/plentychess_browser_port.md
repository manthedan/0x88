# PlentyChess browser port notes

Last updated: 2026-06-05

## Status

PlentyChess has a first Emscripten proof-build and Node UCI smoke. It is **not** integrated into arena/analysis selectors yet.

Current local commands:

```sh
npm run plentychess:build-emscripten
npm run plentychess:smoke-emscripten
```

Generated artifacts are intentionally ignored under `public/plentychess/`.

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

## Node smoke result

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

## Current limitations

- No browser worker adapter yet.
- No lifecycle smoke yet for repeated search, MultiPV, abort/recovery, or missing assets.
- No arena/analysis selector integration yet.
- Abort would currently need the same conservative terminate/recreate strategy used for other synchronous worker engines.
- GPL-3.0 distribution/corresponding-source policy is required before distributing generated artifacts.
- The `.data` file is large because the processed NNUE is preloaded externally rather than embedded with upstream `incbin`.

## Next gates before UI integration

1. Add a reusable `PlentyChessEngine` worker adapter implementing `BrowserUciEngine`.
2. Add `plentychess-smoke.html` or extend a smoke harness for browser lifecycle checks.
3. Verify MultiPV parsing and repeated `ucinewgame` behavior.
4. Verify missing `.js`/`.wasm`/`.data` failures are visible.
5. Run a small rotated-FEN benchmark against Berserk/Reckless/Stockfish.
6. Only then add an experimental staged selector variant.
