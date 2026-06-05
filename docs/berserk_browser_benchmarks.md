# Berserk browser benchmarks

Berserk is still experimental. These notes track the first browser lifecycle and benchmark runs for the Emscripten worker adapter; generated JS/WASM/data artifacts and raw benchmark JSON remain ignored until the GPL/source-archive distribution policy is explicit.

## 2026-06-05 Emscripten worker lifecycle smoke

Environment: Vite dev server with cross-origin isolation headers, HeadlessChrome 149, `SharedArrayBuffer` available.

Adapter: `BerserkEngine` over `/berserk/berserk-emscripten.js`, `/berserk/berserk-emscripten.wasm`, and `/berserk/berserk-emscripten.data`.

Checks passed via `berserk-smoke.html?depth=1&abortDepth=8`:

- `prewarm()` UCI handshake and `isready` completed.
- `newGame()` before searches completed.
- Start position `go depth 1` returned `bestmove d2d4`.
- Non-startpos FEN returned `bestmove e1g1`.
- Repeated start-position search returned `bestmove d2d4`.
- `analyze(..., multipv: 2, depth: 1)` produced two parsed PVs:
  - PV 1: `e1g1`, score `+1.44`.
  - PV 2: `b1c3`, score `+1.20`.
- Abort of a deeper search rejected with `AbortError` after about 11 ms, then the same adapter object recovered by creating a fresh worker and returning `bestmove e1g1` on the test FEN.
- Missing JS asset produced a visible worker/import failure instead of hanging.

UI lifecycle checks:

- Arena selector smoke: `Berserk Emscripten experimental d1` vs Stockfish on a trivial custom K-v-K FEN completed one game as `1/2-1/2 (insufficientMaterial)`; Berserk runtime reported `Emscripten worker ready · asset ok`.
- Analysis selector smoke: one staged row switched to Berserk, `Lines=2`, test FEN analyzed at depth 1; UI rendered both Berserk PVs (`O-O`, `Nc3`) and runtime reported `Emscripten worker ready · asset ok`.

## 2026-06-05 rotated-FEN benchmark capture

Harness: `reckless-benchmark.html`, extended to include Berserk Emscripten as a resident-worker benchmark variant.

Protocol:

- Variant: `Berserk Emscripten experimental`.
- Mode: `persistent` / resident worker.
- Budget: fixed depth 7, hash 16 MiB, 1 thread.
- Positions: 20-position rotated Ruy Lopez FEN suite used by the Reckless/Viridithas browser benchmark page.
- Repeats: cold pass + 1 warm pass, with `ucinewgame`/hash clear before every timed search.
- Raw local JSON artifact: `artifacts/berserk/berserk-emscripten-depth7-rotated-fen-2026-06-05.json` (ignored).

Aggregate result:

| Metric | Value |
| --- | ---: |
| Raw rows | 40 |
| Summary rows | 20 |
| Cold wall avg | 6.65 ms |
| Cold wall min / max | 1.60 / 22.78 ms |
| Warm wall avg | 7.02 ms |
| Warm wall min / max | 1.76 / 20.51 ms |
| Mean nodes/search | 2,242 |
| Nodes min / max by position | 585 / 6,965 |
| Mean engine-reported NPS | 412,816 |

Caveats:

- Depth 7 is still shallow; wall-clock timings are useful mainly for adapter/lifecycle sanity, not strength or final speed claims.
- Emscripten search is currently synchronous inside the worker. Abort is implemented by worker termination/recreation, not graceful `stop` preservation.
- Results are not yet compared to Stockfish/Reckless/Viridithas under the exact same depth/position budget in this run.
