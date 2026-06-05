# Browser UCI adapter contract

Last updated: 2026-06-05

This contract applies to browser-hosted UCI-family engines used by `lc0-arena.html`, `lc0-analysis.html`, smoke tests, and benchmark pages. The typed source is `src/lc0/browserUciEngine.ts`; this document records the behavioral expectations that are hard to express in TypeScript.

## Required adapter surface

Each browser UCI adapter should implement `BrowserUciEngine`:

| Method | Contract |
| --- | --- |
| `prewarm(signal?)` | Initialize worker/runtime state and complete the UCI `uci`/`isready` handshake where there is a resident process to warm. It must not start a real search. One-shot-only runtimes may no-op after checking abort/policy. |
| `search(fen, signal?)` | Search one FEN and return a UCI bestmove, or `null` for no legal move. This is the standard name for new code. |
| `bestMove(fen, signal?)` | Backward-compatible alias used by existing arena/analysis wiring. It should have the same behavior as `search`. |
| `analyze(fen, opts?)` | Return sorted MultiPV/info lines when supported. If an engine cannot support MultiPV yet, keep it out of analysis UI or return the single PV with the limitation documented on the engine card. |
| `newGame(signal?)` | Reset per-game/search state and hash where possible, then wait for readiness. Persistent runtimes should send `ucinewgame`/`isready`; one-shot runtimes may clear local info only if every search starts a fresh process. |
| `lastInfo()` | Return defensive copies of the last parsed UCI info/PV lines. |
| `runtimeStatus()` | Return machine-readable runtime metadata for UI diagnostics and benchmark JSON: mode, worker/WASM URL, persistent availability/fallback flags, NNUE URL/load progress where applicable. |
| `runtimeLabel()` | Return short human-readable status text for UI rows. |
| `dispose()` | Reject pending work and release/terminate workers, shared stdin, pending callbacks, and cached runtime state owned by the adapter. |

## Lifecycle rules

1. **Serialize UCI sessions per engine instance.** Concurrent searches against one worker/process must be queued or rejected; current adapters use an exclusive promise tail.
2. **Handshake explicitly.** A resident UCI process must complete `uci` and `isready` before searches. Optimizations may skip repeated `uci` only after `uciok`, but `isready` remains the synchronization barrier.
3. **Reset deliberately.** Benchmarks and game boundaries should call `newGame()` before timed searches when hash reuse would contaminate results.
4. **Abort honestly.** If graceful `stop` is proven for a runtime, use it and resolve with the engine's current bestmove. Otherwise terminate/recreate the worker and surface an `AbortError`; do not pretend the persistent process survived.
5. **Keep diagnostics current.** Runtime mode, persistent fallback/disabled state, asset URLs, and load progress should be visible through `runtimeStatus()` and rendered by the owning UI surface.
6. **Own asset checks outside the core search method.** Variant modules should expose asset HEAD/status helpers so the UI can show missing local blobs before a search fails. The adapter should still include URL metadata in `runtimeStatus()`.
7. **Parse info defensively.** UCI `info` parsing should tolerate missing `multipv`, `nodes`, or `nps`, preserve PV UCI moves, and sort by MultiPV rank.
8. **Dispose idempotently.** Multiple `dispose()` calls must be safe and should leave `lastInfo()` empty.

## Current adapter status

| Adapter | Contract status | Notes |
| --- | --- | --- |
| `StockfishEngine` | Implements the standard method set. | `prewarm()` initializes the JS/WASM worker. `newGame()` sends `ucinewgame` then waits on `isready`. Abort sends UCI `stop`. Stockfish assets are package/public assets rather than dynamic variant downloads. |
| `RecklessEngine` | Implements the standard method set. | Supports WASI one-shot, persistent WASI, and experimental browser API. Persistent prewarm and `newGame()` are implemented; abort terminates/recreates runtimes when graceful stop cannot preserve state. Variant modules own artifact checks. |
| `ViridithasEngine` | Implements the standard method set. | Persistent prewarm is implemented when SAB/isolation is available. One-shot mode has no resident process to warm. Stop/abort still terminates the worker; robust graceful stop remains open. Variant modules own artifact checks. |
| `BerserkEngine` | Not implemented yet. | Must satisfy this contract before arena/analysis selector promotion. Start with one-shot WASI UCI smoke, then decide whether persistent WASI is worth adding. |

## Promotion checklist for a new UCI engine

Before adding a new family to staged selectors:

- implement `BrowserUciEngine` or document every temporary deviation;
- add a variant module with stable keys, labels, URLs, default/fallback logic, and asset checks;
- verify UCI handshake, startpos search, non-startpos search, repeated `newGame()` searches, abort/dispose, and missing-asset UI state;
- capture raw smoke/benchmark artifacts with `runtimeStatus()` metadata;
- update `docs/engine_catalog.md` and `src/lc0/engineCatalog.ts` in the same commit as UI exposure.
