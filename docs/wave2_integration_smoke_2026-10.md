# Wave 2 integration smoke (October 2026)

## Decision

Ship one analysis pause fix. The combined local main otherwise matches the wave-2 docs on the built app.

## What failed

Hiding the Analysis tab aborted the in-flight search and left the previous PV on screen. The aborted run's `finally` then disabled Stop. With Stop dead, the suspended search could not be dropped, and showing the tab started it again.

Built app, before the fix, a 2,500-visit Lc0 search: worker progress froze at 15, the status was `Analysis paused while this tab is hidden.`, and the line list still showed `d2d4`, `g1f3`, and `e2e4`.

## Fix

`hiddenAnalysisStopEnabled` keeps Stop enabled while analysis is only suspended. The pause path deletes the current FEN from `lineCache` and re-renders the lines, the eval bar, and the arrows, so the page shows the idle copy. Stop while hidden still clears the resume, and showing the tab does not start a search.

## Built-app check after the fix

Product client on `http://127.0.0.1:5182`, `crossOriginIsolated`, 10 cores. Visibility was emulated by overriding `Document.prototype.visibilityState` and dispatching `visibilitychange`.

- A 20,000-visit Lc0 search paused with worker progress frozen at 1, zero PV rows, the idle placeholder, and Stop still enabled. Screenshot: `integration-smoke-shots/analysis-paused-20k.png`.
- Analyze while that pause was deferred. Showing the tab started a new search at `10/20000` visits.
- Stop while hidden set `Analysis stopped.` and cleared the suspended search. Showing the tab left progress at 2 and did not restart. Screenshot: `integration-smoke-shots/analysis-stop-while-hidden-20k.png`.
- Analyze after that stopped run, while hidden, resumed a forced search (`44/20000` visits). An 80-visit budget had looked frozen only because the warm search finished before the first poll.
- `e2e4` while a search was paused updated the FEN, left the line list empty, and resumed `Analyzing (Lc0, 3 lines)…` on that position.
- A queued runtime change to `hybrid-ort-heads` did not search while hidden. Showing the tab spawned the hybrid worker and stopped at `Ready. Drag a move…`.
- Maia3 loaded. After `e2e4` the grid led with `e5`. Hiding and playing `e7e5` kept that grid and set `suspended` to `maia`. Showing the tab replaced it with `Nf3` / `Nc3` / `Bc4` for the new position. Screenshots: `integration-smoke-shots/analysis-maia-paused.png`, `integration-smoke-shots/analysis-maia-resumed.png`. No `Maia3 load failed`.
- Viridithas Relaxed SIMD, after `Analysis complete` at depth 8: one `viridithas-wasi-persistent` worker (id 2). Ten Ruy Lopez plies at 40 ms, then depth 16, kept id 2 and the final FEN `r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6`. Hide and show paused and resumed that same worker and FEN. A cold fast burst can still replace the worker: the 100 ms abort watchdog terminates a process that has not acknowledged `stop`. That is the behavior in `docs/cpu_coop_cancel_2026-10.md`, and the settled line still belonged to the displayed position.
- Play: Stockfish Lite replied `e4 e5`, Berserk `e4 c5`, Viridithas `e4 c5`, Lc0 Small `e4 e5`. Lc0 levels 0, 1, and 2 (8, 32, and 100 visits) each replied `e4 e5`. A hidden tab still received `e4 e5`. Back showed `Viewing 1. e4`; forward returned to `Your move`. No console errors. This section was not repeated after the analysis-only fix.
- Arena, legal mate `7k/5Q2/6K1/8/8/8/8/8 w - - 0 1`, fixed depth 4, Stockfish Lite / Berserk / PlentyChess. Auto (`2 game(s) at once`) and concurrency 1 both finished `Match done (3 games).` Standings matched: Lite +1200 2-0-0, Berserk +0 1-0-1, PlentyChess −1200 0-0-2. PGNs matched (676 bytes), first game `1. Qg7# 1-0`. Stopping a live Auto match (`Game 1/4`, four private workers, `K=2`) left `Stopped after 0 game(s).` and no live workers. Lc0 Small 16 visits vs Berserk played `Game 1/1` with no `active (K=` text. The Berserk blob that remains after that mixed stop is the shared page engine (`prewarm`), not a private tournament worker.

Console noise that was ignored: favicon 404 and the Node `NO_COLOR` / `FORCE_COLOR` warning.

## Rejected

An opening FEN with the side not to move in check (`6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1`) makes the Stockfish worker throw `ExitStatus` from `quit_` and the match reports `Uncaught [object Object]`. The legal mate above plays. No vendored Stockfish edit. `parseFenOpening` still does not reject that FEN.

The cold Viridithas worker replacement is the existing 100 ms watchdog, not a new per-position engine.
