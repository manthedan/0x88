# Hidden-tab pause and Play CPU residency (October 2026)

## Decision

Ship both lifecycle changes.

Analysis pauses an in-flight search, the Maia live panel, and CPU-asset prefetch while the document is hidden, and restarts that same position when the tab is shown. Arena is left running. A Play move is left running.

Play keeps at most one inactive CPU engine. The engine in use is never disposed. Stormphrax is still disposed as soon as the player leaves it. The cap cleared both gates on the built app: keeping every warmed CPU engine used 729.20 MiB more than keeping only the latest CPU engine plus Lc0 Small, and switching between two engines stayed on the warm path (median 0.6 ms after the cap, median 1 ms before it). A first cut applied the cap only inside `ctxCpuEngineFor`, so choosing Lc0 or Maia left the CPU engines resident. The select handler now evicts as well. After that fix, Lite plus full is 451.87 MiB and the following Lc0 Small page is 425.00 and 425.01 MiB (Lite is gone). Warm Lite against full stays at median 0.6 ms.

## Hidden tab

`src/` had no `visibilitychange` handling before this change. Analysis has no `go infinite`. A live search is still unbounded from the player's point of view: it posts progress until a visit or depth budget, and auto-analysis starts another one after each move. That is the work that now pauses.

The state machine is `src/lc0/hiddenTabPause.ts`. Hiding pauses only work that is active. Showing resumes it in the order analysis, Maia, prefetch. A request made while hidden is remembered and does not start. Stop, or a navigation that would not auto-analyze, cancels that memory so the old position does not restart. The page wiring lives in `src/lc0/analysisBrowser.ts`. The helper plus the call sites and the worker-progress counter are larger than the 50–80 line sketch, because the three work kinds, the stale-progress guard, and the tests do not fit in the handler alone. It is still one listener and one pure transition, not a scheduler.

Resume calls `analyzeCurrent()` on `tree.current.fen`. Cancel drops the Lc0 tree, so the line restarts cleanly rather than continuing the partial search. A search started with Analyze is remembered as forced and resumed with `{ force: true }`, so an older cache entry is not treated as the finished line. Any other resume does not force rows that already finished. Navigation clears the forced bit, because the position just entered is not that run. A late `searchProgress` cannot replace the paused sentence: the publish path returns while the tab is hidden. The lines list is empty while paused because the in-progress render is cleared with the search. That is the idle copy, not a stale PV.

A queued Lc0 reload or unused-engine dispose is copied off the run flags before the pause cancels the search, and put back only on show. The hide call does not restore them: the aborted run's `finally` would see the restored flags and start the reload while the tab is still hidden. The aborted run's `finally` would otherwise start that reload while the tab is still hidden. `reloadLc0ForAnalysis` already restarts the search with `force: true`. The big-net worker receives the run `AbortSignal`. `Bt4WorkerSearcher.search` already returns cancelled when that signal is aborted before or after `init()`, so a hide during the first big-net load does not let the search start when init finishes. The small-net worker queue runs one search at a time and applies cancel immediately, so a cancel for a search that is still queued is remembered and that search posts a cancelled result instead of running when the queue drains. Stop while hidden drops the paused search and keeps a queued Lc0 reload or unused-engine dispose. Showing the tab then applies that maintenance without starting a new search, which is what a visible Stop already does. A reload that was not going to start a search is not recorded as analysis, so hiding during that reload does not resume a search. A search queued behind that reload, including Analyze while the tab is already hidden, is resumed, and a forced request stays forced. Navigation that cancels a hidden search keeps the same queued maintenance. A run that finishes while the tab is hidden, including Stop that preserved a reload and then a hide before the worker returns, stashes that follow-up and does not start it. Stop's follow-up has no new search, so showing the tab applies only the reload or dispose.

Left running, on purpose:

- Arena. Tournaments are started to run in a background tab.
- Play. Every family in `ctxRequestEngineMove` uses a finite depth or visit budget, and the ready check is one depth-1 search or one visit. The reply is what the player expects to find. A hidden Play page has no visibility listener.
- Game review. It is a finite walk the player asked for.
- The initial backend model download, and the one-shot LQO preload after a book move. Neither has a resume token that would restart cleanly, and both are attached to opening the page or to a book reply.

Built app, `http://127.0.0.1:5182`, 11 October 2026, load about 2.7–3.1. The check dispatched a real `visibilitychange` after overriding `Document.prototype.visibilityState` (`mode: "event"` in `/tmp/hidden-tab-verify.json`). An Lc0 search at 100000 visits had posted 3 worker progress messages. Those stayed at 3 through a 500 ms burst and the following 1600 ms hidden window. The status was `Analysis paused while this tab is hidden.` The FEN stayed the start position and `searchActive` was false. On show, the same FEN restarted (`Analyzing (Lc0, 3 lines)…`, then `Lc0: 67/100000 visits`, best `d2d4`) and the worker count moved to 4. Screenshots: `/tmp/hidden-tab-shots/analysis-paused.png`, `/tmp/hidden-tab-shots/analysis-resumed.png`.

The same run asked Stockfish Lite for a reply to `e4` while `visibilityState` was `hidden`. The move list became `e4`, `c5` and the status returned to `Your move` with the document still hidden. Screenshot: `/tmp/hidden-tab-shots/play-replied-while-hidden.png`.

The Maia panel and prefetch are covered by the state machine and its unit tests. They were not exercised in the browser: `/models/maia3/maia3_simplified.onnx` 404s here, so the panel never starts, and the Lc0 row used for the search has no CPU asset URL to prefetch.

## Play CPU residency

`cpuEnginePromises` in `src/lc0/playBrowser.ts` used to keep every CPU engine that had loaded. Only Stormphrax was disposed on switch-away. The cap is `inactiveCpuEnginesToEvict` in `src/lc0/cpuEngineResidency.ts`. `retainCpuEngine` moves the engine just requested to the newest slot and disposes older inactive ids past one. The engine-select handler calls the same eviction after the Stormphrax switch-away dispose, including when the new id is Lc0 or Maia and never reaches `ctxCpuEngineFor`. The id in use is omitted either way. A failed load deletes the map entry only when it is still that promise, so an eviction cannot drop a replacement. The Stormphrax switch-away dispose is unchanged and runs before the new engine is retained.

### How it was measured

Built product client, isolated static server on port 5182 (`crossOriginIsolated === true`). `performance.measureUserAgentSpecificMemory` after each engine reported ready (`#engineNote` contains `is ready`, which is the depth-1 or 1-visit ready check). Three reps, interleaved as all-engines page, single-engine page, then switch-back latency. Fresh page per section. Bytes below are what that API returned after it forced a GC. They are not OS RSS, and a wasm file size is not subtracted from them.

The bench mutex was held the whole time by pid 36231 (`bench_resumable_loader_browser.mjs` on the resumable-loader worktree, since 15:08). At 16:02 that node had about 12 s of CPU, its Chrome tree about 2.6 s, it was listening on port 5185 with no connected client, and it had written no output. Its gate-off interruption loop has no timeout, and a hung `page.evaluate` after the 5-minute ready timeout never writes the diagnostic. That process was not killed. The reps ran beside it while it stayed idle (node CPU moved from about 12 s to 18 s across the hour, with no output file). When a different live runner owns the lock, the driver stops. Load during the reps was mostly 0.8–4.8; one earlier sample while waiting was 11.4. The memory totals agreed to a few tens of kilobytes across reps, so that noise did not move the result. Date: 11 October 2026. Machine uptime about 26 days.

The neural-path remeasure at 17:14–17:20 used the same exception. The lock was then held by pid 87095, the same script, started at 16:55. Its jsonl was still the 15814-byte file from 16:55, and the node process moved from about 5.5 s to 6.5 s of CPU across the reps. It was not killed. Load during those reps was 1.0–1.7.

Maia3's onnx 404s, so the baseline is the failed-Maia page (about 3.62 MiB). Reckless's default wasm `/reckless/reckless.wasm` is not in this tree (HTTP 404). Those engines were recorded as skips, not as failed evictions. `lc0-bt4` is disabled in the catalog.

### All engines, then Lc0 Small

| Rep | sf-lite | sf-full | viridithas | berserk | plentychess | stormphrax | then lc0-small |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 132.41 | 451.86 | 452.34 | 733.06 | 1040.50 | 1606.36 | 1142.18 |
| 2 | 132.41 | 451.86 | 452.33 | 733.06 | 1040.50 | 1606.38 | 1142.16 |
| 3 | 132.41 | 451.86 | 452.34 | 733.06 | 1040.50 | 1606.36 | 1142.15 |

Figures are MiB (`bytes / 1048576`) of the whole page. Within one page, sf-full adds about 319.5 MiB on top of sf-lite, berserk about 280.7, plentychess about 307.4, and stormphrax about 565.9. Viridithas becomes ready (about 0.6 s) but the API moves only about 0.5 MiB; its 53 MiB wasm is not visible as a separate reservation here. Leaving Stormphrax for Lc0 Small drops the page by about 464 MiB, which is the existing dispose, not the new cap. The neural net itself is the roughly 102 MiB added above the pre-Stormphrax total.

### One CPU engine, then Lc0 Small

PlentyChess, then Lc0 Small, on a fresh page: final bytes 412.96, 412.96, and 412.95 MiB. Subtracting each page's baseline, the all-engines page held 729.21, 729.19, and 729.20 MiB more than the one-engine page. Median 729.20 MiB. That is the savings of not keeping sf-lite, sf-full, viridithas, and berserk once PlentyChess is the engine you would retain. It clears the 64 MiB gate.

### Switch-back, before the cap

Common case: warm Stockfish Lite against Stockfish full, both left resident. 24 switches after the cold load. 23 were 1 ms and one was 2 ms (integer `performance.now` rounding). Median 1 ms. Cold first load of sf-full was 403, 348, and 336 ms (median 348). Stormphrax, which this page already disposes on switch-away, came back in 43–46 ms (median 45), against a 1 ms warm Stockfish switch. That is the cost of throwing an engine away, and it is why the cap must not throw away the engine you just left.

### After the cap

Same warm pair, now timed to 0.1 ms. 24 switches: median 0.6 ms, spread 0.5–2.2 ms. Against the pre-cap median of 1 ms that is not a regression; the spreads overlap, and a cold sf-full load is still about 340–440 ms. Selecting Stormphrax from that pair evicts Lite (it is the older inactive engine). The next Lite ready check is then a cold load, 76.1, 74.6, and 68.3 ms, and the following Lite switch is warm again (1.9, 2.1, 0.8 ms) because Stormphrax's own dispose left Lite as the one inactive engine.

A separate three-engine check (Lite, full, Berserk) shows the drop. After Berserk the page was 603.83 MiB on all three reps. The pre-cap sweep, with Lite still resident at the same point, was 733.06 MiB. The difference is 129.20 MiB. Lite's own step on the pre-cap page was 128.79 MiB above the 3.62 MiB baseline, so the drop is that engine and not a GC fluke. Full stayed: the total after full and before Berserk was still 451.86 MiB, the same as the pre-cap page.

### Neural selection

The CPU-to-CPU check above does not cover a switch to Lc0. That selection never called `ctxCpuEngineFor`, so the first cut left Lite and full resident next to the net. The select handler now evicts. Fresh page, Lite, then full, then Lc0 Small:

| Rep | sf-lite | sf-full | then lc0-small |
| --- | ---: | ---: | ---: |
| 1 | 132.41 | 451.87 | 425.01 |
| 2 | 132.41 | 451.87 | 425.00 |

Figures are MiB of the whole page. Lc0 Small lands 26.86 MiB under the Lite+full total, and the two reps differ by 3 KiB. Lite's step is 128.79 MiB and the small net is about 102 MiB, so 451.87 − 128.79 + 102 is 425. Keeping Lite would have left the page near 554 MiB.

Warm Lite against full after this handler change, two pages. A third page failed to mount (`crossOriginIsolated` false, empty engine list), the same flake as one earlier rep, and was retried rather than counted. 16 switches: median 0.6 ms, spread 0.6–2.0 ms. Cold sf-full on those pages was 401 ms and 394 ms. That is the same warm band as the cap measurement above. Selecting Stormphrax from the pair still evicts Lite (cold again at 75.2 ms and 70.0 ms); the next Lite switch is warm (2.1 ms and 1.0 ms) because Stormphrax's own dispose left Lite as the one inactive engine.

## What this does not change

Arena residency, analysis TT size, and Play's 64 MiB hash are unchanged. Unmount still disposes only Stormphrax; the page is going away. Play still does not pause for a hidden tab.
