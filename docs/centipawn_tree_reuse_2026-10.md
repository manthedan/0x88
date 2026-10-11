# Centipawn retained-tree / Play yielding pilot — 2026-10

## Decision: park; no product change

Base: `2ea6fa1`, branch `perf/centipawn-tree-reuse`. The strict promotion gates were **not met**. Retain the research probe, evidence and history/budget guardrail tests only. `src/lc0/playBrowser.ts`, `src/lc0/arenaBrowser.ts` and `src/search/puct.ts` are unchanged, including `scoreEdge`. There is no new default, product flag, evaluator-generation cache or lifecycle implementation to merge.

- Play-like two-ply continuation: **8.84% median paired wall saving**, below the required 10%; range 1.99–13.52%. Useful inheritance was common, but not a repeatable enough wall win.
- Same-evaluator every-ply continuation (best-case Arena self-play): **16.68%** median paired saving. This is not representative of two distinct Arena evaluators, which search on alternating plies. Do not promote Arena independently on this small best-case sample.
- Held-out reference: small positive aggregate score differences, but a reproducible **22.5 cp worse choice** in one Play position. This is a per-position regression, not evidence of a population Elo loss; neither a non-regression claim nor move equality is a substitute for a larger strength trial.
- Passing only `yieldEveryMs: 16` **does not yield effectively** in the cached batched workload. A minimal scheduler correction improved cancellation, but cost **41.30% median paired wall overhead** (22.88–54.05%), far above the 5% limit. The correction was reverted.

The conservative decision is to stop this pilot rather than add unqualified product lifecycle state. An Arena-only follow-up, broader strength testing, remaining Play levels and a lower-overhead yield implementation remain unvalidated.

## Source claims checked

The sol/grok caller claims are correct: Centipawn Play and Arena call generic `chooseMove` without a root; both truncate incoming history to 16 positions. LC0 has internal retained-root searchers. GLM's claim that Arena already retains **Centipawn** trees is false: the nearby `reuseTree` options belong to LC0/big-net paths.

`advanceSearchRoot` and `searchRoot` require both FEN **and exact history**, with a separate value/contempt compatibility key. A child grows its history; comparing that to the next caller's truncated array fails after the boundary. No FEN-only fallback is safe. The production SquareFormer encoder consumes `history_plies: 7` (including its repetition flags within that encoding window); search draw detection still needs full game history.

Another verified gap: `runBatchedVisits` initializes `lastYield` on each invocation. The fixed-visit outer loop invokes it for one batch at a time and disables its own yield when batching. Thus fast cached batches individually below 16 ms can starve macrotasks for the whole search. The temporary correction removed `!hasBatchedScheduler` from the outer fixed-visit yield condition, without changing `scoreEdge` or selection. This establishes the mechanism; it is **not shipped**.

## Experimental semantics and limits

`scripts/centipawn_tree_reuse_browser.ts` is manually bundled/injected into a **built** app, not imported by any product entry point. Its `reuse` argument defaults to no product behavior because there is no product integration.

- Fixed replay: three legal 20-ply Ruy Lopez, Nimzo-Indian and Najdorf continuations, specified as UCI moves in the script. Prescribed moves, not adaptive games generated from each arm's choices. This measures replay continuation, not Elo.
- Play-like arm searches even plies, advancing through **both** intervening moves. The Arena best-case arm searches every ply with one evaluator. Each replay game starts with a null root; only one root is held at a time and missed descendants drop it.
- Full newest-first history belongs to search identity, repetition and the cache key. A wrapper bounds **neural evaluation context** to 16 prior positions. The identical 4096-entry `CachedEvaluator` and GPU session are used by both arms; cache starts empty at each A/B run and remains warm across its three games. This isolates tree retention, not cache enablement.
- Budget is **total visits**, not added visits. At target 400, an inherited root with 90 visits runs 310 new simulations. The probe records inherited visits, completed visits, total visits, new backend misses, cache entries, wall time, node/edge/history counts, a tree-size accounting proxy and browser heap samples. Do not describe inherited work as newly completed work.
- Retained roots exceeding a **16 MiB accounting proxy** are dropped wholesale after search. No parent/transposition-table references are retained. This is not a measured 16 MiB heap/RSS cap; accounting uses estimates and double-counts shared FEN references. A production memory policy would need real device validation and timed-search transient bounds.
- Planned product resets (new game, takeback, unrelated history/FEN, evaluator/model/runtime/policy/contempt generation, cancellation/disposal) were deliberately **not implemented** after the gates failed. The probe's local roots cannot leak between games or arms. Fixed-visit product/parity fixtures remain untouched.

## Timing and memory evidence

Machine: Apple M4 MacBook Air, 24 GB, macOS 26.4.1, Chromium WebGPU, Centipawn promoted `f32/v2-shape-k16`, batch 32, 400 total visits. All timing was bracketed by the shared `benchlock` command (a holder process waited for an explicit release file while browser-tool probes ran); native reference work used `benchlock ssh ... taskset -c 0 ...`. No builds ran while holding the lock.

Play order was ABABABAB; Arena order ABAB. Use the **median of adjacent paired percentage changes** as the promotion metric, not the ratio of two independently pooled medians on a drifting shared machine.

| Replay / arm | Total wall median; min–max (ms) | New neural evaluations/run | Useful reuse / eligible turns |
| --- | ---: | ---: | ---: |
| Play, off, 30 searches/run | 8224.36; 7254.55–8541.71 | 7170 | — |
| Play, on | 7193.49; 7110.25–8013.47 | 8002 | 27/27 |
| Arena every ply, off, 60 searches/run | 15739.24; 14830.17–16648.31 | 9207 | — |
| Arena every ply, on | 13065.88; 12986.65–13145.11 | 10039 | 57/57 |

Play paired savings: **6.18%, 13.52%, 11.50%, 1.99%**. The pooled-median ratio suggests 12.53%, but the paired median is **8.84%** and the last pair is only 1.99%. Arena paired savings: **11.36%, 21.99%**. Machine noise/JIT drift was substantial, so these are small pilot estimates, not precision forecasts. An initial long awaited browser eval was retried by the automation transport and produced overlapping runs; that entire contaminated attempt was discarded. Subsequent probes launched once with an immediate return, then polled for completion.

Neural misses actually **increased** with retention: the fresh arm rewalks cached paths, while the retained arm spends its reduced simulation allowance differently/deeper. Reuse is not demonstrated to avoid backend evaluations here. Chosen moves and neural-miss totals were deterministic within each reuse arm across repetitions, but agreement between arms is not the strength metric.

Inherited visits, excluding initial roots: Play median **90**, range **1–306**; Arena every-ply median **280**, range **1–399**. Root reuse remained valid at ply 18, past the truncated-history boundary. Max retained proxy: Play **9.60 MiB** (400 nodes / 16450 edges maxima), Arena **9.95 MiB** (400 nodes / 16399 edges maxima). Browser `usedJSHeapSize` samples ranged about **33–188 MiB** across arms; those include cache, runtime, garbage and UI and exclude GPU memory. They do not isolate retained-tree RSS or establish mobile safety.

Committed per-position diagnostics and run summaries: [`centipawn_tree_reuse_2026-10.json`](centipawn_tree_reuse_2026-10.json). Full raw histories/heap/node diagnostics are also in the session scratchpad files listed below.

## Held-out strength check (not move-equality scoring)

Native Stockfish 19 on Yukon, 500 ms **per forced candidate move**, Threads=1, Hash=16, fresh `ucinewgame` between searches, CPU 0, ABAB fresh/reuse candidate order. All 90 selected positions were paired. Only five pairs changed move; identical choices contribute zero difference by identity rather than paying to re-score the same move. These opening continuations have no repetition; the reference uses their bare FEN. Scores are from the root side's perspective.

| Case / game / ply | Fresh → retained | Fresh cp A1/A2 | Retained cp B1/B2 | Median delta cp |
| --- | --- | ---: | ---: | ---: |
| Play / Ruy / 6 | Ba4 → Bxc6 | 26 / 26 | 5 / 2 | **−22.5** |
| Play / Ruy / 16 | d4 → h3 | 22 / 22 | 43 / 43 | +21.0 |
| Play / Najdorf / 2 | Nc3 → Nf3 | 17 / 17 | 28 / 29 | +11.5 |
| Arena every ply / Ruy / 5 | a6 → Nf6 | −41 / −44 | −21 / −21 | +21.5 |
| Arena every ply / Ruy / 16 | d4 → h3 | 19 / 20 | 51 / 43 | +27.5 |

Mean paired delta including unchanged choices: Play **+0.33 cp** over 30 positions; Arena best-case **+0.82 cp** over 60. This limited positive average does not negate the reproducible per-position regression, nor establish strength parity across levels, tactical/endgame suites or equal-clock games. No Elo claim. Scorer: `scripts/centipawn_tree_reuse_score.py` (`ENGINE INPUT.json OUTPUT.json`; input rows contain `fen`, `fresh`, `reuse`, plus optional diagnostic keys).

## Play yield / cancellation evidence

1600 visits at start position, batch 32. Each trial first primes the same 4096-entry cache, then measures a fresh-root all-cache search (zero new backend misses). Input proxy: a timer requests AbortController cancellation at 20 ms; this tests macrotask dispatch and Stop delivery, not trusted pointer-to-paint latency or p95 user input latency.

- Original scheduler, `yieldEveryMs` 0 vs 16: neither arm serviced that timer before search finished. Completion ranges were 159.86–172.82 ms (off) and 157.99–195.49 ms (16). A null handler in the artifact means it never ran during search; the expired timer was then cleared.
- Temporary corrected outer loop, ABABABAB: cached full-search median **441.28 ms off**, **628.44 ms on**; ranges **412.48–498.50** and **554.04–691.14 ms**. Adjacent overhead **38.65%, 54.05%, 43.96%, 22.88%**, median **41.30%**. All choices remained `e2e4`, all measured misses zero.
- Correction's four cancellation handlers: **40.13, 52.93, 53.11, 60.68 ms** (sorted), median **53.02 ms**; abort returns **40.31–60.88 ms**, median **53.28 ms**. Off searches all completed before cancellation delivery, taking **273.41–425.73 ms**.

Absolute costs differed sharply between the two lock windows on this busy shared machine; compare only adjacent pairs within a window. Every corrected pair exceeded 5% overhead. Rejected both the ineffectual option-only change and the functional but too-expensive correction. A lower-overhead scheduler/worker experiment is separate work, not a covert broad PUCT rewrite.

## Verification and built-app checks

- `npm test`: typecheck **0 errors / 0 warnings**; **830 passed, 3 expected environment skips**, 0 failures (833 tests total including three new guardrails).
- `npx biome check` on touched supported files: clean after formatting.
- New tests in `tests/centipawn_tree_reuse_probe.test.mjs`: full-history root advancement over 20 plies, exact inherited/added accounting, rejection of same-FEN/different-history reuse, fresh default searches, and repetition beyond the neural window with unchanged bounded encoding.
- `BUILD_SCOPE=product npm run build:client`: passed. Served via `PORT=5183 npm run web:isolated:static`. No dev-server WASM path was used.
- Built Play, Centipawn Standard: `e4 ...e5`; Takeback restores the starting board; `d4` resumes play; New game restores the starting board and engine-ready state. Screenshots saved and read.
- Built Arena: Centipawn auto/TVMJS, 32 visits vs Stockfish Lite depth 8, fixed budget. UI retained 10 games (rather than the requested one); all 10 completed, nine losses / one draw, no illegal move/runtime failure; PGNs saved. **Baseline smoke only**, not a reuse-vs-off strength comparison. The `maxPlies=12` query is ignored by Arena; no capped-game claim.

Scratchpad root (`SP`): `/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad`.

Artifacts under SP:

- `centipawn-reuse-400-abababab.json`, `centipawn-reuse-arena-400-abab.json`
- `centipawn-reference.json`, `centipawn-score-input.json`, `centipawn-reference.log`
- `centipawn-yield-original.json`, `centipawn-yield-fixed.json`, `centipawn-yield-fixed-response.json`
- `centipawn-build.log`, `centipawn-tests.log`, `centipawn-arena-pgn.json`
- `centipawn-play-move.png`, `centipawn-play-takeback.png`, `centipawn-play-new-game.png`, `centipawn-arena-match.png`

## Reproduction

Build the product and bundle the probe **outside benchlock**, without installing dependencies:

```sh
BUILD_SCOPE=product npm run build:client
node --input-type=module -e "import {build} from 'esbuild'; await build({entryPoints:['scripts/centipawn_tree_reuse_browser.ts'],bundle:true,format:'esm',platform:'browser',outfile:'dist-client/centipawn-reuse-probe.js'});"
PORT=5183 npm run web:isolated:static
```

Open the built app with a dedicated agent-browser session. Inject `/centipawn-reuse-probe.js` as a module. Under benchlock, call `centipawnReuseProbe.run(reuse, 400, 0, 20, 2)` for Play-like continuation or final argument `1` for best-case Arena. Interleave false/true, initialize/warm once and launch async work with an immediate return to avoid transport retries. `cachedWall(yieldEveryMs)` and `responsiveness(yieldEveryMs)` test the original scheduler. To reproduce the rejected correction, temporarily remove the outer fixed-visit yield condition's `!hasBatchedScheduler` term, bundle into a **different** probe filename, reload, measure, then restore the source. Never import either probe into the product.
