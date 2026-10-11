# Play neural early stop — October 2026

## Decision

Ship `earlyStop: 'root-dominance'` on Play's small Lc0 search and on Centipawn. Leave `best-stable` and `kld-stable` off. Do not add a clock, and do not change big-net Play (t3, BT4, Leela Queen Odds).

Root dominance is the only mode that keeps the full-budget move. On 83 positions, both engines, and visit caps 100, 400, and 1600, it matched the uncapped search on every position, including all 20 tactical positions. Pooled median visit saving is 20%. Lc0 at every measured cap, and Centipawn at 400 and 1600, clear a 10% median visit saving and a paired wall-time cut. Centipawn at 100 visits does not clear 10% (median saving 4%, because the batch is 32). That cell still plays the same move, so the flag stays on for every small-Lc0 and Centipawn level, including the ones too short to stop early.

`best-stable` saves more visits and never missed a tactical move in this suite, but it changes quiet and recapture moves. Several of those changes are 7–24 centipawns worse for the side to move against Stockfish. That is a measurable quality loss, so it does not ship. `kld-stable` has a pooled median visit saving of 0. Where it does stop early on Lc0 it also changes quiet moves, including one that is 9 centipawns worse.

## What was already implemented

`src/search/puct.ts` already has `earlyStop: 'root-dominance' | 'best-stable' | 'kld-stable'`. Root dominance stops when the leading root move has more visits than the second move plus every visit still left in the cap (`rootVisitDominanceStop`). The temperature-0 move is the visit argmax, so a stop at that point cannot change the move of that search. There is no separate minimum-visit knob. A single legal move does not take this path (`edges.length < 2`). The check runs after each batch chunk, not after every visit.

`best-stable` defaults are minimum 32 visits, check interval 8, two confirming checks, visit lead equal to the interval, and maximum Q delta 0.05. The first aligned check only records the leader, so the earliest stop is the third aligned check. Lc0 batch 1 can stop at 48 visits. Centipawn batch 32 only checks at 32, 64, and 96, so the earliest stop is 96. At a cap of 100 that is a 4% saving even in the best case.

`kld-stable` defaults are minimum 32 visits, interval 8, KL threshold `1e-4`, and two stable checks. The in-code comment reserves it for datagen. This measurement used the defaults. No threshold was retuned.

Play itself was fixed-visit. Small Lc0 and Centipawn levels are 8, 32, 100, 400, and 1600 (`src/lc0/engineCatalog.ts`). The UI default level is the middle step, 100 visits. No product page passed `earlyStop`. `docs/native-lc0-search-gap.md` section 5 already describes the three modes and names threshold-versus-strength tuning as the missing piece. This note is that measurement. `tests/search_metamorphic.test.mjs` already checks that root dominance fires on a fixed visit budget and that finishing exactly on the cap is not relabeled as an early stop.

## Kill rule, fixed before the suite

Visit saving on one position is `(requested - completed) / requested`. A mode passes the time bar only if the median saving is at least 10% at equal quality.

Quality loss, as operationalized before the run: mean Stockfish centipawn loss of the chosen move at least 5cp worse than the full-budget move, or any tactical position at least 100cp worse, or a tactical mate the full budget found and the early stop missed. Poor tactical agreement: tactical move-agreement with the full budget under 90%.

The task line is stricter than that 5cp mean: any measurable quality or strength loss also kills the mode. A handful of quiet moves that are 7–24cp worse counts, even when the mean across all positions stays near zero because most moves match and some early stops are slightly better.

## How it was measured

Machine: Apple M4, 10 cores, 24 GB. Node `/opt/homebrew/bin/node` v25.9.0. Both nets ran through the repo's Node ONNX Runtime WebAssembly path (`Lc0OnnxEvaluator`, `SquareFormerEvaluator`). There is no `onnxruntime-node` binding. ORT stays at one thread, which is also what Play's main-thread session uses. Setting `ORT_INTRA_OP_NUM_THREADS=4` failed session creation on the qdq8 net. Absolute milliseconds are therefore CPU-fallback times. Visit counts are the decision metric. Centipawn in the browser prefers TVMJS WebGPU, so product milliseconds will be lower. The visit ratio should carry over. The bench did not pass Play's `yieldEveryMs: 16`, so product wall time is a little higher and the ratio should be similar.

Models are the Play defaults: `public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx`, and `public/models/bt4_soap_rem_c19000_final.onnx` with its `.meta.json`. Lc0 options match Play: temperature 0, `drawScore` -0.25, `searchContemptLimit` 16, cache 2048, `includePv` false, cold tree (`reuseTree` false). Centipawn options match Play's move search: `montyLitePuctPolicy`, temperature 0, batch `min(32, visits)`, cache 4096, no draw score and no contempt. Each search cleared the eval cache.

Search contempt calls `Math.random`. Every search replaced it with `mulberry32(1)` for the duration of that search, so an early stop and the full budget share one noise prefix. Centipawn Play does not use contempt. The replacement was still applied.

Stockfish is `/opt/homebrew/bin/stockfish`, Threads 1, Hash 64, `go depth 16 movetime 1500`. The score is taken from the position after the played move and negated, so positive is good for the mover. Loss is full-budget mover score minus early-stop mover score. Positive means the early move is worse. Mate-scale scores (`|loss| >= 90000`) are counted separately and left out of the centipawn mean.

The main pass is one sample per cell, back to back inside each position (full budget, then root dominance, then best-stable, then kld). It ran under the shared bench lock. Load average on those 1992 rows: median 11.3, p25 5.8, p75 15.8, max 152. Treat the absolute milliseconds as noisy. The paired time ratio (early milliseconds / full-budget milliseconds on the same position) is the wall-time summary. A later 3-round repeat at budget 100, one position per class, ran at load median 2.32 (p25 2.22, p75 2.41, max 2.64).

`best-stable` and `kld-stable` stop visits do not depend on the cap until they hit it. A row that had already stopped under a smaller cap was copied forward. 306 rows are copies. Their visit counts are exact. Their copied milliseconds are not a fresh timing at the larger cap and are excluded from the time ratios below. Root dominance and the full budget were searched again at every cap.

Raw rows were left in `/tmp/play-early-stop/` (`search.jsonl`, `timing.jsonl`, `sf.json`, `summary.json`). They are not committed.

## Positions

The suite is 84 listed positions from fixtures already in the repo, 83 of them searched. An external UHO file is referenced by other eval notes and is not in this worktree, so it was not used.

Sources: `fixtures/lc0/correctness_corpus.json` final positions, recapture plies from those lines (capped while building), `fixtures/lc0/fen_only.json`, non-terminal rows of `fixtures/lc0/search_edge_cases.json`, `fixtures/lc0/history.json` finals and recapture plies, plus named positions already used in repo tests or docs (kiwipete, a perft en-passant, a perft promotion, the rook endgame, Lasker–Reichhelm, and an Italian tabiya).

Primary class, in priority order tactical, recapture, endgame, quiet: in check, mate in one, or a hanging or winning capture is tactical. A position where the previous ply in a repo line captured and the side to move can capture back is a recapture. Non-pawn material at most 13, or at most 8 men, is an endgame. Terminal positions were dropped.

Listed counts: quiet 49, tactical 20, recapture 5, endgame 10. Searched counts: quiet 49, tactical 20, recapture 5, endgame 9. Two suite rows share the id `en-passant-available`. The harness keys on id, so only the first was searched: the quiet fen-only position `rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3`. The skipped row is the edge-case endgame `4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1`. 83 positions are enough for the decision. It was not re-run.

## Root dominance

Agreement is against the same engine's full visit cap. Tactical n is 20. Visit saving is the median. The time ratio is the median of early/full milliseconds on freshly measured pairs. Spread is p25–p75 of that ratio. Main-pass milliseconds are single samples under the load above.

| Engine | Cap | Agree | Tactical | Stopped | Visit median | Visit p25–p75 | Visits median | Time ratio | Ratio p25–p75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lc0 | 100 | 83/83 | 20/20 | 73 | 22% | 6.5–40.5% | 78 | 0.751 | 0.605–0.922 |
| Lc0 | 400 | 83/83 | 20/20 | 80 | 28% | 8.1–44.8% | 288 | 0.723 | 0.547–0.925 |
| Lc0 | 1600 | 83/83 | 20/20 | 81 | 37.9% | 11–47% | 993 | 0.625 | 0.531–0.904 |
| Centipawn | 100 | 83/83 | 20/20 | 64 | 4% | 4–4% | 96 | 0.920 | 0.866–0.948 |
| Centipawn | 400 | 83/83 | 20/20 | 69 | 20% | 4–36% | 320 | 0.750 | 0.576–0.940 |
| Centipawn | 1600 | 83/83 | 20/20 | 78 | 32% | 12–44% | 1088 | 0.668 | 0.547–0.871 |

Pooled across all 498 root-dominance rows: median visit saving 20%, p25 4%, p75 41%, mean 22.4%. Stockfish loss is 0 on every row. No tactical position was 100cp worse. No tactical mate was missed. Move agreement on tactical positions is 100%.

Median visit saving by primary class:

| Engine | Cap | Quiet (49) | Tactical (20) | Recapture (5) | Endgame (9) |
|---|---:|---:|---:|---:|---:|
| Lc0 | 100 | 23% | 34% | 11% | 5% |
| Lc0 | 400 | 31% | 33% | 33% | 3.8% |
| Lc0 | 1600 | 40.2% | 36.8% | 37.9% | 3.8% |
| Centipawn | 100 | 4% | 4% | 4% | 4% |
| Centipawn | 400 | 20% | 32% | 12% | 4% |
| Centipawn | 1600 | 34% | 28% | 26% | 14% |

Tactical positions save as much as quiet ones, or more. The endgame slice on Lc0 (n=9) stays around 4–5% at every cap. Those searches still play the same move. Centipawn at 100 visits stops on a 32-visit boundary, usually 96, which is why every class shows a 4% median. Levels 8 and 32 use a batch equal to the budget, so the stop check cannot fire before the search is already done. Leaving the flag on there is free.

Budget-100 repeats, 3 rounds, mode order rotated, load median 2.32. Medians of the three wall times:

| Engine | Position | Class | Early ms (min–max) | Full ms (min–max) | Ratio | Early visits |
|---|---|---|---:|---:|---:|---:|
| Lc0 | deterministic-seed1-ply12 | tactical | 1515 (1513–1615) | 2885 (2879–2904) | 0.525 | 52 |
| Lc0 | deterministic-seed2-ply20-recap-ply18 | recapture | 2291 (2287–2397) | 2906 (2886–3112) | 0.788 | 79 |
| Lc0 | deterministic-seed1-ply4 | quiet | 2784 (2772–2935) | 2897 (2888–2901) | 0.961 | 96 |
| Lc0 | black-en-passant-available | endgame | 2750 (2718–2771) | 2854 (2844–3005) | 0.964 | 95 |
| Centipawn | deterministic-seed1-ply12 | tactical | 137 (132–141) | 157 (145–162) | 0.873 | 96 |
| Centipawn | deterministic-seed2-ply20-recap-ply18 | recapture | 83 (83–87) | 100 (96–104) | 0.830 | 96 |
| Centipawn | black-en-passant-available | endgame | 245 (242–259) | 257 (242–266) | 0.953 | 100 |
| Centipawn | deterministic-seed1-ply4 | quiet | 278 (258–279) | 265 (259–279) | 1.049 | 100 |

The two Centipawn rows that did not stop (100 visits, `visit-budget`) sit inside the noise, and one is slightly slower. The rows that actually stop are faster in every round. The Lc0 full-budget recapture has one 3112 ms outlier against two runs near 2900. Spreads on the stopped Lc0 rows are a few percent.

## Why the other two modes stay off

Tactical agreement is 20/20 in every cell below. `tacticalWorse100` is 0. `missedMateWhereFullFoundIt` is 0. The tactical kill does not fire. The failures are visit saving, or quiet-move quality.

| Engine | Cap | Mode | Agree | Visit median | Stopped |
|---|---:|---|---:|---:|---:|
| Lc0 | 100 | best-stable | 82/83 | 52% | 59 |
| Lc0 | 400 | best-stable | 77/83 | 88% | 79 |
| Lc0 | 1600 | best-stable | 74/83 | 97% | 80 |
| Centipawn | 100 | best-stable | 83/83 | 0% | 18 |
| Centipawn | 400 | best-stable | 77/83 | 52% | 78 |
| Centipawn | 1600 | best-stable | 75/83 | 88% | 83 |
| Lc0 | 100 | kld-stable | 83/83 | 0% | 11 |
| Lc0 | 400 | kld-stable | 82/83 | 22% | 56 |
| Lc0 | 1600 | kld-stable | 77/83 | 80.5% | 83 |
| Centipawn | 100 | kld-stable | 83/83 | 0% | 0 |
| Centipawn | 400 | kld-stable | 83/83 | 0% | 5 |
| Centipawn | 1600 | kld-stable | 83/83 | 26% | 66 |

Pooled median visit saving: best-stable 65% (n=498), kld-stable 0% (mean 25.6%). Kld fails the 10% median bar on the pooled set. Best-stable at Centipawn 100 also fails it, which the batch-32 check schedule forces.

Early moves that Stockfish scores worse than the full budget (positive loss). Scores are the mover's point of view after the move. Depth-16 Stockfish still moves by a couple of centipawns between similar quiet moves, so +2 and +3 are weak evidence. +7 and above is not.

| Engine | Cap | Mode | Position | Class | Early | Full | Loss (cp) |
|---|---:|---|---|---|---|---|---:|
| Lc0 | 400 | best-stable | deterministic-seed3-ply7 | quiet | c7c6 | d7d5 | +2 |
| Lc0 | 1600 | best-stable | deterministic-seed3-ply7 | quiet | c7c6 | d7d5 | +2 |
| Lc0 | 400 | best-stable | deterministic-seed10-ply7-recap-ply6 | recapture | g1f3 | a2a4 | +3 |
| Lc0 | 1600 | best-stable | deterministic-seed10-ply7-recap-ply6 | recapture | g1f3 | a2a4 | +3 |
| Lc0 | 1600 | kld-stable | deterministic-seed10-ply7-recap-ply6 | recapture | g1f3 | a2a4 | +3 |
| Centipawn | 400 | best-stable | deterministic-seed3-ply4 | quiet | d2d4 | e2e4 | +4 |
| Centipawn | 1600 | best-stable | deterministic-seed1-ply4 | quiet | g2g3 | e2e4 | +7 |
| Lc0 | 1600 | best-stable | deterministic-seed2-ply15 | quiet | a7a5 | f6d5 | +9 |
| Lc0 | 1600 | kld-stable | deterministic-seed2-ply15 | quiet | a7a5 | f6d5 | +9 |
| Centipawn | 1600 | best-stable | deterministic-seed10-ply7 | quiet | d7d5 | e7e5 | +14 |
| Centipawn | 400 | best-stable | deterministic-seed2-ply15 | quiet | e7b4 | a7a5 | +24 |
| Centipawn | 1600 | best-stable | italian-four-ply-history | quiet | f1c4 | f1b5 | +24 |

The mean loss over all 83 positions is about 0 to -1cp for these cells (early slightly better on average), because many disagreements go the other way. Examples of early-better moves: Centipawn 400 `deterministic-seed10-ply12` e2e4 instead of g2g4 (-113cp), Lc0 1600 Italian f1b5 instead of f1c4 (-24cp), Centipawn 1600 `deterministic-seed5-ply12` a4b5 instead of d1b3 (-53cp). The pre-registered 5cp mean rule does not fire. The task rule does: the table above is a measurable loss on individual positions, and the move is not the full-budget move.

One mate-scale disagreement is an endgame, not a tactical miss. On `perft-promotion` (`4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1`) at Lc0 1600, both best-stable and kld play `a7a8q` while the full budget plays `h7h8q`. Stockfish prefers the early promotion by a mate-scale margin (loss about -98709, filtered out of the cp mean). It is still a different move, so those modes are not equal-quality.

## Paired games

Root dominance's move equals the visit-cap move on a cold tree, which is what the table above measured (`reuseTree: false`). Play's small Lc0 keeps the tree (`reuseTree: true`). A shallower stopped tree can be the root of the next search, so a later ply could diverge even though each isolated search would not. Centipawn Play does not reuse a tree. It does pass up to 16 history FENs, which the position suite did not.

Confirmation, under the bench lock, is one 16-ply game per engine from the start position. The neural engine is White. Black is native Stockfish, Threads 1, Hash 16, `go depth 12 movetime 300`, `ucinewgame` each move. Two Lc0 searchers, or two cold Centipawn searches, see the same position. Contempt is reseeded to `mulberry32(1)` on every search. Lc0 uses Play's draw score, contempt limit, history, and tree reuse, with `yieldEveryMs` left unset. The two Lc0 searchers share one eval cache, so the second search of a ply is warm and its milliseconds are not a timing result. Centipawn clears its cache every search and uses Play's batch, history, and policy. The move that is played is the full-budget move, so both sides see the same position on the next ply and the early searcher is judged on its own reused tree. Load during the games was about 1.5–1.8.

A long equal-movetime Elo match was not run. Play has no clock, and this stop does not spend the saved time on a harder position. The move lists matched, so the games against this deterministic opponent are the same game. Arena's equal-movetime mode would measure a search that fills a time budget. That is not what Play does.

White moves, all eight plies, agreed.

Lc0 at 100 visits, with tree reuse. `completed` is visits added this search. A reused root already holds visits from the previous ply, and Play only searches the gap up to the cap (`visits - priorRootVisits`), which is why several full-budget rows are under 100 and still stop on `visit-budget`.

| Ply | Move | Early visits | Full visits added | Early stop |
|---:|---|---:|---:|---|
| 0 | d2d4 | 94 | 100 | root-dominance |
| 2 | c2c4 | 75 | 87 | root-dominance |
| 4 | g2g3 | 86 | 90 | root-dominance |
| 6 | f1g2 | 84 | 88 | root-dominance |
| 8 | g1f3 | 76 | 90 | root-dominance |
| 10 | c1d2 | 49 | 91 | root-dominance |
| 12 | d1c2 | 60 | 58 | root-dominance |
| 14 | c2d2 | 39 | 80 | root-dominance |

Stockfish answered g8f6, e7e6, d7d5, d5c4, f8b4, a7a5, b4d2, c8d7. Ply 12 is the useful tree-reuse case: the full search only needed 58 new visits because the reused node was already visited, and the shallower early tree added 60 and stopped on dominance. The move was still d1c2.

Centipawn at 400 visits, cold tree, cache cleared each search. Early was faster on every ply, including the four plies where it searched second. This is one game, not a repeated timing study.

| Ply | Move | Early visits | Full visits | Early ms | Full ms |
|---:|---|---:|---:|---:|---:|
| 0 | e2e4 | 256 | 400 | 698 | 1159 |
| 2 | g1f3 | 256 | 400 | 559 | 1050 |
| 4 | f3e5 | 224 | 400 | 367 | 940 |
| 6 | e5f3 | 224 | 400 | 435 | 983 |
| 8 | d2d4 | 352 | 400 | 953 | 1131 |
| 10 | f1d3 | 224 | 400 | 493 | 1041 |
| 12 | e1g1 | 288 | 400 | 776 | 1125 |
| 14 | c2c4 | 224 | 400 | 476 | 1054 |

Stockfish answered e7e5, g8f6, d7d6, f6e4, d6d5, f8d6, e8g8, c7c6. Median early visits on these eight plies are 240 of 400, a 40% saving. Median early/full time ratio on this single game is about 0.50.

## What changed in Play

`src/lc0/playBrowser.ts` passes `earlyStop: 'root-dominance'` from the Centipawn `chooseMove` and from the small-Lc0 `search`. The one-visit preparation search, Maia, Stockfish-family engines, and the big-net branch are unchanged. `src/search/puct.ts` is unchanged, including `scoreEdge`. There is no clock UI. Levels 8 and 32 are unchanged as numbers. The flag is a no-op there for Centipawn because one batch consumes the budget, and for Lc0 it can only stop if the leader is already mathematically decided inside 8 or 32 visits.
