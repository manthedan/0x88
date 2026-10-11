# Analysis continuity — October 2026

Branch base: `2ea6fa1`. Scope: Analysis, one Arena loader flag, and minimal shared loader/UCI compatibility changes. No engine/net upgrades, SMP changes, second cache, or changes to Play.

## Decisions

- Small LC0 uses existing persistent model storage, a 2048-entry evaluation cache, and compatible subtree reuse. Retention is disabled above 16,000 visits. A new `GameTree` (New Game, PGN load, FEN load) cancels work and queues a tree/eval-cache reset; runtime/model reload disposes the worker. Analysis has no mutable contempt setting; search compatibility checks value/contempt identity.
- Aborted/failed internally retained LC0 searches discard the root: partial edge backups precede final root accounting. Completed evaluations remain reusable. The main-thread fallback uses the same bounded evaluator/searcher.
- Arena opts into the existing small-model cache. TVMJS wasm uses the **same** `lc0-browser-models-v1` Cache Storage bucket and IndexedDB metadata store. A SHA query parameter identifies manifest content; the loader accepts an explicit validated manifest entry. Metadata mismatch/expiry rehashes. Optional storage failures retain verified downloaded bytes without refetching; integrity/cancellation errors still propagate. Unavailable-cache fallback fully hashes executable bytes.
- CPU preparation resets for a different game/adapter or failed/aborted search bookkeeping. Ordinary navigation sends `position fen <root> moves ...` through a typed UCI option, including variations and repetition history. Unchanged Threads options are not resent by Analysis.
- **Adapter limitations:** experimental Reckless browser-api exposes `set_fen`, not replay; its label now explicitly says FEN-only. UCI Reckless supports history. Viridithas's forced one-shot Analysis policy is unchanged and cannot preserve a TT. Other adapters' internal Hash handling is outside this lifecycle change; the measured CPU gain is specifically SF Lite.
- Review uses exact engine/variant/depth/history/MultiPV/runtime/CPU-settings keys and already White-relative scores. It remains single-PV: **default three-line interactive entries deliberately miss**. Matching Lines=1 results and repeated Reviews can hit. Results and completed selection markers are bounded to 512 entries. Same-FEN transpositions restore only complete exact-history selections; navigation cancels incompatible searches and suppresses stale publication. New games clear Review state.
- Maia has one active request and one replaceable pending board, immediate navigation invalidation before debounce, error-safe draining, and disposal invalidation. Worker ORT operations are serialized. Submitted GPU inference is not preemptible.

## Measurements

Every timed experiment owned the shared `scratchpad/benchlock`; A/B ordering was interleaved. Built product assets were served with COOP/COEP on port 5182 and driven by agent-browser (Chromium, shared macOS machine, WebGPU available). Other agents' builds/tests could run concurrently: cold loader timings were noisy; CPU walks and final cache-heavy searches were relatively stable. No mobile/RSS or playing-strength claim is made.

Ranges are min–max, not confidence intervals. Loader timings exclude runtime/session compilation; LC0 searches exclude initialization. Reused fixed-visit budgets are **cumulative**, not additional visits. Review timing excludes the interactive searches that prepopulate the cache; it measures avoiding work already paid for.

| Experiment | A median (range), ms | B median (range), ms | Evidence |
|---|---:|---:|---|
| LC0 byte loader, cache disabled/enabled, 3 AB pairs | 170.93 (107.47–425.21) | 30.33 (25.74–104.30) | A hash median 18.30; B trusted metadata/hash 0 |
| 400-visit LC0 search followed by its PV move; cache 0/reuse off vs 2048/reuse on, 3 AB pairs | 6444.13 (6137.39–6798.24) | 4942.02 (4566.54–5140.20) | 23.3% lower wall; B reused root, misses 401→294 |
| Centipawn 20,192,787-byte wasm read/verify, 4 AB pairs | 66.51 (57.22–173.86) | 22.64 (15.95–27.24) | Same SHA; A hash median 19.13, B metadata/hash 0 |
| SF Lite d12, 128 MiB TT, **one thread**, 8-ply walk; reset+bare FEN vs continuity, 4 AB pairs | 402.11 (399.25–414.31) | 280.16 (273.77–291.86) | 30.3% gain, above 10% gate |
| SF Lite Lines=1, Review of 8 plies, empty vs interactively populated cache, 3 AB pairs | 239.63 (210.81–450.96) | 0.505 (0.505–0.560) | 9 searches→0; 0/9→9/9 cached positions |
| Synthetic serialized Maia workload (400 ms/request), eight boards 160 ms apart, 3 AB pairs; latency after last navigation | 2081.42 (2080.65–2083.88) | 480.06 (476.12–486.75) | Dispatches 8→4; obsolete completions 7→3 |

SF walk per-ply median: 49.445→27.140 ms; nodes: 36,253→21,315. Opening: `e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6`. Each regime repeated identical bestmoves; A/B agree at 6/8 plies. Retained TT/history changes exploration, so this is not bit-equivalent fresh-TT search. LC0 chose `d2d4`, then `d7d5`, in all six runs.

The Review experiment used the built page's actual **eight-thread** broker grant (confirmed by UCI `Threads value 8`). A temporary localStorage attempt to pin one thread did not change the already-mounted broker; these are not one-thread Review measurements. Fresh SMP searches/TT walks can differ from earlier interactive scores. Cache hits preserve those earlier results, not a promise of equality to a fresh search. Repeating the default flow's completed Review reused 4/4 positions with zero `go` commands and identical accuracy/classification summary. Unit tests separately verify Black score perspective and exact setting/history misses.

## Rejected: `yieldEveryMs: 16` on Centipawn Analysis

Real TVMJS WebGPU evaluator, ORT fallback disabled, warmed 4096-entry cache, start position, 1600 visits, batch 32, MontyLite policy, three PVs. Five interleaved wall/input pairs and three Stop pairs:

| Metric | Existing 0: median (range), ms | Candidate 16: median (range), ms |
|---|---:|---:|
| Wall | 147.325 (145.125–149.460) | 147.195 (145.385–152.080) |
| Zero-delay input timer | 147.355 (145.155–149.500) | 147.265 (145.440–152.160) |
| Five-ms Stop timer dispatch | 146.480 (146.165–146.840) | 146.020 (145.770–147.570) |

All ten searches had 1600 visits, 1529 cache hits, zero neural misses, and identical bestmove/full PV. **Neither setting aborted any of the three Stop trials**: the timer ran after search completion. The negligible wall difference is not a demonstrated gain. Although overhead did not exceed 5%, responsiveness did not improve, so the nominal flag is **not shipped**.

Source explains the false premise: `src/search/puct.ts` calls `runBatchedVisits` in batch-sized chunks; that function resets its yield clock on each call, while the outer fixed-visit loop skips its own yield check for batched searches. Cache-hot 32-leaf chunks stay below 16 ms individually, starving timers across the full search. A meaningful fix belongs to the shared PUCT scheduler, concurrently owned by another optimization task; this branch does not silently change Play's scheduling or claim this flag solves Stop latency.

## Built-flow verification and limits

- Final-code product build: default LC0 small 400 + SF Lite depth 12, Lines=3; start plus `d2d4 d7d5 c2c4`. Each position completed, all LC0 results had 400 visits; misses were 401/294/218/236, and all three stepped searches reused their roots. One `ucinewgame`, then progressively extended root-FEN/history commands. First Review completed three moves with 0/4 exact hits (expected MultiPV mismatch); repeated Review hit 4/4 with unchanged summary.
- Typed-vs-raw UCI history parity: two AB pairs each for opening and repeated-knight histories. All eight runs agreed within each position on bestmove, nodes and score (opening `d2d4`, 22,033 nodes, +34 cp; repetition `e2e4`, 14,150 nodes, +36 cp).
- Live Maia delayed-reply/error test was **blocked**: the optional `/models/maia3/maia3_simplified.onnx` asset is absent from both public and built model directories (404). No real Maia inference-latency or live error-recovery claim. Coalescing/navigation/error/disposal tests and the explicitly synthetic backlog experiment passed.
- Browser DOM/worker/UCI evidence was saved. Native screenshot calls repeatedly timed out even after inference workers were stopped; no PNG was produced or claimed as verified. Visual PNG inspection remains unverified.

Evidence in the assigned scratchpad: `analysis-continuity-{neural,neural-summary,tvm,cpu,review,maia,yield,cpuParity,defaultFlow,final-metrics}.json`, `analysis-continuity-review-equivalence.json`, `analysis-continuity-maia-live-initial.json`, and `analysis-continuity-default-snapshot.txt`. Default-flow `expectedFen` was reconstructed from its recorded UCI move sequence; the original `fenInput` field was blank because it is a manual load input, not a current-position display. Local probe/driver copies are experiment artifacts, not deployed product routes.

Validation: typecheck, full suite (850 passed, 3 environment-dependent skips, zero failures), branch-owned Biome check with no new issues, and product build succeeded. Structured review findings and the final-commit verdict are recorded in the scratchpad report, not an interim claim here.
