# Batching strength re-gate (October 2026)

Branch `perf/batching-strength`, base `5c161a9`. **No product default changes.**
This follows the speed/output observations in [the wave-1 note](ort_batching_jspi_2026-10.md), not a new ORT upgrade.

## Decisions

- **B1: diagnostic only, not a product win.** The batch-8 export has positive point scores against an artificial batch-1/export **search-batch-8** baseline at 8 and 32 visits, but only 16 distinct openings were silently repeated. The gains are not significant in the first-pair-per-opening sensitivity. It does not beat the shipped search-batch-1 configuration (B2). No independently validated equal-time B1 claim; no default can be justified by B1 alone.
- **B2: abandoned by user decision.** The product batch configuration is negative at the completed 8/32-visit levels even when restricted to one pair per opening; the partial 100-visit record is also negative, not a completed gate result. The repeated suite invalidates a formal 200-game gate claim. No resumed B2-100, B2-400, equal-time B2, or threshold rollout. This is not evidence about untested higher levels.
- **LQO: keep serial physical batch 1.** Logical batching is about 2.2–2.4x faster per move, but these queen-odds self-games are nearly saturated by the side that still has a queen. Zero head-to-head Elo is not output or move equivalence, and does not clearly establish product strength. Do not promote a dynamic-batch default for all models on this evidence.

The original gate is point Elo >= -10 and CI upper bound >= 0 at every tested equal-visit level, plus equal-time not worse. A reported per-match `gate.pass` is only that local arithmetic test, **not a ship decision**, a minimum-sample check, or proof of equivalence. Final compact evidence gives partial/not-run **and repeated-opening** matches `gate: null`; a not-run match has no Elo estimate.

## Arms and product-path audit

All arms use the built Svelte client, strict `webgpu->webgpu`, ORT 1.27, the shared `searchWorkerCore.ts` -> `Lc0PuctSearcher` -> `src/search/puct.ts` path. No wasm fallback was accepted. Both arms have temperature 0, pipeline depth 1, a 2048-entry evaluator cache, full position history, fresh search trees per move, and no early search stop. Games stop at actual terminal rules or 160 searched plies (then draw); no evaluation-based adjudication.

| Comparison | Candidate | Baseline |
|---|---|---|
| B1 | `t1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx`, search batch 8, physical 8 (pad short batches) | `.batch1.f16.qdq8.onnx`, **search batch 8**, eight serial physical-1 runs |
| B2 | Same B1 candidate | `.batch1.f16.qdq8.onnx`, **search batch 1** (today's product batch configuration) |
| LQO | `lqo_v2.f16.qdq8.onnx`, search batch 8, one unpadded logical physical run | Same file, search batch 8, serial physical-1 runs |

B1/B2 use Play's `drawScore=-0.25`, `searchContemptLimit=16`. LQO's queen-less side uses Play's cpuct 1.5, fpu 0.4, white draw -0.4 / ScLimit 40, black draw -0.6 / ScLimit 32; black queen-less inputs and returned moves use the existing `bt4Engine.ts` color swap. The side with its queen uses regular Play contempt. Legal-prior masking and product contempt are unchanged.

The harness is not the complete Play controller: Play reuses compatible trees, the small net has root-dominance early stop, and LQO has randomized books/policy-book openings. Those are disabled/bypassed to isolate a fixed-visit comparison. B2 matches the product's **batch/model** configuration, not all its game-state policy. Neither B1's baseline nor its gain is today's product baseline.

**Opening-suite correction (autoreview-1):** small-net runs requested 100 openings, but used only the **16 Arena built-in lines**. All 500 `/lc0/opening_books/queen500.epd` entries lack White's queen, so the equal-material filter rejects every filler. The inherited harness silently cycled the 16 openings. State's `openingCount: 100` was a request, not a measured cardinality. Every pair still has consecutive games with engine colors swapped and exactly the same book history/FEN, but the 100 pairs are **not 100 independent opening units**. The final harness rejects cardinality shortfalls and targets requiring repeated pairs before any worker initialization. LQO includes both product missing-queen start FENs and varied highest-weight product-book lines (100 openings, predominantly black missing the queen). Engine roles swap per pair; the odds FEN itself is not mirrored between those two games.

## B1 sanity check

`python3 scripts/compare_batching_strength_models.py` parses the actual ONNX protobufs without inference or external Python dependencies:

- 650 serialized graph nodes are identical.
- 473 initializer names match; 472 serialized initializers, including learned weights and quantization scales/zero-points, are byte-identical.
- The sole initializer difference is `/const/pos_encoding_shape`: `[1,64,64]` vs `[8,64,64]`. Input/output shape metadata also differs.
- Actual file SHA-256: batch 1 `9396c11226de7f29dbad8c60147b1082b8093adc7a0b77819cd0fb33103a095f`; batch 8 `203237314baa01a05edc41f1a59f3e34c8f86c445bc47156ad8cab0c379d9273`.

In B1, PUCT collects logical leaves with virtual visits/collision retries **before** awaiting evaluation, then backs up returned rows in logical order after the entire batch completes. Both arms take exactly that scheduler. Serial physical runs do not back up between rows. The new focused test supplies identical evaluations through serial promise turns vs grouped evaluation and proves identical move, root policy, search trace/collisions, and exact visits at 8/32/100/400. Thus physical latency alone cannot create a B1 fixed-visit search gain; differing tensor outputs can change priors/values, and consequently later search selections.

The raw 200-game B1-8 record is 100/38/62. Candidate White: 56/20/24; Black: 44/18/38. Both colors have positive score differences; each pair has matching opening/history and opposite engine colors. These are repetitions of only 16 opening units, not 100 distinct openings. B1-32 is 107/30/63: White 69/6/25, Black 38/24/38 (gain concentrated in White). No B1 game is a timeout, resignation or illegal-move forfeit. Every recorded equal-visit move has exactly its requested visits. Independent replay verifies legality and checkmate/material terminal results. The 160-ply draw cap is a limitation (31 draws at the cap for B1-8, 6 for B1-32), not a timeout.

**Randomness caveat:** the inherited build `2026-10-11.1` only tested repeated start-position moves at 8 visits. `searchContemptEdge` samples `Math.random()` after an opponent node reaches ScLimit, so temperature 0 does **not** make longer searches reproducible. At B1-8 no node can reach the 16-visit threshold: that gain is a genuine numerical/search-path difference on this opening suite. At 32 and above, numerical drift and uncontrolled ScLimit samples cannot be disentangled from the saved data. No rerun is justified to rescue a product lever already rejected by B2. The final lab-only seeded worker makes future runs reproducible without changing product contempt or its sampling distribution; historical unseeded games are not mixed with it.

## Results and uncertainty

Candidate-relative W/D/L and Elo use **complete opening pairs only**. The following CIs preserve the nominal all-pair computation (including the requested B1-100 +6.9 [−24,38.5] record), **not valid opening-population uncertainty for the repeated small-net suite**. Pair profiling models correlation between the two games of a pair, not across repeated copies of that opening. Intervals are also not corrected for repeated looks or multiple tested levels.

| Match | Recorded / estimated games (pairs) | Candidate W/D/L | Draw rate | Elo [nominal all-pair CI] | Status / interpretation |
|---|---:|---:|---:|---:|---|
| B1 8 visits | 200 / 200 (100) | 100/38/62 | 19% | +66.8 [22, 111.5] | Complete; artificial-baseline diagnostic only |
| B1 32 visits | 200 / 200 (100) | 107/30/63 | 15% | +77.7 [34, 123] | Complete; diagnostic, unseeded ScLimit caveat |
| **B1 100 visits** | **200 / 200 (100)** | **63/78/59** | **39%** | **+6.9 [−24, 38.5]** | **Complete; diagnostic only, not a product win** |
| B1 400 visits | 29 / 28 (14) | 8/10/10 | 35.7% | −24.9 [−119, 74] | **Partial, user cut; no gate verdict** |
| B2 8 visits | 200 / 200 (100) | 12/37/151 | 18.5% | −297.9 [−351.5, −249.5] | Complete record; repeated-suite negative, no formal gate |
| B2 32 visits | 200 / 200 (100) | 24/113/63 | 56.5% | −68.6 [−94.5, −44.5] | Complete record; repeated-suite negative, no formal gate |
| B2 100 visits | 139 / 138 (69) | 23/57/58 | 41.3% | −90.1 [−136.5, −44] | **Partial, user abandonment; no completed gate verdict** |
| LQO 16 visits | 40 / 40 (20) | 20/0/20 | 0% | 0 [−31.5, 31.5] | Complete minor level; odds suite saturated |
| LQO 32 visits | 200 / 200 (100) | 99/2/99 | 1% | 0 [−7.5, 7.5] | Complete; odds suite saturated, not equivalence |
| LQO 64 visits | 80 / 80 (40) | 40/0/40 | 0% | 0 [−16, 16] | Complete minor level; odds suite saturated |
| LQO 256 visits | 92 / 92 (46) | 46/1/45 | 1.1% | +3.8 [−14, 29] | **Partial, user cut; no gate verdict** |
| LQO 1600 ms/move | 0 / 0 (0) | — | — | — | **Not run, user skip** |

**Opening-unit sensitivity:** keep only the first complete pair by game index for each exact opening FEN/book history (no outcome-based selection), discarding later repetitions. This treats distinct openings as the sampling units without inventing independent information from cloned games. It is a post-hoc sensitivity, not a replacement gate or proof about arbitrary openings. At B1-8/32 and B2-8/32, all 100 pairs contain only 16 distinct pair move histories, so repetitions add no move/outcome information. At 100 visits uncontrolled ScLimit samples do produce different repeats; discarding those avoids assuming they are new opening diversity.

| Match | Distinct opening pairs used | First-pair Elo [nominal pair-profile CI] |
|---|---:|---:|
| B1 8 | 16 | +65.9 [−50, 182] |
| B1 32 | 16 | +77.2 [−31, 196] |
| B1 100 | 16 | +10.9 [−50.5, 87.5] |
| B1 400 (partial) | 14 | −24.9 [−119, 74] |
| B2 8 | 16 | −293.0 [−437, −168.5] |
| B2 32 | 16 | −65.9 [−137.5, −2.5] |
| B2 100 (partial) | 16 | −43.7 [−147.5, 42.5] |

These wider intervals remove the apparent significance of B1-8/32 and of partial B2-100. The latter is inconclusive at the opening-unit level; B2 remains abandoned, not rescued. LQO really generated 100 distinct openings; the complete and partial runs never cycled them. JSON records requested/actual cardinalities, observed units, repeat counts, and the first-pair sensitivity separately from the all-pair historical estimate.

B2 400 and B2 equal-time were not run after abandonment. B1 has no equal-time strength result. No lever meets the full product ship gate.

The committed compact evidence is `docs/batching_strength_2026-10.json`. Raw games and PGNs are local ignored files at `artifacts/batching-strength/<match>/games.jsonl` and `games.pgn`; evidence includes their SHA-256 hashes. Pilots are excluded. Unpaired final games are retained for audit but excluded from the Elo/WDL estimate.

**Estimator correction:** inherited summaries used an independent-game convolution with a fixed observed draw rate, misleadingly labelled pair-based. For example, B1-8/32 originally reported +65/+74.5 and B2-32 -86. Final results use logistic score Elo and profile the unrestricted five-category pair distribution, allowing within-opening correlation. The nominal 95% CI uses LR cutoff 3.841, a 0.5-Elo grid, and endpoints clipped at +/-2400. Tests cover split vs correlated sweep pairs, all-win boundaries, and empty data. Old summary numbers/methods remain in the compact evidence for provenance. Neither estimator's all-pair CI fixes between-pair pseudoreplication; use the opening-unit sensitivity for the small-net suite's diversity limitation.

**User stops, not gate stops:** `lqo-v256/early-stop.json` records the request at **88 games** (2026-10-12 ~22:55 PDT), because more saturated odds games would add hours rather than information. The active chunk finished/persisted four more games before stopping: the final artifact has 92. `b1-v400/early-stop.json` records **26 games** (2026-10-13 ~01:00 PDT), because B1 was only diagnostic after B2 rejected the product lever; the final chunk persisted 29, of which 28 form pairs. `lqo-time/early-stop.json` records **0 games**, explicitly skipped by user at ~01:00 PDT for the same saturated-suite limitation. B2-100 was abandoned at ~134 games in the takeover instructions; its inherited final artifact has 139, one unpaired. There is no B2-100 early-stop file. These trailing games are retained transparently, not used to relabel any stopped match as complete.

The future driver can stop once a nominal CI lies entirely on one side of −10 after at least 20 pairs. This is a resource-saving heuristic, **not sequentially valid inference**; `--no-gate-stop` is required for a predeclared fixed-size run. None of the three user stop files above contains a measured Elo or CI. No matches were run during this closeout.

The LQO 16/32/64 records have only 2/25/11 identical full-move pairs out of 20/100/40. Outcomes nearly all favor the side with its queen regardless of engine assignment (queen-side score 1 / 0.995 / 1). Therefore **0 Elo is not identical games**. The narrow pair CI at 32 visits is conditional on this saturated odds suite; it cannot bound degradation against humans or a stronger independent opponent.

## Timing, noise and reproduction

Observed game-move wall time, milliseconds, **median [p10, p90]** across recorded moves (including the unpaired audit game where present):

| Match | Faster candidate | Baseline |
|---|---:|---:|
| LQO 16 | 79.21 [48.37, 105.98] | 173.38 [79.32, 247.89] |
| LQO 32 | 157.80 [86.14, 203.07] | 365.75 [167.95, 486.83] |
| LQO 64 | 339.65 [170.47, 405.38] | 811.47 [332.04, 993.22] |
| LQO 256 (partial) | 2004.69 [674.86, 3307.36] | 4310.94 [1229.32, 6481.28] |
| B1 8 | 34.25 [32.13, 36.69] | 77.90 [35.75, 87.41] |
| B1 32 | 68.85 [50.72, 73.35] | 228.39 [105.04, 281.99] |
| B1 100 | 215.24 [172.00, 299.38] | 733.31 [441.88, 1079.72] |
| B1 400 (partial) | 853.19 [608.70, 1174.75] | 2967.12 [1503.73, 4353.00] |
| B2 8 | 34.61 [32.33, 37.23] | 64.30 [31.32, 86.07] |
| B2 32 | 69.52 [52.95, 74.59] | 234.97 [101.84, 286.05] |
| B2 100 (partial) | 209.13 [165.25, 236.94] | 665.02 [263.42, 889.04] |

This is a noisy shared Apple M4 machine: 94 saved chunk-end one-minute load averages span **1.29–48.09**. Benchlock excludes cooperating agents' timing/GPU loads, not all background activity. These repeated game-move distributions support the speed observation, but not a controlled same-position speed ratio or an equal-time strength claim.

Paired games interleave arms move by move, with both engine colors played for each opening. Table timing is per-move median [p10,p90], not same-position AB benchmark latency: game trajectories and terminal/cache hits affect it. Equal-visit counts are exact; equal-time is the shared PUCT deadline, checked between logical batches, so one in-flight batch can overshoot. A 1600 ms LQO budget targets roughly 100 visits for the serial arm according to wave 1. Report actual realized visits rather than assuming that calibration still holds.

The queue holds `SP/benchlock` for nominal 16-minute chunks (13 minutes active deadline plus up to 3 minutes to finish/pause a ply), releasing it and sleeping 20 seconds between chunks. Browser/search pauses preserve in-flight game progress; disk checkpoints save completed games only. A retry restarts the browser and discards an incomplete game, not scores it as a loss. Missing/failed pause confirmation now throws through the driver's owned-browser close path before returning control to benchlock; it is never treated as proof of GPU idleness. Failed or unacknowledged closes retry until `closed: true` is received. A persistently broken browser-control channel deliberately holds benchlock and logs the need for recovery, even beyond the nominal chunk limit; manual recovery must stop that owned browser before terminating the driver. Killing a timed-out CLI is not shutdown proof. Model loading/probes are within the lock. The takeover queue PID was 90154, with log `artifacts/batching-strength/remaining-queue.log`; the inherited port-5187 server was PID 50735. They were already stopped at this closeout. The final idle-page smoke used owned server PID 69484, then stopped it and its browser; port 5187 has no listener.

**Trailing shell error:** the log ends with `QUEUE_DONE` followed by `scripts/batching_strength_queue.sh:96: parse error near ')'`. Both `zsh -n` and `bash -n` accept the persisted script, now covered by a no-execution unit test. The error happened after all matches exited and artifacts were checkpointed, so it is harmless to these results (but the historical queue's shell exit was not clean). A live-edit/reread-offset mismatch is a plausible explanation, not a proven cause; the queue now warns against editing its file while running and uses the explicit array element count. No queue was re-executed to diagnose it.

```sh
# Never npm install/ci in this worktree (node_modules is shared).
BUILD_SCOPE=product npm run build:client
PORT=5187 npm run web:isolated:static
# The script obtains benchlock itself. B2 is no longer in its default queue.
zsh scripts/batching_strength_queue.sh b1-v100 lqo-v256 b1-v400 lqo-time
# Read-only replay/statistical audit, no engine load:
node --experimental-strip-types scripts/analyze_batching_strength.mjs \
  artifacts/batching-strength docs/batching_strength_2026-10.json
python3 scripts/compare_batching_strength_models.py
npx biome format --write docs/batching_strength_2026-10.json
```

The queue command above is historical guidance, **not a request to resume these stopped runs**. With the final cardinality guard, today's small-net suite cannot meet a 100-opening request and fails before loading an engine. A future 16-opening diagnostic must explicitly choose `--opening-count 16 --games 32` in a new directory; a genuine 100-opening strength experiment needs a suitable equal-material suite first. No such experiment was run here.

Final build is seeded lab version `2026-10-13.2`; use **new output directories** for new runs. The driver refuses incompatible build/config checkpoints rather than mixing protocols. `--no-gate-stop` disables the nominal-CI early stop for a predeclared fixed-size match. To re-run a deliberately stopped match, use a new directory (do not resume abandoned B2). `--minutes`, `--games`, `--opening-count`, `--max-plies`, `--movetime-ms`, `--dir`, `--base-url`, and `--screenshot` are driver controls. A direct driver call must be wrapped in `SP/benchlock`.

## Verification

- `npm run typecheck`: clean, 0 errors/warnings after review fixes (`artifacts/batching-strength/review-2-fix-typecheck.log`).
- Full `npm test`: **1023 passed, 3 skipped, 0 failed** (1026 total; `artifacts/batching-strength/review-2-fix-full-tests.log`).
- Focused batching suite: **21 passed**; with PUCT core/ScLimit tests **23 passed** (`review-2-fix-focused-tests.log`). Tests cover pair likelihood, default physical batching, paired openings, exact logical-search equivalence at 8/32/100/400, seeded lab RNG, incompatible checkpoints/unsafe arguments, shell syntax without execution, partial/not-run evidence, PGN legality, actual small-suite shortfalls, repeated-opening sensitivity, the two-opening LQO boundary, failed/missing pause confirmations, and failed/unacknowledged browser shutdown.
- Read-only audit independently replayed **1580 non-pilot games**, checked opening/color/history pairs and exact equal-visit counts, and hashed all raw game/PGN files. The model protobuf comparison reproduced the identical learned-weight/quantization bytes. No new inference or games were used for either audit.
- `npx biome check` over all supported touched files: no errors/new issues; one pre-existing `tensorData` unused-function warning in `src/lc0/onnxEvaluator.ts` remains. Shell/Python/Markdown are not parsed by Biome; shell syntax was checked separately.
- `BUILD_SCOPE=product npm run build:client`: passed again after review fixes (`review-2-fix-build.log`). The initial closeout build also passed (`closeout-build.log`); its built-app **idle-only** browser smoke at `/lab/batching-strength/`: build `2026-10-13.2`, `crossOriginIsolated=true`, 0 games, no model/wasm requests or worker probes (`closeout-idle-browser.json`). Screenshot saved and read: `artifacts/batching-strength/screenshots/closeout-idle.png`. No autostart query was supplied.
- Historical built WebGPU pilot screenshots were also read: `artifacts/batching-strength/screenshots/pilot-b2-v8.png` and `pilot-lqo-v32.png`; they display strict WebGPU backends and correct physical sizes. They are **not** new seeded-run validation. The lab-only RNG is unit-tested/built but not validated by further GPU games, as the user explicitly stopped all matches.
- Autoreview-1's accepted findings corrected the 16-opening pseudoreplication and unsafe missing pause confirmation. Autoreview-2's accepted findings added acknowledged shutdown retries and fixed the two-opening LQO boundary. All have regression tests. Mandatory final-commit Codex branch review and any remaining accepted/rejected findings are recorded in scratchpad `batching-strength-review-N.{txt,json}` and `report_batching-strength.md`.

No manifest, model selection, product batch default, legal masking, queen-odds color-swap behavior, product contempt, or ONNX test tolerance was changed. Only lab overrides, harness/driver, evidence and tests ship. Nothing was merged, pushed, published or deployed.
