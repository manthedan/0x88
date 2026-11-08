# Arena simul feasibility gate (November 2026)

## Verdict

**GO for the bounded scheduling experiment, not for product promotion.** Four independent Lc0 Small games can share one worker/session and perform real physical batch-4 inference without changing their logical search. On this shared Apple M4 machine, N=4 completed **2.03–2.41×** as many games/hour as its interleaved batch-1 serial runs (three A/B pairs). All eight games in every simul trial exactly match the **same-export** serial control. N=2 also helps. N=8 helps against serial but shows no convincing advantage over N=4 and approximately doubles p95 move latency.

This is **not** equivalence to the tested **unquantized f16 batch-1** export: batch-N floating-point outputs differ and only 2/8 full games match it. Arena's shipped Small model is **batch1.f16.qdq8.onnx**, which this gate did **not** benchmark. The gains are conditional on the f16 exports, not a speed or strength claim against the shipped QDQ8 configuration. No Arena code, search numerics, model manifest, global batch default, Docs page or BT4 setting changed. Nothing here ships by default. No B1/B2/LQO strength run was resumed, and this is not strength evidence.

**An idle-machine re-measure is required before any promotion decision.** Four openings, eight-visit searches, repeated timing workloads and a 160-ply draw cap cannot establish general throughput or strength.

## Method

- `/lab/arena-simul/` is opt-in, unlinked from product navigation, and idle until Run gate. `arenaSimulWorker.ts` owns exactly one `Lc0OnnxEvaluator` session at a time. Each game has its own `Lc0PuctSearcher`, root, chronological board history and AbortController. No per-board network/session copies.
- `ArenaSimulBroker` sits below search. Each game admits one outstanding leaf; FIFO service prevents a fast game overtaking an admitted slow one. Queued **plus in-flight** slots are bounded at eight. A full batch dispatches on a promise turn; partial batches have a 1 ms coalescing deadline. This is not a wall-clock guarantee: a previous GPU run or a busy worker can delay service. Each physical call is awaited before another starts.
- Cancellation rejects a game's queued/in-flight promise promptly, discards stale readback, and leaves siblings running. Tokens include a page-generated run UUID and trial index. Stop aborts the trial; neither an interrupted game nor a partial trial is scored/published as completed performance. Failures stop admission, cancel siblings and await settlement before session disposal.
- Fixed settings: **8 visits/move**, per-game logical search batch **1**, pipeline **1**, temperature/drawScore/ScLimit/contemptElo **0**, fresh tree each move, default Lc0 cpuct/FPU schedules, no early stop or evaluator cache. Seed identity is **1** throughout; these settings do not sample RNG at all. A global `Math.random` override would be unsafe across coroutines and is not used. This tests deterministic Arena-style search, not Play's stochastic ScLimit settings or tree reuse.
- Openings are Italian, French Advance, King's Indian and English, retaining their legal book histories. Each has two games with A/B engine colours reversed (same-net self-play). Four actual distinct opening units, not eight independent samples; no cycling to pretend there are more openings. Every timing repetition intentionally repeats the same workload, not a new strength sample.
- Each N has **A/B/A/B/A/B**, eight games per row: A is serial physical batch 1; B is N simultaneous games. A final unpaired serial control uses B's exact model/export, padded to the same physical size, to isolate scheduling from export numerics. Session creation, fixture probes and one equal-visit warmup search precede each timing interval and are excluded. Rates include game/search orchestration, tails and broker waits. Games end by actual rules or 160 **searched** plies, then draw; no eval adjudication.
- Used existing local **unquantized f16** `t1-256x10-distilled-swa-2432500.batch{1,4,8}.f16.onnx`. There is no batch-2 export, so N=2 physically uses 4 with padding. Strict `webgpu -> webgpu`, no WASM fallback. Every returned evaluation checks `ortPhysicalBatchSize` and row-zero markers: a broker dispatch must be one physical run, not N serial batch-1 runs. Batch 16/32 and BT4 were not tested.
- GPU occupancy = real evaluated rows / physical slots, including tails/padding. NN evals/s excludes padding; visits/s counts committed search visits. p95 is the nearest-rank p95 across per-game searched-move wall latencies in that trial, including broker waiting. Three-row spreads are ranges, **not confidence intervals**.

## Results

Values are **median [minimum, maximum]** of the three paired timing rows per arm. Games/hour is aggregate completed games, including explicit ply-cap draws.

| Gate / arm | Games/hour | NN evals/s | Visits/s | Physical occupancy | p95 move ms |
|---|---:|---:|---:|---:|---:|
| N=2 serial | 275.7 [271.0, 304.0] | 57.8 [56.8, 63.8] | 52.5 [51.6, 57.9] | 100% | 185 [183, 189] |
| N=2 simul, physical 4 | 367.9 [361.8, 386.2] | 100.6 [98.9, 105.6] | 90.6 [89.0, 95.0] | 50% | 211 [197, 217] |
| N=4 serial | 285.6 [273.9, 291.1] | 59.9 [57.5, 61.1] | 54.4 [52.2, 55.5] | 100% | 173 [171, 183] |
| N=4 simul, physical 4 | 649.2 [592.3, 687.2] | 177.5 [161.9, 187.8] | 159.8 [145.8, 169.1] | 86.1% [86.0, 86.1] | 206 [182, 225] |
| N=8 serial | 289.2 [280.3, 393.8] | 60.7 [58.8, 82.6] | 55.1 [53.4, 75.0] | 100% | 168 [161, 179] |
| N=8 simul, physical 8 | 635.9 [522.2, 657.7] | 173.8 [142.7, 179.8] | 156.5 [128.5, 161.9] | 46.2% [45.6, 49.8] | 430 [387, 512] |

Adjacent-pair games/hour ratios: **N=2: 1.211, 1.335, 1.401**; **N=4: 2.034, 2.406, 2.370**. **N=8: 1.615, 1.863, 2.274**. Ratios of arm medians are 1.33×, 2.27× and 2.20× respectively. N=8's early-finishing games leave a poorly filled tail; its median aggregate rate does not exceed N=4's, though these separate noisy gates are not a controlled N=4-vs-8 regression test. Increased move latency is a tradeoff, not a regression hidden by reporting only aggregate throughput.

N=8 was unlocked only after the full N=4 gate showed >10% median gains in **both** games/hour and NN evals/s, with all simul sequences matching its batch-4 serial control. Eligibility matches visits/ply cap and physical/model identity, not any convenient shorter pilot.

Memory is measured separately **after all timed rows**, with the same-export control session still loaded. `measureUserAgentSpecificMemory` includes page, dedicated worker and WASM, not GPU residency; it is **not peak active-search memory**. Loaded-session samples: **N=4: 144.2 MiB; N=8: 144.4 MiB**. The initial N=2 timing build did not take this sample; a separate, excluded **8-visit / 2-ply** UI follow-up measured **143.6 MiB** with its physical-4 control loaded. These post-trial numbers do not establish equal peak memory at higher concurrency. GPU `createBuffer` request volume is retained separately and is not called live/peak GPU memory: native releases/GC need not invoke JS `destroy()`. There is no reliable GPU residency measurement here. A pending GC-backed memory promise blocks later gates even after Stop or client-side navigation.

## Numerical and game audit

The existing protobuf comparison script confirms 552 identical serialized graph nodes and 375 matching initializer names for both batch 1 vs 4 and batch 1 vs 8. **374 initializer payloads, including learned weights, are byte-identical**; only `/const/pos_encoding_shape` and graph shape metadata differ. Files are 40,491,226 bytes each; hashes are in the compact JSON evidence.

On the four fixture histories, batch 4 differs from batch 1 by at most **0.00244140625 WDL** and **0.0071360062 legal prior**. Those small differences can redirect later PUCT selections. Grouped vs single-row padded evaluation on the **same export** has zero observed fixture drift. Full same-export game equality is the stronger check:

- N=2, N=4 and N=8: every simul repetition matches **8/8** same-export serial-control games, including all searched moves, visits, results and termination reasons. Only **2/8** match the batch-1 trajectory; serial repetitions are themselves stable.
- Batch 1 searches 686 plies per eight-game trial; batches 4 and 8 search 886. Thus games/hour alone understates its inference speedup here; it is **not** a fixed-work or strength comparison between exports. Exact-visit and same-export controls prevent treating changed game length as a concurrency win.
- The read-only analyzer independently replays **168 complete games / 16,806 searched plies / 134,448 visits** (144 paired timing games plus 24 control games). Endings: 84 threefold, 42 checkmate, 42 explicit ply-cap draws. It replays all book/search moves, verifies rule endings and final FENs, rejects premature cap draws/illegal moves, checks exact visits and physical/model identities, and records per-game move hashes. Unit tests also run identical deterministic evaluations serially vs N=2/4/8 and prove per-game sequence/visit/result equality.

## Reproduction and limitations

```sh
BUILD_SCOPE=product npm run build:client
node scripts/serve_isolated_static.mjs dist-client 5198
# Open /lab/arena-simul/, keep 8 visits / 160 plies, Run N=2 then N=4.
# N=8 is enabled only if the matching N=4 gate helps. Export evidence before leaving.
node --experimental-strip-types scripts/analyze_arena_simul.mjs \
  docs/arena_simul_gate_2026-11.json artifacts/arena-simul/n2.json \
  artifacts/arena-simul/n4.json artifacts/arena-simul/n8.json
```

The analyzer accepts one run object or the browser's exported run array. Raw games are local ignored artifacts; committed compact evidence includes raw SHA-256 hashes, all row metrics and per-game move hashes. Timing used the real built static app on port 5198, not a synthetic logical batching benchmark.

Discarded smoke/calibration runs are not mixed with the gate. In particular, an N=4 pilot that sampled memory between A/B rows was stopped and excluded: a timed-out UASM call can still trigger GC later. Final timing runs never interleave memory sampling with timing. N=2 predates the memory-telemetry correction but uses the identical search, model, broker and game protocol; its old GPU live/peak counters are intentionally omitted from the evidence summary.

Shared-machine load, a small deterministic self-play suite, short searches, capped draws, missing active-search peak/GPU residency measurements and no device-loss recovery remain limitations. There is no background-tab completion promise, no equal-time strength claim, no stochastic-search equivalence claim and no default rollout. Keep CPU parallel Arena unchanged. Prefer N=4 as the next idle-machine investigation, not a user-facing switch.

## Verification

- Baseline: `npm test` **1,230 passed, 3 skipped, 0 failed** (1,233 total), before edits; typecheck 0 errors/warnings.
- Final `npm test -- --test-concurrency=4`: **1,244 passed, 3 skipped, 0 failed** (1,247 total). An earlier full run during concurrent GPU/review load hit an unrelated `engine_upgrade_audit` mock-process timeout; its complete rerun passed without modifying that test. Final `npm run typecheck`: **0 errors/warnings**; `BUILD_SCOPE=product npm run build:client` passed. Logs: `/tmp/arena-simul-{full-test-passed,typecheck7,build-final}.log`.
- Focused lab suite: **14 passed**, covering broker ordering/bounds/fairness, queued/in-flight cancellation, stale generations, Stop between physical probes, failure admission, per-game equivalence, incomplete-result rejection, memory timing isolation/remounts, workload/model eligibility and exported evidence replay.
- Built-app browser verification: strict WebGPU full N=2/4/8 gates, same-export controls, real Run/Stop/Cancel buttons, cancelled-game suppression, restart, parked N=8, and a separate memory-only UI follow-up. No model loads on idle lab entry. The unchanged Arena page was also checked: Lc0 Small + Stockfish Lite, no simul entry point (`arena-unchanged.png`). Screenshot files were saved, verified and viewed at 1440×900 and 390×844, light/dark: `/Users/macthedan/projects/lc0_browser/notes/ui-hillclimb/features/wave2/shots/arena-simul/{desktop,mobile}-{light,dark}.png`. Browser export emitted a download event, but agent-browser could not verify its requested destination (upstream artifact limitation); raw evidence was instead captured directly from the lab API, and the exported-array format is CLI/unit-tested.
- Codex branch autoreview: seven rounds; thirteen accepted findings fixed, none rejected; round seven clean. Findings tightened cancellation, comparable-work/export checks, memory measurement integrity and evidence handling. A final test assertion was broadened to accept either of two valid identity-rejection boundaries; no runtime change followed the clean review.
