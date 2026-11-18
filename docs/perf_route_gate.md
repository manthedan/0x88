# Real-route performance and memory merge gate

**Every refactor branch must attach a `perf:routes` comparison against main before merge. Any regression beyond noise requires explicit owner sign-off.** A noisy, incomplete, failed, subset-only, or unavailable-browser run is not a clean merge gate. Do not substitute the single-net lab matrix or CPU canary for this route gate.

This protects root-scoped DOM lookups, per-mount controllers, Svelte forms/engine cards, and SvelteKit upgrades against making the real application slower or larger. No production page/controller behavior is changed by the harness.

## Run on built product previews

```sh
# From the branch worktree, with local, manifest-pinned model/runtime assets staged:
npm run perf:routes -- --baseline main --candidate HEAD \
  --browser all --repeats 5 --benchlock /absolute/path/to/machine-wide/benchlock \
  --out .local-dev-artifacts/perf-routes/main-vs-branch

# Or supply two already built directories. Never use a Vite development server.
BUILD_SCOPE=product npm run build:client
npm run web:isolated:static
npm run perf:routes -- --baseline /path/to/baseline/dist-client --candidate dist-client \
  --browser chrome --benchlock /absolute/path/to/benchlock

# Print/validate the complete interleaved plan, without building or writing:
npm run perf:routes -- --baseline main --candidate HEAD --dry-run
# Regenerate verdicts without opening browsers or discarding failed attempts:
npm run perf:routes -- --report .local-dev-artifacts/perf-routes/main-vs-branch
```

Commit inputs are resolved to immutable SHAs, archived and built under the **current worktree's** ignored output directory with `BUILD_SCOPE=product npm run build:client`. Neither source checkout nor another worktree is modified. The build is the same product build served by `web:isolated:static` (`scripts/serve_isolated_static.mjs`), not a benchmark-only page. A self-comparison builds once and serves identical bytes on two origins. Uncommitted source edits are not part of `HEAD`: commit first, or explicitly pass the newly built directory.

Same-lockfile archives use the worktree's installed dependencies read-only; changed lockfiles get an isolated `npm ci --ignore-scripts`. Historical tracked assets/manifests are preserved; missing ignored model assets are linked from existing local sources and still undergo the app's normal pinned verification. Assets must stay same-origin through the proxy: a `VITE_LC0_BROWSER_ASSET_BASE_URL`/other asset-hosted build with traffic bypassing it is incomplete, never a JS-byte saving. Main fetch/resource URLs, worker bootstrap/init/ready URLs, and Chrome context requests detect such traffic; stage the pinned assets locally rather than suppress that check. Keep dependencies, model bytes, hardware, browser, viewport and settings fixed unless that is the treatment. `builds.json`, session specs, ready messages, UA, process IDs and host load retain provenance. Externally served URLs must be origins (no path prefix, query or fragment). They are operator-supplied build identities: preserve their build manifests alongside the results.

### Mac mini transport

Reuse `scripts/perf_matrix/mini.sh`, a **branch-isolated** remote workspace, separate local state, and the existing shared `~/perf-matrix/safari.lock`. Do not overwrite another agent's scripts. Never remove a stale/foreign mutex, delete someone else's WebDriver session, restart a shared driver, or kill by process pattern.

```sh
# Prepare production source independently of measurements:
npm run perf:routes -- --baseline c808bad0 --candidate c808bad0 --build-only \
  --out .local-dev-artifacts/perf-routes/production-build

export PERF_MINI_WORKSPACE=perf-route-gate
export PERF_MINI_STATE="$PWD/.local-dev-artifacts/perf-routes/mini-state"
scripts/perf_matrix/mini.sh up \
  "$PWD/.local-dev-artifacts/perf-routes/production-build/builds/baseline/dist-client" 62911 18911
scripts/perf_matrix/mini.sh run-routes production-self-safari \
  --baseline c808bad0 --candidate c808bad0 --browser safari --repeats 5
scripts/perf_matrix/mini.sh fetch production-self-safari \
  .local-dev-artifacts/perf-routes/production-self-safari
scripts/perf_matrix/mini.sh down
```

For different builds, run `up` with **two distinct local state directories, local ports and mini ports**, then pass `--baseline-url http://127.0.0.1:<baseline-mini-port> --candidate-url http://127.0.0.1:<candidate-mini-port>` to `run-routes`. Both servers remain live; only the measured session is serialized. Default URL flags intentionally use the same upstream for a self-comparison.

Safari needs only Node, clang/pidfp and the existing driver (`4446` on the mini); no browser is installed or driver started. It must have an **unlocked, visible automation window**. The harness activates Safari only after acquiring its own session, records visibility changes, and rejects any hidden interval throughout the natural journey/settle/residual windows, even if visibility is restored before the final poll. A locked console is an external blocker; do not falsify visibility, disable the app's hidden-tab pause, or unlock it unattended. Headless Chrome can run on the locked mini: provision `playwright` and `playwright-core` matching the worktree under the isolated workspace's `node_modules`, and use `--browser chrome`. It launches its own installed Google Chrome, never a user's browser/profile. Chrome and Safari both hold the same machine mutex for the entire measured session. Capture laptop load as well when it serves mini assets.

## Journeys (fresh browser session/origin each time)

1. **Analysis default:** cold mid-game FEN URL, Lc0 Small + SF Lite default rows; auto-analysis on, real user-move hook while analyzing, then a 60 s observation window starting at the move. Defaults are finite search budgets, not an artificial infinite search. Within that window require a worker progress message or result for the exact post-move FEN and visible PVs for every selected engine (a zero-new-visit reused-root response proves the request but adds no speed sample).
2. **Analysis four:** same, add two rows through the UI and select Berserk and PlentyChess (CPU); after 60 s, make ten further legal user plies, waiting for FEN-associated worker progress and every engine's new PV after each.
3. **Play Maia3 / Small:** separate ten-full-move games (20 plies) as White at the UI's fastest strength. A deterministic host-side three-ply material/development chooser avoids trivial early checkmates, adding no browser engine/worker. It replays committed SAN, only runs on the user's turn and must match the board's legal destinations (including back-rank castling/rook moves). Real engine replies are required. An early terminal game is invalid, not silently restarted.
4. **Arena CPU / Small:** one completed SF Lite–Berserk game, then a separate Small–SF Lite game, 100 ms equal movetime, one game/opening, concurrency one. Require a completed game chip/log; errors/forfeits/timeouts, illegal replies, missing-move resignations and cancellation are invalid.
5. **Navigation:** Analysis → Play → Arena → Analysis via actual header links, three cycles, then 20 s settle. Verify the document identity survives every transition; a hard reload cannot masquerade as client-side cleanup.

Every journey has a fixed **20 s settle**, then a header navigation to **Docs** and another **20 s residual window** without destroying the document. Require Docs' rendered heading/section markers, not merely `/docs/` in the address bar; unexpected HTTP failures, transport failures, and unfinished requests through the residual window invalidate the run, while Docs transfers stay separate from the journey's JS-byte metric. Session cache storage is cold by construction (new ephemeral browser session plus fresh counting-proxy origin); navigation cycles intentionally retain that session's warm caches. No forced GC or pressure notification occurs during these natural footprint windows, and no backend override or engine implementation is substituted. The optional UA-memory API is called only afterwards, in a separately labeled GC-affected phase.

`--journey NAME` (repeatable) is diagnostic only. A full comparison has seven journeys × two browsers × two arms × at least five repeats. Repeats are round-robin across journeys/builds; arm order reverses every repeat. No selective rerun/removal of outliers. Each failed attempt remains visible. Output directories are immutable; use a new one for a new balanced cohort.

## Measurement contract

Reuse `perf_matrix/common.mjs` (kernel `pidfp` / `proc_pid_rusage`, worker-ready observation) and its counting proxy. An early HTML probe observes actual workers, progress and rendering before hydration; the product's CSP/configuration on disk is unchanged. The proxy requests identity encodings so JS body bytes are comparable. The same probes/viewport/Chrome flags are applied to both arms.

- **Memory (MiB = 2²⁰ B):** Chrome samples only its owned instance's renderers and GPU PIDs from browser CDP, every ~100 ms. `memory.peakMb` is the largest **simultaneous sampled renderer+GPU sum**. Separately report per-process kernel interval high-water renderer/GPU peaks: these may occur at different times and **must not be summed and called a simultaneous measurement**. Renderers are summed, not just the largest renderer. Missing Chrome GPU or renderer readings invalidate the attempt. Safari requires exactly one new WebContent PID after starting the owned session, and uses that process's kernel high-water (including navigation before PID discovery); reused or ambiguous PIDs are invalid. Safari's shared GPU process is deliberately not attributed. Kernel counters catch short process spikes; sampled simultaneous totals can miss sub-interval combined spikes.
- **Settled/residual:** median last two seconds of each fixed 20 s window, not proof of a plateau or reclaimed native/GPU allocations. Residual includes caches/retained runtime state after client-side navigation to Docs. Retained Play Maia worker(s) are measured, not excused away.
- **UA memory:** Chrome attempts `performance.measureUserAgentSpecificMemory()` **only after all natural footprint/worker/long-task/input/speed observations are frozen**, recording `memory.ua.postResidualGcMb`. The [API triggers garbage collection](https://web.dev/articles/monitor-total-page-memory-usage#local-testing), made eager by `ForceEagerMeasureMemory`; measuring before settling/navigation would change the very retention being tested. This post-residual, GC-affected page/worker JS/WASM value is not the natural settled footprint or a GPU/native total. Retain unsupported/errors/5 s timeouts explicitly, never zero; active-page UA snapshots need a separate diagnostic cohort.
- **Workers:** live and peak main-realm-created worker counts, decrementing once on terminate/disposed. Nested/pthread worker creation is not visible to this injected wrapper; process footprint still includes it. Worker counts do not prove native reclamation. Ready messages retain backend/IO/model provenance, not model buffers.
- **Speed:** median observed progress visits/s or UCI NPS, separately per engine. Worker Lc0 rates use only the current search's `completedVisits` (or final `stats.completedVisits`), never inherited root visits divided by the new search's elapsed time. Play Small uses rendered current-search progress. Arena Small derives completed work from the card's total visits minus the diagnostics' per-search reused-visits delta, divided by its last search time (rounded UI counters/time); it does not trust the card's reuse-inflated Visits/s. A skipped telemetry search contributes no rate, never a fabricated one. Maia3 uses worker evaluation round-trip evaluations/s (not NPS). This is route workload throughput, not the lab's eight-visit smoke or a strength benchmark. Missing speed for any selected engine makes the gate incomplete.
- **First line / interactivity:** early probe timestamp to first engine output/progress (including initial loading) and to board DOM plus installed page controls/hook. Board interactive does **not** mean every model is loaded. Both timings are in-page, excluding HTML network time before probe installation.
- **Long tasks:** supported `PerformanceObserver('longtask')` entries on the Analysis route: count, total ms, p95. Safari's missing API is unavailable, not zero. The probe itself has equal overhead in both arms.
- **Input:** scheduled real user-move hook to two animation frames, including main-thread queue/handler/render delay; not hardware-to-screen latency or a trusted physical drag. Record/verify the changed FEN or committed ply.
- **JS:** response body bytes for `.js`/`.mjs`, including worker/glue scripts, excluding WASM/models/CSS. Per-route attribution uses request initiation times (cached repeats may transfer zero); navigation also reports the journey total. It is transferred JS for this route journey, not a static import-graph estimate. Keep raw request logs. Confirmed 404/410 responses from optional model, engine-runtime, or logo HEAD probes, concurrent sibling probe cancellations after a confirmed missing runtime asset, and absent or browser-cancelled speculative Reckless requests are retained separately, as are cancelled Viridithas prefetches in Analysis Four. Neither engine is selected by a gate journey, and every selected engine must produce speed/progress.

## Verdict / owner sign-off

For each browser/journey/metric, retain both arms' **unrounded** samples and compute median, MAD, min/max. Let `noise = max(3 × baseline MAD/|median|, 3 × candidate MAD/|median|)`.

- Memory and speed floor: **5%**; long-task floor: **10%**. Timing, bytes and worker counts use 5%.
- A regression must exceed **max(floor, noise)** in the adverse direction. Lower speed is adverse; higher memory/timing/tasks/bytes/workers is adverse. Equality at the boundary is within noise.
- If either arm's own MAD/|median| exceeds the metric's floor, report **noisy**, not a performance pass. Run a new balanced cohort on a quieter host.
- At least five valid samples **per arm per metric** are required. Missing required metrics, failed journeys or partial cohorts are incomplete. Both-zero baselines are unchanged; zero→positive cost is an absolute regression, not division-by-zero or an automatic pass.
- Exit **0** means clean measured cells, **1** regression, **2** incomplete/noisy. A subset or missing browser still needs its remaining coverage before merge. Owner sign-off must name the adverse metric, magnitude and accepted tradeoff; a signed-off regression is not relabeled an improvement.

Raw specs, full time series, process/context/worker evidence and immutable failure records stay under `.local-dev-artifacts/perf-routes/`. The coordinator appends completed attempts to `runs.jsonl`; `--report` regenerates `summary.json` and `summary.md` from that index. Attach the summary (and local evidence location), not model binaries or browser profiles. Nothing is pushed, deployed or published by the runner.

### Known limitations

- `--report` does not reconstruct a missing or incomplete `runs.jsonl` from per-session files. If the coordinator stops before indexing a session result, that result remains on disk but is absent from regenerated reports.
- On SIGINT or SIGTERM the coordinator stops scheduling attempts. It waits for the owned session to clean up before releasing a verified `--lock-dir`. If browser closure cannot be confirmed, it retains that mutex for owner recovery and records a `.cleanup-pending` marker next to the session result. With an external `--benchlock` wrapper it lets the current invocation finish, but a queued session will not open a browser after cancellation. It retries deletion of a known owned Safari session if needed. If recovery requires manual intervention, verify the owned browser is closed before removing its `.cleanup-pending` marker to let the wrapper finish.

## Initial production baseline

Production source is **`c808bad0e2955d3841daff5372ee0c0598ec4d56`**, built from its archive, not a later refactor. Local build: `.local-dev-artifacts/perf-routes/production-build/`. Initial smoke diagnostics are excluded from the balanced cohort. The first calibration cohort (`production-self-chrome`) was stopped as a whole after calibration exposed weak move selection and a turn predicate that stalled when the human was in check; every attempt is retained with `ABORTED.txt`. Its owned coordinator was stopped, the active browser child was allowed to finish/clean up, and only that coordinator's lock was removed after ownership verification. A second entire calibration cohort (`production-self-chrome-v2`) was stopped after verified review found probe-owned Play DOM retention and inherited-root visits in worker throughput. Those attempts are also preserved, not used as the corrected baseline. The replacement cohort starts from zero with the corrected probes/chooser; no successful outlier is selectively dropped.

The production self-comparison uses the **same built bytes** in both arms, five independently measured repeats per arm/journey, interleaved/reversed order on the M4 Mac mini's headless Chrome. Its baseline arm is the production baseline; the candidate arm independently estimates the gate's self-comparison false-positive rate. See the attached `summary.md`, not a fabricated duplicated baseline dataset.

Safari 27 on the mini was attempted for all seven journeys but **externally blocked by the locked console**: WebDriver and page initialization succeed, `CGSSessionScreenIsLocked=Yes`, and `document.visibilityState='hidden'`. Those are invalid observations, not Safari baseline numbers. No lock-screen/security setting was changed, no foreign browser/driver was terminated, and no iPhone was touched. Unlock/foreground the mini and run the same balanced Safari comparison before claiming complete two-browser coverage.
