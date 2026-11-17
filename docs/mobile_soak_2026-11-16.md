# Mobile soak lab — 2026-11-16

Route: `/lab/mobile-soak` (included in the product build; deliberately absent from navigation).

The original measurements below are desktop evidence. A subsequent **owner-reported real iPhone run (Safari 26.6 / iOS 26.6)** passed all CPU engines, Lc0 Small, Maia3, Centipawn and t3 (171MB plain f16, split-K). LQO (`lqo_v2.f16.qdq8.onnx`, reported 96MB int8 QDQ) killed the tab in **2 of 3 trials**, roughly **8–10 seconds** into its step. BT4 (policytune-332 batch8 f16 QDQ, reported 187MB) killed it during load. These phone results were provided by the owner, not independently collected by this agent. The exact memory-kill mechanism remains a hypothesis; neither file size nor a passed desktop run establishes peak mobile memory usage.

## Running / automation

Open the page and tap **Start**, or use:

```text
/lab/mobile-soak?auto=1&steps=sf-lite,lc0-small,t3,heat&heatSeconds=120
```

Without `steps`, all 15 steps run in catalog order. Selection preserves URL order, removes duplicates, and reports unknown IDs in the event log. `heatSeconds` defaults to 120 and is clamped to 1–1800 integer seconds. On a local build without staged models, append `&assetBase=https%3A%2F%2Fassets.0x88.app` (the existing production asset resolver, integrity verification and cache paths still apply).

`window.__mobileSoak` exposes live `state`, `results` (the whole JSON report), readonly `unsafeDisposal`, async `start(stepIds?)`, async `continueAfterCrash()`, and `reset()`. Running calls cannot overlap, including across route remounts waiting for retired cleanup. Reset/Start are ignored during a run; Start cannot bypass a pending crash choice. A successfully disposed completed run can be started again. If any disposal rejects, `unsafeDisposal` stays true for the entire document lifetime: Start, Continue and Start over are disabled/no-ops, an alert asks for a reload, and Copy results remains available. Leaving/reopening the route cannot reset this latch; a full document reload is required because failed production release promises cannot be assumed retryable. The latch is live document ownership state, not a saved flag that would incorrectly block a fresh document. Auto mode starts a new run, or resumes after an interrupted step without retrying it, unless ownership is unsafe or its controller has been destroyed. Export results before starting over.

Stable selectors:

- `#mobile-soak-start`, `#mobile-soak-continue`, `#mobile-soak-reset`, `#mobile-soak-copy`
- `#mobile-soak-status`, `#mobile-soak-env`, `#mobile-soak-results`
- `#mobile-soak-unsafe-disposal` / `[data-testid="unsafe-disposal"]` when teardown has failed
- `#mobile-soak-step-<id>` / `[data-testid="step-<id>"]`
- Button test IDs: `start`, `continue`, `start-over`, `copy-results`; status: `status`.
- Step rows expose `data-status`; the current step's result status is genuinely `running` in the API, JSON, storage and UI throughout load/search/disposal. Terminal status is assigned only when its in-progress marker is cleared. Reload recovery still replaces any marked step with `crashed`, including old saved reports that pessimistically used `failed`.

Poll the API rather than assuming a fixed load delay. Stockfish and WASM handshakes use their production deadlines; neural initialization/inference can take much longer on a phone. There is no lab timer that abandons a pending load and starts another resident engine. A hung browser/runtime may require a reload; that interruption is captured by the breadcrumb.

## Steps and production paths

Ordinary steps prepare/prewarm, search **startpos** and **Kiwipete** (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`), then dispose. CPU searches request 1,000ms; tree-search nets request eight visits per position. These are smoke/stress workloads, not engine-strength benchmarks. For runtimes that actually prewarm, load time includes download/cache access plus initialization, but excludes environment/memory probes. Deferred CPU setup instead has `loadDeferred: true`, `loadMs: null`, and a measured `prepareMs`; its first init-plus-search is recorded separately, as detailed below.

| ID | Work and production module |
| --- | --- |
| `sf-lite` | `StockfishEngine` / `stockfishFlavorUrl` in `src/lc0/stockfishEngine.ts`; Play's single-thread Lite choice, feature-detected relaxed SIMD, 64MB hash. |
| `sf-full` | Same adapter, Arena's auto balanced thread budget via `resourceBroker.ts::cpuThreadBudget`, clamped to 32; threaded only with isolation + SAB, otherwise single; 64MB hash. |
| `reckless` | `engineProvision.ts::createDefaultBrowserUciEngine`, including the shipped variant/asset fallback; `recklessEngine.ts`. |
| `viridithas` | Same production provisioner; `viridithasEngine.ts`. |
| `berserk` | Same production provisioner; `berserkEngine.ts` / Emscripten UCI adapter. |
| `plentychess` | Same production provisioner; `plentychessEngine.ts` / Emscripten UCI adapter, including its large data sidecar. |
| `stormphrax` | Same production provisioner; `stormphraxEngine.ts` / Emscripten UCI adapter. |
| `lc0-small` | `lc0SmallModelSelection.ts::createLc0OnnxEvaluatorFromUrl`, cache + device model selection enabled as on product pages; `onnxEvaluator.ts` and `search.ts::Lc0PuctSearcher`. |
| `lqo` | `bt4Engine.ts::BIG_NETS.lqo`, production asset probe, `bigNetSessionPool.ts::acquireBigNetSearcher`, `Bt4WorkerSearcher` / `searchWorkerCore.ts`. Standard positions intentionally test the odds net even without queen odds. |
| `t3` | Same keyed production pool/worker, `BIG_NETS.t3`; normal Apple split-K selection and fallback. |
| `bt4` | Same keyed production pool/worker, `BIG_NETS.bt4`. |
| `maia3` | `maia3.ts::Maia3BrowserEvaluator.create`, including graph/QDQ/fp16 fallback and worker path. Maia is a policy engine: one inference per position, measured as **evaluations/s**, not invented tree visits. |
| `centipawn` | `nn/browserRuntimeEvaluator.ts::createBrowserSquareformerRuntimeEvaluator`, current s4fix-v-ema model/meta, `runtime=auto`, fallback enabled; `search/puct.ts::searchRoot` with `search/centipawn.ts` options. Same default zero-cache workload as Arena, without needing a page-owned cache to retain it. |
| `arena` | Acquire **BT4 and t3 in the same production keyed pool** and keep both initialized while an independent full Stockfish worker is also resident. Search all three concurrently for each position; await all seats even on a failure. This is deliberately harsher than ordinary Arena turn-taking, not an alternate loader or a claim that normal Arena always runs three simultaneous searches. |
| `heat` | Reload the strongest successful net in this run (BT4, t3, Centipawn, Small, LQO, Maia, in that priority; successful Arena residency also qualifies). One sustained middlegame search with a large visit ceiling and requested movetime. Maia-only heat is repeated production policy inference. Progress yields approximately 10-second slices, with actual durations/count deltas and a final partial slice. |

Every engine is owned before initialization begins. Teardown is awaited before the marker is cleared or another step starts. CPU and big-net worker teardown is synchronous `Worker.terminate()` at the production API boundary; Small, Maia and Centipawn release asynchronously. If disposal rejects, the step fails and remaining steps are skipped rather than risking overlapping residency. A document-wide unsafe-disposal latch also blocks all subsequent runs and resets until a full reload; returning to the route is not sufficient.

Missing assets detected by the production big-net probe are skipped with their URL/reason; load/search failures are retained as failed steps and do not erase earlier results. Heat is skipped if no net succeeded earlier. No catalog step had to be omitted or skipped in the successful desktop catalog runs. A separate heat-only selection verified the explicit no-prior-net skip reason.

## Environment, memory and crash semantics

The report collects UA, isolation, SAB, hardware concurrency, nullable device memory/storage estimate, high-performance WebGPU adapter info, features (including shader-f16), and buffer/storage/workgroup limits. Adapter field selection follows `lab/webgpu-lc0-diag`; no extra diagnostic GPU device is created. New runs refresh environment data; resumed results retain their original environment.

Memory samples now carry phases: `before-load`, `resident-after-load` (or `after-prepare` for deferred CPU setup), `after-search`, `after-dispose`. A no-op prewarm sample is never labeled resident. Chrome JS heap fields are copied when exposed. UA-specific memory sampling is attempted only when available and isolated, with a two-second diagnostic timeout. After a timeout it is disabled for the remainder of that page lifetime, avoiding a growing pile of GC requests. GPU memory is not covered reliably. Worker WASM heaps are private in these shipped adapters; `wasmHeapBytes` is explicitly null. Safari can legitimately have null for every memory field.

Actual ORT `env.webgpu.device.lost` is observed (not the diagnostic adapter). Big-net and Maia workers forward reason/message as unsolicited notifications **without request IDs**; main-thread Small/Centipawn use the same observer. Each ORT device gets one lost-promise listener with removable subscriptions, and each worker owns a single refreshable monitor rather than registering a new callback on every re-init. Loss marks a completed step failed. No real device loss was observed on this Mac; the notification payload/unsubscribe behavior is unit-tested with a controlled device.

One `localStorage` value (`0x88-mobile-soak-v1`) contains the entire report plus the in-progress marker, written **before** each step allocates/loads. Partial search/heat results are checkpointed. On recovery the marked step becomes `crashed`, previous results remain, reload count increments once, and Continue advances beyond it. Pagehide/visibility events are retained in a bounded 100-event log. A detached pagehide snapshot prevents worker rejection/cleanup during unload from clearing the stored marker, including when Safari has no asynchronous memory probe to delay cleanup. BFCache restoration unfreezes the live runner. A reload during environment collection or exactly between steps has no engine marker: it restores idle/done rather than stranding a non-existent runner or attributing an engine crash.

Destroying a controller immediately disables both persistence and UI publication, removes its document-safety subscription, and prevents pending auto-start continuations or stale API calls from starting/resetting runs. Pending environment probes and asynchronous cleanup may still settle, but cannot overwrite a reopened controller's report or crash marker. A document-scoped `ExclusiveQueue` keeps a reopened controller's allocations behind retired cleanup; if that cleanup fails, the unsafe-disposal latch notifies the mounted UI and prevents its queued run from allocating engines. Mounted pagehide snapshots still remain frozen through cleanup and unfreeze on BFCache pageshow.

A breadcrumb proves interruption, **not** an OS OOM: manual reload/navigation and a killed tab are indistinguishable. All storage reads/writes are caught; denied/quota storage leaves an in-memory runner and a `storage-unavailable` event. No account, upload or external results service is involved.

## iPhone diagnostic controls

Plain-f16 LQO trial (one variant per run, no suffixed step IDs):

```text
/lab/mobile-soak?auto=1&steps=lqo&lqoModel=/models/lc0/lqo_v2.f16.onnx
```

Lab-only `lqoModel`, `t3Model` and `bt4Model` accept a path under `/models/lc0/` or an absolute URL on exactly `https://assets.0x88.app`. Other origins/schemes, credentials, fragments, control characters and path escapes fail closed with an error; they never silently load the shipped QDQ net. Relative paths use the existing production asset-base resolver. Manifest/hash verification is unchanged: an uncatalogued or unavailable model fails/skips with its actual reason, with no unverified-loader escape hatch.

Overrides use an isolated instance of the **production `BigNetSessionPool` / `Bt4WorkerSearcher` / search-worker path**, even when the supplied URL spells the shipped URL. They cannot collide with a Play/Arena/Analysis seat or another diagnostic variant. Overrides are saved with the run, preserved by API step selection and crash continuation, and applied to the corresponding Arena/heat net too. Runtime includes `requestedModelUrl`, actual `servedModelUrl`, `modelOverride`, `registeredSplitKUrl` and `splitKApplied`. The production Apple-WebGPU split-K rule remains enabled; an unregistered filename (including plain `lqo_v2.f16.onnx`) explicitly reports that no production rewrite exists. Step-label size estimates still describe the shipped model, not the override.

`ortGraphOpt=disabled|basic|extended|layout|all` chooses ORT `sessionOptions().graphOptimizationLevel`; absent/invalid values retain **all**. Effective `graphOptimizationLevel` is included in ORT runtime diagnostics and the worker's ready payload, visible under each neural step's `runtime.<engine>.ortRuntime`. Big-net and Maia workers receive the option explicitly because worker URLs do not inherit the page query. Other search-worker entrypoints forward the same diagnostic options. Cached big-net initialization is invalidated if graph/folding settings change. Requested model/ORT settings are checkpointed before load; `ortRuntimeSource` distinguishes pending requested settings from completed session diagnostics if the phone dies during load.

**Important independent control:** this repo already performs an app-side JavaScript weight-only `DequantizeLinear` folding pass in `ortRuntime.ts::prepareOrtModelInput`, enabled by default. It materializes plain initializers **before** ORT session creation and can grow the model. Disabling ORT graph optimization alone does not disable that pass. To test QDQ without either optimization layer, use:

```text
/lab/mobile-soak?auto=1&steps=lqo&ortGraphOpt=disabled&ortDequantFold=0
```

The existing `ortDequantFold` choice now reaches big-net and Maia workers as well. Its default remains enabled; diagnostics report `dequantFoldEnabled` and the fold summary, including before/after model bytes when available. The owner's theory that ORT constant folding expands QDQ weights is **not established by these changes**. Disabling both layers is an experiment, not a guarantee that ORT/provider initialization retains int8 weights exclusively on GPU or avoids other CPU/GPU copies. Compare with `ortGraphOpt=all` (and explicitly keep `ortDequantFold=0` if isolating only ORT's optimizer). Plain-f16 LQO can also be tried with either optimizer setting by appending the two flags to its override URL above.

### Diagnostic-control regression verification

- `npm run lint`: passed. **217 relevant tests passed**, covering active status through deferred load/search/teardown, crash continuation with saved overrides, URL rejection, isolated shipping/diagnostic pools, production split-K reporting, all five graph levels, cache invalidation when options change, and actual search/Maia worker handlers receiving settings without a page URL.
- `npm test`: **2,277 passed, 4 existing optional skips, zero failures** (including Svelte typecheck). Product build passed.
- Built-page agent-browser check on this Mac, isolated static server on available port :60037: during LQO loading, the row's `data-status`, API result, and stored result were all **running**, with requested model/options checkpointed before session creation.
- Plain LQO override with `ortGraphOpt=disabled&ortDequantFold=0`: **ok**, load **11,535ms**, first search **555ms**; actual served URL was `https://assets.0x88.app/models/lc0/lqo_v2.f16.onnx`, worker diagnostics confirmed graph **disabled**, app fold **false**, no registered split-K rewrite. Maia with the same flags: **ok**, load **1,649ms**, first inference **249ms**, worker-confirmed settings identical.
- Plain LQO with default settings: **ok**, warm load **1,079ms**, first search **219ms**, worker-confirmed graph **all** and app fold **true**. Its model bytes were **189,397,245** before/after the app pass with zero folded nodes. These mixed cache/optimizer timings are functional checks, not a controlled performance comparison.
- Reloaded during a subsequent active heat step: recovery preserved `lqo ok`, recorded `heat crashed`, counted one reload and retained the plain-f16 override. No OS kill was induced or inferred by this manual reload test.
- These new controls have **not been rerun on the physical iPhone by this agent**; the phone findings at the top remain the owner's reported measurements.

## Desktop verification

Built with `BUILD_SCOPE=product npm run build:client`, served from this worktree's `dist-client` using `PORT=65230 npm run web:isolated:static` (an available port chosen without disturbing :5181). Response headers checked: COOP `same-origin`, COEP `require-corp`. Browser: agent-browser-managed Headless Chrome 157 on macOS, Apple / metal-3 adapter, shader-f16 present, hardware concurrency 10, reported deviceMemory 16GB. Adapter limits: maxBufferSize/maxStorageBufferBindingSize 4,294,967,292 bytes; workgroup storage 32,768 bytes; invocations 1,024. Production CDN assets were selected with the assetBase URL above.

Representative successful runs from feature commit `7bf59a73` (mixed initial/warm HTTP and model caches; not comparable cold-start benchmarks). The ownership-review follow-up checks are listed separately below; the engine loading/search paths are unchanged.

| Step | Load ms | First search ms | Start / middlegame rate |
| --- | ---: | ---: | --- |
| Stockfish Lite | 943 | 1,001 | 855,172 / 840,424 nodes/s |
| Stockfish full, 8 threads | 845 | 1,006 | 1,640,237 / 1,331,566 nodes/s |
| Reckless | 1,934 | 1,007 | 361,923 / 336,498 nodes/s |
| Viridithas | 4,857 | 1,007 | 180,024 / 161,427 nodes/s |
| Berserk | 1,713 | 1,006 | 644,855 / 370,601 nodes/s |
| PlentyChess | 2,859 | 914 | 298,069 / 435,175 nodes/s |
| Stormphrax | 2,847 | 1,011 | 308,855 / 310,430 nodes/s |
| Lc0 Small, Apple split-K | 2,083 | 221 | 36 / 58 visits/s |
| Queen Odds | 12,650 | 141 | 57 / 183 visits/s |
| t3, Apple split-K | 9,381 | 103 | 78 / 123 visits/s |
| BT4 | 4,231 | 208 | 38 / 40 visits/s |
| Maia3 | 4,901 | 124 | 8.0 / 39.3 evaluations/s |
| Centipawn | 1,446 | 184 | 43.5 / 97.1 visits/s |
| Arena triple residency | 20,915 | 250 (first seat) | Nets 25–41 visits/s; SF 1.40M / 1.11M nodes/s |

Neural backend strings came from production adapters: principally `auto->webgpu`; Maia, Centipawn and LQO reported `auto->webgpu,wasm` (the configured provider list, not proof that every operator executed on GPU). CPU runtime metadata includes actual worker/variant URLs and thread counts.

On the original feature-verification build, an `auto=1` run at 375px ran Lite, Small, t3, and a **120.032-second sustained t3 search** successfully. Warm load times: 99 / 587 / 1,203 / 448ms. The sustained search averaged about **170 visits/s**. Slice rates were **160.3, 153.3, 160.2, 162.5, 169.9, 179.4, 176.9, 167.9, 175.6, 180.0, 181.1**, then **182.9** for the final 7.17s partial slice; full slices lasted 10.18–10.34s. No desktop thermal decline was established. Sampled main-thread JS heap reached approximately 107.5MB after Small load, then approximately 8.5MB after search; these are emphatically not total tab/GPU/worker residency. UA-specific memory hit its diagnostic timeout; WASM heap fields were null.

Additional checks:

- Reloaded mid-heat with agent-browser: earlier Lite/t3 results preserved, heat `crashed`, reloadCount 1, choice message shown. Clicked Continue; Small ran next and completed without retrying heat.
- Repeated with UA-memory API disabled (Safari-like timing); departure marker survived immediate teardown. Inspected the recovery UI and [375px screenshot](mobile_soak_375_2026-11-16.png). Document scrollWidth and viewport width both **375px**; no horizontal scroll.
- Final-build `auto=1` reload mid-heat automatically continued: `t3 ok → heat crashed → lc0-small ok`, reloadCount 1, pagehide + hidden visibility event retained.
- Forced Storage.setItem to throw SecurityError: heat-only prerequisite skipped with reason, Lite still completed, storage-unavailable event remained after reset/start.
- Forced `ortEp=wasm`: Small loaded the original (non-split-K) model in **2,019ms**, first search **411ms**, approximately **19.5 / 18.7 visits/s**, backend `wasm`.
- Copy results button reported “Results copied”; rendered JSON parsed and matched live API state.
- Ownership-review follow-up: rebuilt the product and served with COOP/COEP on an available port (:51769). Normal Stockfish Lite still completed (481ms load, 1,003ms first search). Then injected a rejecting `Worker.terminate()` during a second Lite run: Lite failed, t3 was skipped, `window.__mobileSoak.unsafeDisposal` became true, Start/Continue/Start over were disabled and API calls could not erase the failed report; Copy results remained enabled. Same-document Svelte navigation to another lab route and back retained the latch/alert. A full reload cleared the latch and re-enabled Start while preserving the original failure evidence. The test-only Worker override was restored; no actual release failure was claimed.
- First ownership-review regression run (`a777b4ee`): **55 passed**. `tests/mobile_soak.test.mjs` contains 17 tests covering query/selection, recovery transitions, departure snapshot isolation, safe teardown failure handling, malformed storage, strongest-net selection, slices, awaited Centipawn teardown and device-loss payloads. `tests/mobile_soak_controller.test.mjs` adds six executable controller regression tests with deferred environment/search/disposal promises: initial auto-start and per-run probes after retirement, late cleanup versus reopened breadcrumbs, shared-queue ownership, unchanged pagehide/BFCache semantics, failed teardown blocking API/reset/remount until a fresh document, and late retired disposal failure refreshing/blocking the current controller. Production engine imports are lazy and controller services are injectable so these tests run the real controller without loading engines or GPU code.
- `npm run lint`, `npm test` (including Svelte typecheck), and product build passed. Four existing optional tests skip in the full suite; no failures.

## Minimal production changes

1. `nn/squareformerEvaluator.ts`: add idempotent, awaitable `dispose()` that drains the existing `ExclusiveQueue` and awaits ORT session release. Existing `destroy(): void` retains its fire-and-forget/nonthrowing behavior by delegating to it. No engine/search refactor.
2. `nn/ortDeviceLoss.ts`: observer for ORT's actual device, with unsubscribe, one promise subscription per device, and a refreshable single-owner monitor; no extra device allocation.
3. `lc0/searchWorkerCore.ts` + `bt4Engine.ts`, and `lc0/maia3Worker.ts` + `maia3.ts`: forward/store device-loss notifications with no request ID; request handlers explicitly separate them from replies. Other search-worker consumers have equivalent guards (audit below). Load/backend selection is unchanged.
4. `lc0/bigNetSessionPool.ts`: minimal `detach(key, expectedInstance)` identity check (and default-pool wrapper), canceling any idle timer without destroying the worker. Existing production acquire/release/dispose behavior is unchanged.

The lab-specific pure logic, browser orchestration and adapters are respectively `lc0/mobileSoak.ts`, `mobileSoakBrowser.ts`, and `mobileSoakEngines.ts`; `mobileSoakCpu.ts` measures existing CPU adapters without implementing a loader. The Svelte route contains presentation and button wiring only.

## Second review: worker protocol, pooled handoff, deferred CPU loading

### Worker notification audit

A repo-wide grep of worker entrypoints, constructors and message handlers found these consumers of `searchWorker.ts` / its `hybridSearchWorker.ts` and `batchingStrengthWorker.ts` wrappers, and `maia3Worker.ts`. Every request handler now explicitly returns on `deviceLost` (Bt4/Maia inference store its diagnostics instead), before looking up or settling requests. Numeric request IDs are required for replies. No emitted device-loss notification has an `id` property, so even a notification delivered before `ready` cannot impersonate the pending init response.

| Consumer / handler | Audit / regression coverage |
| --- | --- |
| `analysisBrowser.ts::initWorker` | Ignores telemetry, including a forged current init ID; real handler body tested. |
| `criticalMomentsEngines.ts::initLc0` | Same guard before its pending map; real handler body tested. |
| `policyOnlyBrowser.ts::ensureSearchWorkerInstance` | Covers normal + hybrid worker; real handler body tested. Its separate `postLabWorkerRequest` handler is guarded too, even though `searchLabWorker.ts` does not emit this notification. |
| `batchingStrengthBrowser.ts::createWorker` | Covers the batching wrapper of the same core; real handler body tested. |
| `bt4Engine.ts::Bt4WorkerSearcher.init` | Stores loss without settling init; controlled Worker test sends notifications before `ready`. Play/Arena/Analysis/soak big nets all use this adapter. |
| `maia3.ts::postInit` and `onMessage` | Init ignores it; inference records it without resolving a pending request. Both handlers tested. All Maia page/report consumers go through this adapter. |

`searchWorker.ts`, `hybridSearchWorker.ts` and `batchingStrengthWorker.ts` all enter `startSearchWorker`; there are no other direct clients in src/scripts/lab/public. Maia has only the adapter-owned constructor. The monitor is constructed once outside each worker's init handler, refreshed only for a different device, and unsubscribed on retirement/session disposal or runtime switch. A WeakMap shares a single lost-promise callback per actual device even across session replacements. Tests perform 100 refreshes, switch devices, unsubscribe retired owners, and confirm no callback accumulation. A real-core regression holds old-session disposal pending, triggers loss while replacement init is awaiting it, verifies the notification has no ID and init remains unsettled, then releases disposal and receives the genuine `ready` reply.

### Big-net ownership on same-document navigation

The soak now synchronously retires every owned adapter during controller destroy, **before** a destination's mount can acquire from the shared pool. For a big net, retirement detaches only the exact acquired instance from its key and clears its idle timer; it does not abandon ongoing work. Async cleanup later disposes the captured instance directly, never `disposeNow(key)`. Therefore a Play/Arena/Analysis destination gets a fresh pool record and cannot have that replacement terminated by late soak cleanup. An asset probe that returns after abort cannot acquire a new pooled instance. No new lease system or pool redesign was introduced.

Pool tests cover matching/mismatching identities, cleared idle timers and replacement survival. A controller regression keeps a big-net search pending, destroys the soak, immediately acquires the same key as a production destination would, then lets old cleanup finish: only the old instance is disposed, and the replacement remains in the pool.

### CPU preparation versus initialization/search

All CPU prewarm implementations were checked: Stockfish completes UCI readiness; Berserk/PlentyChess/Stormphrax inherit the Emscripten readiness handshake; Reckless browser-API mode initializes its worker; **WASI Reckless and Viridithas may return without loading WASM** when SAB/persistent mode is unavailable, forced off, or disabled after fallback. Each one-shot request can instantiate a fresh WASI runtime, not merely the first request.

The lab measurement adapter uses the production runtime status after prewarm, not engine names or elapsed-time guesses. Known resident modes are `worker`, `emscripten-worker`, `persistent`, `browser-api`; idle/one-shot/unknown modes are not resident. Deferred steps report:

- `loadDeferred: true`, `loadMs: null`, measured `prepareMs`, `residentAfterPrepare: false` in runtime metadata;
- `firstInitSearchMs`: measured end-to-end first search that included startup, **not an init-only time**; `initializationMs` remains null because production one-shot adapters do not expose separate fetch/instantiate/UCI timing;
- each affected search has `initializationIncluded: true` and `perSecond: null`, rather than publishing startup-inclusive wall time as pure throughput; `engineReportedNps`, when present, is kept separately and labeled as engine-reported;
- memory immediately after preparation is tagged `after-prepare`, not `resident-after-load`.

The UI labels deferred setup and init-plus-search explicitly. Runtime status is refreshed after searches; a persistent-to-one-shot fallback also suppresses that search's wall-throughput even if prewarm initially succeeded. Tests exercise real Viridithas/Reckless no-SAB prewarm (no worker created), both one-shot searches, all resident modes, mid-search fallback, and persisted report/memory shaping. Existing CPU engine implementations and constructor/asset selection paths were not changed.

Second-review validation: `npm run lint` passed; **57 relevant regression tests passed** across worker protocol/consumer, pool, controller, CPU measurement, existing Analysis/critical/Maia lifecycle, and pure soak tests. `npm test` (including Svelte typecheck) passed **2,256 tests, 4 optional skips, zero failures**; `BUILD_SCOPE=product npm run build:client` passed.

Quick agent-browser check against this follow-up product build, served from the worktree with COOP/COEP on an available port (:56831):

- Stockfish Lite: **ok**, load/prepare **482ms**, first search **1,001ms**, approximately **831k / 844k nodes/s**.
- t3: **ok**, WebGPU Apple split-K, load/prepare **8,644ms**, first search **128ms**, approximately **63 / 77 visits/s**; no device loss or unsafe disposal.
- Then temporarily disabled the page's SAB availability (restored afterward; this is a desktop mode-selection test, not physical Safari). Both real production WASI adapters completed in one-shot mode: Viridithas prepare **110ms**, first init-plus-search **4,787ms**, second **1,603ms**; Reckless prepare **111ms**, first init-plus-search **2,929ms**, second **1,034ms**. Both reported `loadMs: null`, `loadDeferred: true`, `initializationMs: null`, `residentAfterPrepare: false`, `after-prepare` memory phase, and null wall `perSecond` on both positions. UCI NPS remained separately labeled (Viridithas ~327k / 320k, Reckless ~607k / 590k). Inspected the rendered deferred-load/init-plus-search labels as well as API/JSON metadata.

## iOS Safari risks / unverified items

- Agent verification remains desktop-only; the owner-reported physical iPhone findings are recorded above. No Android run is recorded. Desktop success does not establish mobile safety, total memory bounds, or resistance to Jetsam/tab reloads.
- Big-net decoding, dequantization, ORT graphs and GPU buffers can multiply the 96–178MB compressed-file sizes. Two pooled big nets plus full Stockfish's network, WASM heaps, pthreads and 64MB hash are intentionally hazardous on a phone.
- Worker WebGPU, shader-f16, adapter limits and ORT operator support may differ from desktop. Production fallback can be much slower and consume substantial CPU/memory; do not infer support solely from `navigator.gpu`.
- Safari memory APIs generally provide no warning before a kill. Storage/private-mode/quota denial can also remove crash detection; a discarded browser profile/cleared storage cannot preserve evidence.
- Backgrounding, locking or heat may throttle/suspend workers. Foreground the device for comparable heat slices; consult lifecycle events before interpreting a slowdown. Search progress can be delayed by a slow inference; slices record actual elapsed time rather than fabricating precise ten-second samples.
- The phone needs a **secure, trusted origin**, and COOP/COEP plus compatible asset CORS/CORP for isolation. Plain HTTP to a Mac LAN address is not equivalent to desktop localhost; losing secure-context/isolation capabilities changes the runtime/thread plan. USB WebDriver does not itself supply these headers or HTTPS.
- A browser-reported GPU lost reason is observable only if the JS context survives long enough to deliver it. An OS-killed tab may leave only the previous breadcrumb; do not expect an explicit OOM error message.
