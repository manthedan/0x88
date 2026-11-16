# Mobile soak lab — 2026-11-16

Route: `/lab/mobile-soak` (included in the product build; deliberately absent from navigation).

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

Poll the API rather than assuming a fixed load delay. Stockfish and WASM handshakes use their production deadlines; neural initialization/inference can take much longer on a phone. There is no lab timer that abandons a pending load and starts another resident engine. A hung browser/runtime may require a reload; that interruption is captured by the breadcrumb.

## Steps and production paths

Ordinary steps load once, search **startpos** and **Kiwipete** (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`), then dispose. CPU searches request 1,000ms; tree-search nets request eight visits per position. These are smoke/stress workloads, not engine-strength benchmarks. Load time includes download/cache access plus initialization, but excludes the environment and memory probes.

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

Memory samples are ordered: before load, resident after load, after searches, after disposal. Chrome JS heap fields are copied when exposed. UA-specific memory sampling is attempted only when available and isolated, with a two-second diagnostic timeout. After a timeout it is disabled for the remainder of that page lifetime, avoiding a growing pile of GC requests. GPU memory is not covered reliably. Worker WASM heaps are private in these shipped adapters; `wasmHeapBytes` is explicitly null. Safari can legitimately have null for every memory field.

Actual ORT `env.webgpu.device.lost` is observed (not the diagnostic adapter). Big-net and Maia workers forward reason/message; main-thread Small/Centipawn use the same observer. Loss marks a completed step failed. No real device loss was observed on this Mac; the notification payload/unsubscribe behavior is unit-tested with a controlled device.

One `localStorage` value (`0x88-mobile-soak-v1`) contains the entire report plus the in-progress marker, written **before** each step allocates/loads. Partial search/heat results are checkpointed. On recovery the marked step becomes `crashed`, previous results remain, reload count increments once, and Continue advances beyond it. Pagehide/visibility events are retained in a bounded 100-event log. A detached pagehide snapshot prevents worker rejection/cleanup during unload from clearing the stored marker, including when Safari has no asynchronous memory probe to delay cleanup. BFCache restoration unfreezes the live runner. A reload during environment collection or exactly between steps has no engine marker: it restores idle/done rather than stranding a non-existent runner or attributing an engine crash.

Destroying a controller immediately disables both persistence and UI publication, removes its document-safety subscription, and prevents pending auto-start continuations or stale API calls from starting/resetting runs. Pending environment probes and asynchronous cleanup may still settle, but cannot overwrite a reopened controller's report or crash marker. A document-scoped `ExclusiveQueue` keeps a reopened controller's allocations behind retired cleanup; if that cleanup fails, the unsafe-disposal latch notifies the mounted UI and prevents its queued run from allocating engines. Mounted pagehide snapshots still remain frozen through cleanup and unfreeze on BFCache pageshow.

A breadcrumb proves interruption, **not** an OS OOM: manual reload/navigation and a killed tab are indistinguishable. All storage reads/writes are caught; denied/quota storage leaves an in-memory runner and a `storage-unavailable` event. No account, upload or external results service is involved.

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

On the final build, an `auto=1` run at 375px ran Lite, Small, t3, and a **120.032-second sustained t3 search** successfully. Warm load times: 99 / 587 / 1,203 / 448ms. The sustained search averaged about **170 visits/s**. Slice rates were **160.3, 153.3, 160.2, 162.5, 169.9, 179.4, 176.9, 167.9, 175.6, 180.0, 181.1**, then **182.9** for the final 7.17s partial slice; full slices lasted 10.18–10.34s. No desktop thermal decline was established. Sampled main-thread JS heap reached approximately 107.5MB after Small load, then approximately 8.5MB after search; these are emphatically not total tab/GPU/worker residency. UA-specific memory hit its diagnostic timeout; WASM heap fields were null.

Additional checks:

- Reloaded mid-heat with agent-browser: earlier Lite/t3 results preserved, heat `crashed`, reloadCount 1, choice message shown. Clicked Continue; Small ran next and completed without retrying heat.
- Repeated with UA-memory API disabled (Safari-like timing); departure marker survived immediate teardown. Inspected the recovery UI and [375px screenshot](mobile_soak_375_2026-11-16.png). Document scrollWidth and viewport width both **375px**; no horizontal scroll.
- Final-build `auto=1` reload mid-heat automatically continued: `t3 ok → heat crashed → lc0-small ok`, reloadCount 1, pagehide + hidden visibility event retained.
- Forced Storage.setItem to throw SecurityError: heat-only prerequisite skipped with reason, Lite still completed, storage-unavailable event remained after reset/start.
- Forced `ortEp=wasm`: Small loaded the original (non-split-K) model in **2,019ms**, first search **411ms**, approximately **19.5 / 18.7 visits/s**, backend `wasm`.
- Copy results button reported “Results copied”; rendered JSON parsed and matched live API state.
- Ownership-review follow-up: rebuilt the product and served with COOP/COEP on an available port (:51769). Normal Stockfish Lite still completed (481ms load, 1,003ms first search). Then injected a rejecting `Worker.terminate()` during a second Lite run: Lite failed, t3 was skipped, `window.__mobileSoak.unsafeDisposal` became true, Start/Continue/Start over were disabled and API calls could not erase the failed report; Copy results remained enabled. Same-document Svelte navigation to another lab route and back retained the latch/alert. A full reload cleared the latch and re-enabled Start while preserving the original failure evidence. The test-only Worker override was restored; no actual release failure was claimed.
- Relevant lifecycle + unit tests after review fixes: **55 passed**. `tests/mobile_soak.test.mjs` contains 17 tests covering query/selection, recovery transitions, departure snapshot isolation, safe teardown failure handling, malformed storage, strongest-net selection, slices, awaited Centipawn teardown and device-loss payloads. `tests/mobile_soak_controller.test.mjs` adds six executable controller regression tests with deferred environment/search/disposal promises: initial auto-start and per-run probes after retirement, late cleanup versus reopened breadcrumbs, shared-queue ownership, unchanged pagehide/BFCache semantics, failed teardown blocking API/reset/remount until a fresh document, and late retired disposal failure refreshing/blocking the current controller. Production engine imports are lazy and controller services are injectable so these tests run the real controller without loading engines or GPU code.
- `npm run lint`, `npm test` (including Svelte typecheck), and product build passed. Four existing optional tests skip in the full suite; no failures.

## Minimal production changes

1. `nn/squareformerEvaluator.ts`: add idempotent, awaitable `dispose()` that drains the existing `ExclusiveQueue` and awaits ORT session release. Existing `destroy(): void` retains its fire-and-forget/nonthrowing behavior by delegating to it. No engine/search refactor.
2. `nn/ortDeviceLoss.ts`: small observer for ORT's actual device, with unsubscribe, no allocation.
3. `lc0/searchWorkerCore.ts` + `bt4Engine.ts`, and `lc0/maia3Worker.ts` + `maia3.ts`: forward/store device-loss notifications without changing load/backend selection or worker ownership.

The lab-specific pure logic, browser orchestration and adapters are respectively `lc0/mobileSoak.ts`, `mobileSoakBrowser.ts`, and `mobileSoakEngines.ts`. The Svelte route contains presentation and button wiring only.

## iOS Safari risks / unverified items

- **No physical iPhone or Android run was performed.** Desktop success does not establish mobile safety, total memory bounds, or resistance to Jetsam/tab reloads.
- Big-net decoding, dequantization, ORT graphs and GPU buffers can multiply the 96–178MB compressed-file sizes. Two pooled big nets plus full Stockfish's network, WASM heaps, pthreads and 64MB hash are intentionally hazardous on a phone.
- Worker WebGPU, shader-f16, adapter limits and ORT operator support may differ from desktop. Production fallback can be much slower and consume substantial CPU/memory; do not infer support solely from `navigator.gpu`.
- Safari memory APIs generally provide no warning before a kill. Storage/private-mode/quota denial can also remove crash detection; a discarded browser profile/cleared storage cannot preserve evidence.
- Backgrounding, locking or heat may throttle/suspend workers. Foreground the device for comparable heat slices; consult lifecycle events before interpreting a slowdown. Search progress can be delayed by a slow inference; slices record actual elapsed time rather than fabricating precise ten-second samples.
- The phone needs a **secure, trusted origin**, and COOP/COEP plus compatible asset CORS/CORP for isolation. Plain HTTP to a Mac LAN address is not equivalent to desktop localhost; losing secure-context/isolation capabilities changes the runtime/thread plan. USB WebDriver does not itself supply these headers or HTTPS.
- A browser-reported GPU lost reason is observable only if the JS context survives long enough to deliver it. An OS-killed tab may leave only the previous breadcrumb; do not expect an explicit OOM error message.
