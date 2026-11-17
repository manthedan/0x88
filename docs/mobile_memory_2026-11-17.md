# Mobile memory, phases 1–3 — 2026-11-17

**Current policy is [Phase 3](#phase-3-lqo-batch-one-capture-and-maia3-folding-restored): LQO capture at batch 1; Maia3/Small folding on; big-net folding off.** Phase 2's Maia fold-off policy is historical. Mini measurements and review are complete; laptop Safari measurement remains externally blocked.

**Phase 1 below is historical. Its sampled peaks and same-tab “warm” rows are superseded by [Phase 2](#phase-2-fresh-process-measurements-and-per-net-policy).** Same-tab reloads inherited the previous process's allocations; the variable post-disposal tail also captured WebAssembly JIT growth. Those rows must not be used as load-peak or speed comparisons. The phase-1 blanket CPU-IO policy was a real fixed-net regression, not random noise; Phase 2 removes it. Phase 2 also supersedes the old all-model fold-on policy.

Companion to [mobile soak](mobile_soak_2026-11-16.md). Worktree/branch: `mobile-soak` / `lab/mobile-soak`; baseline `720aaf4a`. Desktop measurements below are **not iPhone validation**. The Carbon analysis is a source of hypotheses, not allocation evidence.

## Phase 1 scope and invariants (historical)

Four separately tested changes: (1) big-net CPU IO and pre-session shape eligibility, (2) channel integrity and one-shot warm-cache parse recovery, (3) byte-identical single-output-buffer dequantization and shorter input lifetimes, (4) Analysis big-net retirement. Folding stays enabled by default. No OPFS, incremental hashing, model-format change, ORT patch, deployment, or cache-policy redesign.

Keep pinned SHA-256/length verification, verified-only cache writes, the existing metadata-trusted warm-cache policy, resumable/shard verification, and cache version/LRU limits. Do not trade these for memory. The existing metadata fast path attests previously verified storage; it is not a new hash of every warm read. Parse-failure recovery is defense in depth, not a claim that protobuf parsing authenticates tensor weights.

## Owner evidence (copied, not newly measured)

Physical iPhone Safari 26.6: LQO dies at WebContent `ActiveHard 2048 MB`; footprint 0.34 s before death was 1639 MB. BT4/Arena also die; t3, Maia3, Centipawn and Small survive.

Earlier desktop LQO default trace, MB:

| t (ms) | Total | JS VM Gigacage | WebKit malloc | Graphics |
|---:|---:|---:|---:|---:|
| 3083 | 186 | 120 | 26 | 31 |
| 4816 | 341 | 183 | 117 | 31 |
| 5941 | 830 | 751 | 38 | 31 |
| 7764 | 1443 | 376 | 1004 | 32 |
| 9732 | 2294 | 948 | 1085 | 229 |
| 13424 | 2738 | 1304 | 1164 | 236 |

Newer owner desktop LQO peaks, MB:

| Configuration | Total | Gigacage | WebKit malloc |
|---|---:|---:|---:|
| Default | 3130 | 1356 | 1518 |
| `ortDequantFold=0` | 1629 | 93 | 1331 |
| `ortDequantFold=0&ortGraphOpt=disabled` | 1520 | — | — |
| Plain `lqo_v2.f16.onnx` | 2423 | — | 1986 |

Owner's WASM-only LQO peak: 1349 MB, including 1025 MB WebKit malloc. This rules out attributing *all* that category to GPU upload; it does not identify every native allocation.

## Measurement method

- Safari **26.4** on this Mac; Apple WebGPU, COOP/COEP, `crossOriginIsolated=true`. Runtime reports asyncify, one WASM thread; SharedArrayBuffer is available.
- Own static server: `scripts/serve_isolated_static.mjs dist-client`, initially port 61981. Existing `127.0.0.1:4445` safaridriver was used, not started/stopped.
- Owner's unmodified `mac-footprint.mjs` samples new WebContent processes. Its 300 ms sleep is **not** the effective sampling interval: synchronous `footprint` calls take additional time. Peaks can miss shorter spikes.
- URL: `/lab/mobile-soak/?auto=1&steps=<lqo|t3|bt4>&assetBase=https%3A%2F%2Fassets.0x88.app`. Shipped URLs, automatic Apple t3 split-K, default graph `all` and app folding enabled unless stated otherwise.
- A measurement-only WebDriver preload navigates with `auto=0`, observes small worker-ready messages, then calls the same `__mobileSoak.start()` used by auto-run. No alternate model loader. Tracing runs additionally instrument the real worker; their footprints are **excluded** from the comparison tables.
- **Cold:** clear Cache Storage before starting. **Warm:** hard-navigate and rerun inside the **same automation session**, asserting the worker reports `hit … sha256 trusted`. Separate Safari automation sessions did not preserve Cache Storage; initial purported warm runs were misses and were discarded.
- Warm rows deliberately include the real same-WebContent reload/allocator-retention cost. They are not fresh-process cache-hit estimates. Neither `disposeMs` nor `worker.terminate()` guarantees immediate OS footprint reclamation. A warm increase alone does not prove two live pool records.
- Each table cell is **sampled peak / end-window footprint**, MB as converted by the supplied script. “End” is the median of the final three samples, about eight seconds after the first `done` poll. This is the requested settled observation, **not a guaranteed steady-state plateau or forced-GC result**. Some native totals were still growing. One cold/warm pair per model per stage; do not overinterpret small differences.
- Raw samples, ready messages, reports, trace events and measurement helpers are retained locally under ignored `.local-dev-artifacts/mobile-memory/`; `base-final-*` is the valid baseline and `step1-*` the first change.

## Verified load path and copy accounting

`Bt4WorkerSearcher.init()` posts a **URL and settings**, not model bytes, to `searchWorkerCore`. `createLc0OnnxEvaluatorFromUrl()` chooses the device variant, then `modelCache` downloads/verifies/caches **inside that worker**. There is no main-to-worker full-model structured clone to replace with a transfer. The ordinary big-net path is monolithic, not `modelPack`/resumable assembly.

Cold LQO instrumentation counted **one model SHA-256 call, 95,662,716 bytes**. On this input, the pinned manifest length preallocates the download target, so capacity-doubling is not the observed load path. Verification and `cache.put` are awaited before folding. This does not establish that cache/hash copies are free, nor audit every resumable path.

Page-relative instrumented baseline timeline (ms):

| From–to | Event | Evidence |
|---|---|---|
| 7701–7801 | Whole-model SHA-256, 95,662,716 bytes | Wrapped real `crypto.subtle.digest`; one call |
| 7805–8149 | `new Response(bytes)` / Cache Storage put | Wrapped real Response/Cache APIs; native buffer lifetime beyond API completion not observable |
| 8245–9225 | Fold #1, 95,662,716 → 189,399,671 bytes | Temporary hooks around existing fold |
| 9225–10690 | ORT create #1 | Temporary hooks around real `InferenceSession.create` |
| 9515–9654 | WASM linear memory grows 199,098,368 → 413,073,408 bytes | Real exported Memory and `Memory.grow` observations |
| 10699 | GPU IO rejected: `/input/planes: dynamic shape` | Existing warning |
| 10699–11936 | Fold #2, same sizes | Same hooks |
| 11939–12841 | ORT create #2 | Same hooks |

The large later peak is not contemporaneous with the digest call in this trace. Crucially, the observed **413 MB linear-memory high-water is not the entire 1–2 GB “WebKit malloc” category**. Carbon's categorical attribution of that bucket to a ~1 GB WASM heap / 4–5× protobuf multiplier is not demonstrated here. Native JIT, driver allocations, staging, allocator retention and ORT internals need separate attribution.

Copy table (sizes are exact bytes where given; aliases are not counted twice):

| Copy / allocation | Realm / owner | LQO / BT4 size | Alive from–to | Evidence / certainty |
|---|---|---|---|---|
| Download target / warm `arrayBuffer()` result | Search worker JS, `modelCache` | 95,662,716 / 186,676,343 | Download/cache read through evaluator preparation; formerly retained across ORT creation by `modelLoad.model` and `prepared.original` aliases | Source-proven single shared input buffer; aliases are not extra copies |
| WebCrypto native input, Response/cache native body | WebKit internals | API input 95,662,716 / 186,676,343; exact native residency unknown | Digest / Response construction and Cache put/read; release time unknown | API calls measured; exact native copy count/lifetime **not** inferred from the category total |
| Per-tensor f32 dequantized values | Worker JS, old `dequantize()` | Largest 6,291,456 / 100,663,296 | One tensor's conversion; may remain uncollected afterwards | Source plus shipped-model tensor sizes |
| Per-tensor f16 encoding scratch | Worker JS, old `encodeFloat()` | Largest 3,145,728 / 50,331,648 | Tensor encoding; may remain uncollected | Source plus shipped-model tensor sizes |
| Encoded replacement initializer Map | Worker JS, old fold | Aggregate 188,246,499 / 369,072,867 | Accumulates during folding; live through graph serialization | Exact sum of replacement TensorProto lengths |
| Serialized graph intermediate | Worker JS, old `graph.finish()` | 189,399,626 / 370,637,592 | Graph finish through final model finish | Exact GraphProto payload size |
| Final folded model | Worker JS → ORT | 189,399,671 / 370,637,638 | Final model finish through ORT byte ingestion | Exact output; 144 / 147 nodes folded |
| ORT input `_malloc` / `HEAPU8.set` | ORT WASM linear memory | Folded model length | Native model load until native session creation finishes | Installed ORT-web loader source; allocation is inside the measured heap, not additive to it |
| Parsed graph / CPU initializers / temporaries / allocator slack | ORT internals | Not individually measured | Native session initialization and session lifetime | Hypotheses; cannot split them using `footprint` alone |
| Resident GPU weights / staging / pipelines | WebGPU / WebKit | Category total is observable, individual buffers not fully attributed | Upload through session/device teardown and native reclamation | Baseline cold peak graphics about 220 / 411 MB; not a byte-exact weight-only measurement |

Before rewriting, **input + replacement Map + graph + final model alone** necessarily coexist at final serialization: **662,708,512 bytes LQO / 1,297,024,440 bytes BT4**, excluding per-tensor garbage, runtime and ORT. This is source/size accounting, not an assertion that each byte is simultaneously physically resident. LQO then repeats the fold/session sequence; t3's plain-f16 split-K model is a no-fold identity return.

## Baseline and step 1: session creation / IO

Baseline:

| Model | Cold peak / end MB | Warm peak / end MB | Load cold / warm ms | Midgame visits/s cold / warm |
|---|---:|---:|---:|---:|
| lqo | 3627 / 3576 | 5255 / 4781 | 9332 / 6248 | 67.5 / 65.8 |
| t3 | 1747 / 1744 | 3545 / 3436 | 12972 / 4163 | 70.2 / 49.2 |
| bt4 | 2800 / 2800 | 5862 / 3225 | 19050 / 6562 | 18.8 / 22.7 |

After step 1:

| Model | Cold peak / end MB | Warm peak / end MB | Load cold / warm ms | Midgame visits/s cold / warm |
|---|---:|---:|---:|---:|
| lqo | 2903 / 2820 | 5164 / 5162 | 8299 / 3563 | 74.3 / 71.4 |
| t3 | 2871 / 2807 | 5554 / 5510 | 11424 / 3202 | 65.1 / 64.7 |
| bt4 | 3754 / 3630 | 6487 / 6139 | 14814 / 6357 | 23.0 / 21.4 |

- Big-net init explicitly sends `ortIoMode:'cpu'`, independently of `deviceModelSelection:true`; worker-ready messages confirm CPU IO. Small's `defaultLc0OrtIoMode(true) === 'gpu-capture'` is unchanged. A real Safari production search-worker probe selected Small split-K, reported `gpu-capture`, and completed eight visits.
- New protobuf-only IO reader inspects declared tensor types and input/output shapes before any GPU-IO session attempt. Symbolic/missing/nonpositive input dimensions, unresolved output dimensions, nonfloat outputs and missing heads reject without a probe session. Small's policy declaration is `['batch',1858]` despite a fixed batch-1 input: preflight permits that known head-batch convention for ORT to infer, while post-create validation still demands numeric dimensions. Initializers listed as old-format graph inputs are excluded. Raw payloads are only viewed, not copied. URL-only lab inputs conservatively use CPU IO rather than fetching an unverified shape-probe copy. Post-create metadata validation remains defense in depth.
- LQO's duplicate initialization is removed. Its Gigacage at the cold total-footprint peak fell from **1180 to 274 MB** in these runs. **Total footprint did not improve consistently across models:** native malloc/end-window retention was larger on t3/BT4. These data do not establish the consultant's predicted total-memory saving.
- Quick speed check uses the second fixed eight-visit position, after initial inference. t3 cold −7%, warm +32%; BT4 cold +22%, warm −6%: no consistent regression, but this small sample cannot prove a <2% bound. CPU IO moves the old captured zero-input warm-up cost into first search; do not compare first-search speed without accounting for load/warm-up time.
- Validation: lint, 80 relevant tests, Svelte typecheck (zero errors/warnings), product build. Baseline output SHA-256 goldens were captured **before** changing the folding algorithm for all seven shipped/fallback QDQ artifacts (Small variants, LQO, BT4, Maia variants), after verifying their pinned input hashes.

## Step 2: integrity and warm-cache recovery

After step 2 (compare directly with step 1 above):

| Model | Cold peak / end MB | Warm peak / end MB | Load cold / warm ms | Midgame visits/s cold / warm |
|---|---:|---:|---:|---:|
| lqo | 3007 / 2998 | 4735 / 4668 | 6077 / 2892 | 100.7 / 99.9 |
| t3 | 2838 / 2808 | 5420 / 5281 | 9508 / 2982 | 30.9 / 60.7 |
| bt4 | 3839 / 3750 | 6999 / 6827 | 14170 / 6652 | 23.1 / 21.8 |

This is an integrity/recovery change, **not a normal-load memory optimization**. Native totals remain large, and the very short speed samples vary considerably. All six runs passed; warm ready messages confirmed metadata-trusted Cache Storage hits.

- Channel v1 and v2 **decoded** SHA-256/bytes must agree with every pin supplied by the app-origin model manifest. Conflicting metadata throws before cache/body access; it is outside the unavailable-channel catch and cannot silently select another body. Missing legacy channel fields inherit app pins; hexadecimal hash case is compared case-insensitively. Brotli representation hashes are not confused with decoded-model hashes.
- Existing unavailable/ambiguous-channel fallback remains, still verifying the chosen body against the app pins. No new origin fallback, cache version, eviction limit, resumable protocol, or weaker validation mode was introduced.
- A metadata-trusted parse/protobuf failure evicts **only the resolved artifact and its metadata**, reloads with HTTP `cache:'reload'`, freshly hashes the result, and retries **once**. `forceReload` bypasses Cache Storage reads and cannot coalesce with an ordinary read. Corrupt fresh downloads are rejected before a second session attempt/cache write.
- The loader requests parse errors be surfaced before ORT's unfolded/provider fallback (or GPU-IO's CPU fallback) can mask a poisoned selected model by successfully running another artifact. Device loss, OOM, unsupported operators and aborts do not evict good models. Existing split-K-to-original WASM selection remains correct, including recovery of the original rather than the selected cache identity when appropriate.
- Recovery covers the common Lc0 loader (big nets, Small and the standalone policy page) and Maia's transferred-model initialization. Maia retries in a fresh worker; cancellation and the original model-family cache namespace are preserved. TVM/model-pack and resumable integrity behavior is unchanged.
- **Real Safari recovery test:** loaded Small split-K, corrupted a same-length **20,580,952-byte** cached body without changing its verification metadata, then started again. It passed on WebGPU with no variant/provider fallback, made **exactly one** model request with `cache:'reload'`, and repaired the cached SHA-256 to `3a5ccb636ebc596a66021b90a39a3d5b708fbd280048b07ff2d7855d240b37a1`.
- Validation: lint; **253 relevant tests** including all model-cache/model-pack/resumable/engine-integrity cases, recovery, Maia and IO lifecycle tests; Svelte typecheck; **`npm test`: 2323 passed, 4 existing asset-dependent skips, zero failures**; product build. No streaming/incremental hash path was added.

## Step 3: one-output-buffer folding and model lifetimes

After step 3 (same protocol; Safari pairing contention was waited out without terminating the other session):

| Model | Cold peak / end MB | Warm peak / end MB | Load cold / warm ms | Midgame visits/s cold / warm |
|---|---:|---:|---:|---:|
| lqo | 2351 / 2194 | 4569 / 4477 | 9593 / 7289 | 83.1 / 18.2 |
| t3 | 3011 / 2949 | 5549 / 5402 | 16960 / 3103 | 59.1 / 58.5 |
| bt4 | 3733 / 3667 | 6667 / 6280 | 15254 / 6218 | 22.9 / 21.4 |

The isolated cold LQO total peak fell **3007 → 2351 MB** versus step 2 (baseline 3627 MB), but remains above the phone's reported 2048 MB limit. BT4 total peak barely changed; native malloc still dominates. Short-run throughput and load times remain noisy. Observed fold times were LQO **544/928 → 686/1002 ms**, BT4 **2037/2392 → 2202/2580 ms** (cold/warm, step 2 → step 3): memory savings are not presented as a startup-speed optimization.

### What changed

- The fold first validates/plans and computes **exact TensorProto, GraphProto and ModelProto sizes**, then allocates one final model buffer. It reads quantized weights/scales through views/readers and writes f16/f32 values directly at their final offsets. There is no dequantized Float32Array, unpacked int4 tensor, encoded-weight replacement buffer Map, graph intermediate, or final concatenation copy.
- The Map now contains only small plans/readers into the original. Legacy packed `int32_data` and `float_data` encodings remain supported. All unknown fields, initializer ordering, shared-input retention, nested-subgraph consumers, output dtype and skip behavior are preserved.
- Native half IO is used when available, with the manual fallback tested too. The old **f64 → f32 → f16** rounding is preserved (`Math.fround` before native half stores); NaN handling follows the old converter.
- The common verified loader returns **provenance without `.model`**, consumes owned byte holders before awaiting creation, and supplies verified lazy reload callbacks for unfolded/provider/IO fallback. Successful loads neither reread nor redownload a model merely to save a fallback copy. Shared/coalesced results are **not mutated or detached**.
- Simply assigning `modelPath=undefined` inside an async function was insufficient: a Node heap snapshot showed the original still held by the async generator's **`parameters_and_registers`** in both evaluator and ORT creation. Synchronous consuming entry wrappers now pass only a source function to private async frames. The regression test forces GC in an isolated child and proves the unfolded ArrayBuffer is collectible **while native creation is still pending**.
- Maia forwards verified reload provenance with its transferred input, clears the worker-owned init message's buffer reference, and reacquires through the same pinned loader only on fallback. Worker-side metadata-cache parse failures are surfaced to the parent for the bounded repair even if the original main-thread load was hashed.
- Low-level byte/opaque-URL callers **without a verified reload source** retain the original only to preserve their existing unfolded fallback contract; an uncacheable repaired WASM-original input is similarly retained only until needed. No new unverified fallback fetch is substituted. Physical reclamation still depends on the browser.

### Copy budget after the rewrite

| Ours, excluding metadata/runtime | LQO bytes | BT4 bytes | Lifetime |
|---|---:|---:|---|
| Old input + encoded replacements + graph + final model | 662708512 | 1297024440 | Necessarily coexist at old final serialization |
| New input + sole final model | 285062387 | 557313981 | While streaming the fold into the final output |
| Eliminated simultaneous serialization buffers | **377646125** | **739710459** | No longer allocated |
| Folded input handed to ORT | 189399671 | 370637638 | ORT's ingestion lifetime; rederivable original no longer rooted by our async frames |

At the cold total peak, LQO Gigacage was **182 MB**, versus 1180 MB at baseline. This is consistent with the final 189.4 MB model being the remaining large JS input, not proof that every browser/native copy has disappeared. BT4's cold peak still included 2749 MB WebKit malloc; its warm peak 5412 MB. The baseline linear-memory measurement cannot account for these totals by itself.

### Byte identity and regression evidence

[Committed goldens](../tests/fixtures/onnx_dequant_fold_golden.json) were generated using the unchanged `720aaf4a` fold **before this rewrite**. All **seven** complete folded protobufs match, including LQO, BT4, three Small variants and both Maia QDQ variants. Input lengths/hashes are pinned independently by committed manifests and the immutable release catalog.

- LQO folded SHA-256: `4cacdd89a647ba0cdb99009560f5628369e4188db5e0f7fbc81fe9fe0180b997`.
- BT4 folded SHA-256: `b6dda28f480ca9632e1eaa26440014cd7bc3177c83acdf4273f9b90e8e1fa206`.
- Allocation regression traps Float32Array construction and counts exactly **one** large Uint8Array output allocation. Numeric tests cover signed/unsigned int8/int4, scalar/per-axis/blocked scales, zero points, f16/f32 scale/output combinations, signed NaNs, native/manual half IO and shared quantized inputs.
- The existing real-ORT shipped Small folded-vs-in-graph inference/move/prior parity test **ran and passed**, not skipped. Full-buffer identity also preserves the already-tested old folded models for the large nets.
- CI fixture staging now includes all seven goldens, keeps hash-verified local staging zero-network, and includes Maia's committed manifest in the CI cache key. No model binaries are committed and this is not a new production streaming/hash path. An offline check of the seven local fixtures passed with `network:false`.
- Validation: required lint; 208 relevant tests plus 46 post-format lifecycle/recovery/IO checks; Svelte typecheck; **`npm test`: 2340 passed, 4 unrelated existing skips** (three Python-transformer fixtures and the LC0 C++ encoder source); product build. All changed TS/JS/JSON files pass targeted Biome check. The optional repository-wide `check:biome` also reports pre-existing import/format findings in unrelated soak/telemetry files; no broad cleanup is folded into this memory change.

## Step 4 and final controls

### Step 4: Analysis immediately retires unused big nets

**Choice: TTL 0 on all platforms, like Play.** Both Analysis maintenance/model switching (`disposeUnusedEngines`) and runtime/page teardown now cancel and immediately release unused big-net workers, rather than retain them for three minutes. There is no measured justification here for a desktop exemption, and Safari does not expose a reliable RAM threshold. Switching back sacrifices native session reuse; the verified model-byte cache remains warm. Multiple **explicitly active** Analysis rows remain supported. The shared pool's default TTL and its identity-checked detach/replacement protection are unchanged for other owners.

Tests execute the **actual extracted controller functions** with a real pool: T3 is disposed synchronously before acquiring BT4; no idle timer is created; both active T3/BT4 rows survive unused-engine cleanup; runtime teardown cancels/disposes all nets; the pool default remains 180000 ms outside this Analysis policy. Required lint, 19 relevant lifecycle/pool/override tests, targeted Biome, Svelte typecheck and the product build passed. Final **`npm test`: 2344 passed, 4 existing skips**, with all seven goldens and real Small inference parity running.

After step 4, same cold/warm protocol:

| Model | Cold peak / end MB | Warm peak / end MB | Load cold / warm ms | Midgame visits/s cold / warm |
|---|---:|---:|---:|---:|
| lqo | 2169 / 1822 | 5355 / 5193 | 13436 / 3657 | 12.3 / 38.1 |
| t3 | 2585 / 2563 | 5577 / 5529 | 11931 / 3966 | 63.5 / 57.3 |
| bt4 | 3768 / 3629 | 6989 / 6637 | 18455 / 8572 | 23.2 / 21.8 |

The soak adapter already retires each net immediately, so the Analysis-only TTL change is **not expected to reduce these isolated lab loads**. Differences versus step 3 are another indication of run-to-run native/GC/compilation variability, not evidence of an Analysis TTL effect. On the 24 GiB desktop host, the final cold LQO total peak had 1826 MB WebKit malloc and 308 MB graphics; Gigacage was absent from the harness's ≥5 MB category list at that instant. Large JS serialization copies are no longer an adequate explanation for the remaining total.

### Final `ortDequantFold=0` controls

The same product build, production-selected artifacts, CPU IO, graph optimization **`all`**, and cold/warm protocol were used. Every ready message was checked for **effective** `dequantFoldEnabled:false`, `graphOptimizationLevel:'all'`, `ortIoMode:'cpu'`; every step completed successfully and every warm load was an actual metadata-trusted hit. Only this explicit diagnostic query disables app folding; **the default remains enabled**.

| Model | Cold peak / end MB | Warm peak / end MB | Load cold / warm ms | Midgame visits/s cold / warm |
|---|---:|---:|---:|---:|
| lqo | 2534 / 2468 | 4813 / 4679 | 8592 / 2405 | 64.2 / 62.0 |
| t3 | 2844 / 2769 | 5649 / 5575 | 10834 / 3451 | 58.7 / 48.9 |
| bt4 | 2562 / 2482 | 6065 / 6050 | 15617 / 5032 | 12.1 / 13.0 |

BT4's cold no-fold peak is lower here, but it still exceeds 2 GB and its tiny-search speed is lower. LQO's no-fold cold peak is **higher** than the final folded run despite the smaller JS input, unlike the owner's earlier comparison. T3 is already plain f16 and its fold-off flag does not remove weight conversion. These one-pair observations, post-disposal allocator retention, and the wide LQO speed variation do **not** justify a universal fold-off memory/performance claim or a default change.

### Verification limits and remaining native cost

- Baseline → final sampled cold peaks: **LQO 3627 → 2169 MB; T3 1747 → 2585 MB; BT4 2800 → 3768 MB**. Warm reloads remain substantially higher. There is no claim of uniform total-footprint improvement; the deterministic wins are one normal LQO session instead of two, the eliminated serialization allocations, and no three-minute unused Analysis worker retention.
- The requested **sub-1 GB LQO / sub-2 GB BT4 targets are not achieved**, including the final no-fold controls. This is desktop Safari evidence, **not physical-iPhone validation**.
- The post-done window is still an observation, not proof of a steady plateau. Some totals continue to change after disposal. Investigate longer-tail JSC/WASM compilation, ORT runtime/allocator and GPU-driver retention before attributing the whole WebKit-malloc category to C++ tensor copies. The captured baseline **413 MB WASM linear-memory high-water** does not establish Carbon's proposed 4–5× native model-copy multiplier.
- The new folded models are still **189.4/370.6 MB** for LQO/BT4 and still enter ORT's normal native ingestion. Dropping JS roots does not shrink the WASM heap, release the native runtime, or guarantee immediate OS reclamation. Immediate Analysis retirement proves logical worker/pool lifetime, not a particular physical-memory drop.
- The quick T3/BT4 comparisons above show no consistent slowdown across cold/warm observations, but do **not establish a <2% performance bound**. An additional warmed, IO-only 10-second A/B and refreshed shader/heap instrumentation were prepared but could not be run during repeated shared-Safari pairing conflicts. The last recorded owned session was confirmed invalid; no other session or driver was terminated. Native graph-optimizer folding/fusion behavior is not inferred merely from the app-fold flag.
- Full-model goldens and the existing Small numerical/move/prior parity test are green. Small's real production Safari worker had already selected split-K, reported **gpu-capture** and completed eight visits in step 1; the final build retains that default and the preflight checks, but no new post-step-3 Safari Small worker probe was completed.

**Phase-1 code and four staged desktop measurement sets are complete.** No OPFS work, incremental/streaming hash redesign, model-format change, ORT patch, global fold disable, integrity weakening, deployment or navigation link was introduced. More native-memory evidence and a real-phone retest are needed before promising an iPhone fix.

## Phase 2: fresh-process measurements and per-net policy

### Corrections and implementation

- `4661d8e8` removes the blanket CPU-IO pin. Device-selected models request GPU capture; **t3 and BT4 use it**, while protobuf preflight rejects dynamic LQO **before** any probe session. Tests exercise one native creation for all three and check actual captured vs CPU IO. Small's captured default is unchanged. The independent perf-harness A/B showed the earlier CPU pin added ~1.1 GB to t3's lifetime peak, rather than its load peak; phase-1's “no consistent regression” interpretation was wrong.
- `41723dd8` adds explicit config: **fold off for LQO, BT4, t3 and Maia3, on for Small**, on every device. No size/RAM heuristic. Worker pin > URL/environment override > model default; `?ortDequantFold=1/0` forces either arm. Main/worker and cache-reuse identity tests cover both directions. Graph optimization remains `all` and is independent.
- All phase-1 integrity, one-shot parse repair, verified lazy reload, byte-identical folding, and immediate unused Analysis retirement remain. No LQO free-dimension override/capture, worker reuse, or production session/device-release-before-terminate was added.

### Method and provenance

New protocol: `lab/perf-harness`, `docs/perf_measurement_protocol_2026-11-17.md`, read-only sibling worktree. Its runner and `pidfp.c` were copied into this branch as `scripts/perf_matrix/`; **every Phase-2 memory figure below comes from it**. Local adaptations observe worker ready messages without changing init payloads, accept a fresh native-tool-owned Chrome CDP endpoint, serialize Chrome under benchlock too, record/reject Safari launchctl JSC overrides, and bound CDP waits. Pre-session Safari pairing conflicts yield the mutex and retry outside it; no foreign session is deleted.

Host: Apple M4 MacBook Air, 24 GiB, macOS 26.4.1; Safari 26.4 / Asyncify and Chrome 157.0.8103.50 / production JSPI. COOP/COEP isolated. **No physical-iPhone validation.** Normal Safari JSC settings were checked inside the lease. All GPU/search/speed groups used the supplied machine-wide `scratchpad/benchlock`.

Baseline is **actual `720aaf4a` source**, archived and product-built under this worktree's ignored `baseline-src`, not the CPU-pinned phase-1 build. Final engine source is `41723dd8` (later changes are measurement tooling/docs). Both use the same installed ORT 1.27.0 and the same manifest-pinned local model bytes; the archived build's generated model links point at those identical staged bytes. Independent isolated static servers use ports 61982/61981. No other checkout or its build was modified.

Each cold session has a fresh proxy origin/profile and empty model cache. Each warm load uses a **new window/tab and page process**, verifies the previous Safari PID exited, and fetches **zero model bytes** through the proxy. t3's actual served model is the Apple split-K artifact in both builds. Ready messages confirm **CPU IO for LQO, gpu-capture for t3/BT4**, `all` graph optimization, and the correct effective fold flag for every valid load.

Kernel `proc_pid_rusage` high-water counters eliminate sampling gaps. “Load” includes navigation through the two smoke searches; **lifetime includes disposal + a fixed 10 s settle**, and is not a load metric or guaranteed final plateau. MB here means **MiB (2²⁰ B)**. Tables show median [min–max], baseline → final. Repeats are round-robin across builds/models, not selectively rerun outliers. Cold n≈10 would be needed to resolve a 150 MB difference; these ≥5-session comparisons establish the large LQO/BT4 changes, not a narrow t3 equivalence bound.

Raw local evidence: `.local-dev-artifacts/mobile-memory-phase2/{safari,chrome}/`, `speed-r{0,1,2}.json`, `speed.json`, `parity-{lqo,bt4}.json`. `runs.jsonl` and per-session JSON retain validity inputs, exact counters, phases, wire bytes, worker settings and JSC compiler-thread CPU. The runner's `--report` regenerates summaries.

### Safari load and lifetime peaks

**Blocked measurement requirement:** baseline Maia3 warm has n=4; all others have n≥5. Balanced extra cohorts were attempted, not selective outlier reruns. Safari refused new sessions as already paired elsewhere from about **07:03 through 11:34 UTC**; waits were blocking, and the final coordinator yielded benchlock between attempts. After over four hours of refusals, only this session's recorded retry coordinator (Node PID 53773, verified idle with no child/session) was stopped. Its last owned Safari session was independently confirmed HTTP 404/invalid. No foreign session, driver or WebKit PID was terminated. **The requested n≥5 baseline Maia3 warm cell is not complete**, and the phase's measurement requirement must not be labeled fully satisfied.

| Net | Cache | n baseline/final | Load peak MB, baseline → final | Lifetime peak MB, baseline → final |
|---|---|---:|---:|---:|
| lqo | cold | 5/6 | 3092 [2534–3194] → 1344 [1282–1386] | 3788 [3594–4082] → 2549 [2200–2759] |
| lqo | warm | 5/6 | 2955 [2795–3104] → 1331 [1261–1373] | 3567 [3450–3843] → 2536 [2257–2580] |
| t3 | cold | 6/6 | 1723 [1626–1787] → 1735 [1672–1789] | 1757 [1731–1787] → 1752 [1736–1789] |
| t3 | warm | 6/6 | 1763 [1695–1788] → 1738 [1691–1818] | 1780 [1754–1788] → 1775 [1749–1818] |
| bt4 | cold | 5/5 | 3808 [3641–3897] → 1733 [1692–1820] | 3933 [3857–4031] → 1841 [1802–1878] |
| bt4 | warm | 5/5 | 3434 [3239–3712] → 1759 [1697–1826] | 3622 [3389–3720] → 1759 [1697–1826] |
| maia3 | cold | 5/5 | 1018 [968–1063] → 933 [917–976] | 1052 [1041–1123] → 2138 [1842–2220] |
| maia3 | warm | 4/5 | 997 [956–1034] → 938 [888–941] | 1028 [989–1065] → 2148 [1947–2243] |

**Important adverse result:** Maia3's new load peak is lower, but its lifetime peak is ~1.1 GB higher. The process remains busy with WASM compiler-thread work after disposal (median tail compiler CPU ~12.8 s baseline vs ~22.3 s final over the 10 s wall window). This is consistent with the Asyncify/JSC tier-up problem, **not attribution to a particular function or proof of CPU operator placement**. The explicit owner-selected no-fold default is retained; this Safari tradeoff needs follow-up. LQO also still grows after disposal. Native/JIT retention is not explained away as a live pool record.

Invalid/excluded: baseline LQO r0 aborted its WebDriver operation (no valid cold/warm pair); baseline Maia3 r3 warm navigation returned HTTP 408. The eight r5 sessions never launched because the extra-cohort command had a benchlock path typo (all retained as `session-failed`; the runner now validates the path). Later pre-session pairing conflicts produced no measured load, and are retained in the session-failure rows/logs. No successful high/low sample was discarded. Chrome had **no invalid measured loads**. The original 40-pair plan and each balanced extension remain recorded locally.

### Chrome renderer and GPU-process load peaks

n=3 per cell; a fresh native-tool-owned Chrome instance per cold/warm pair. GPU is the **instance's own GPU-process interval peak**, not a delta, not all system GPU memory, and not synchronized with the renderer maximum. Never add the two maxima and call that a measured simultaneous total. Renderer lifetime peaks equal its load peaks in these runs.

| Net | Cache | Renderer MB, baseline → final | GPU process MB, baseline → final |
|---|---|---:|---:|
| lqo | cold | 1224 [1222–1229] → 498 [482–498] | 392 [390–409] → 351 [339–364] |
| lqo | warm | 1210 [1208–1228] → 422 [416–431] | 377 [372–416] → 344 [332–354] |
| t3 | cold | 945 [944–952] → 925 [914–932] | 438 [433–444] → 439 [432–473] |
| t3 | warm | 804 [794–927] → 811 [797–815] | 442 [439–469] → 470 [444–470] |
| bt4 | cold | 1809 [1789–1829] → 907 [851–921] | 629 [623–645] → 523 [519–531] |
| bt4 | warm | 1767 [1762–1771] → 824 [782–838] | 654 [634–674] → 538 [535–550] |
| maia3 | cold | 454 [454–461] → 323 [321–325] | 266 [251–267] → 255 [254–267] |
| maia3 | warm | 400 [394–410] → 268 [257–271] | 237 [235–240] → 224 [223–233] |

UA resident samples agree with the direction: LQO ~400 → 214 MB, BT4 ~919 → 472 MB, Maia3 ~115 → 80 MB; t3 ~521 MB unchanged. UA bytes cover page/worker JS and WASM, **not GPU/native allocations or transient peaks**. `ForceEagerMeasureMemory` was enabled only for these Chrome memory runs. The page's after-dispose UA timeout is retained, not interpreted as zero.

### Parity and dedicated speed benchmark

Both final WebGPU no-fold candidates pass the **unchanged** `ort-profile.html?parity=wasm` ten-FEN shipping gates: LQO raw policy/WDL/MLH max abs **1.296875 / 0.008545 / 0.125**, legal-prior max **0.039005**, 8/10 best moves with only the two existing allowlisted mirror-pair flips; BT4 **0.293701 / 0.002930 / 0.96875**, legal-prior max **0.013007**, 10/10 best moves. No threshold or exception was widened. Backend is strictly `webgpu->webgpu`, not a silently accepted WASM candidate.

Maia3 relies on `spike/maia3-nofold`'s `docs/maia3_nofold_2026-11-17.md` plus existing tests as requested: its off/on WebGPU outputs were bit-identical across 1083 rows, both passed the existing browser release corpus, and **both failed the expanded WASM-reference corpus**. The latter was not waived or represented as a pass.

Speed is **not** the soak page's eight-visit throughput. Same dedicated `ort-profile.html`, forced WebGPU, synchronized CPU readback, 5 warmups + 15 invocations, 3 fresh document runs per build/net, under benchlock. Interleave baseline/final, reverse arm order in round 2. Explicit fold=1 baseline / fold=0 final because the raw-session bench does not select product per-net config. Presets use b8 for LQO/t3/BT4 and b1 for Maia3. The t3 timing preset is the original f16 graph, whereas soak selects Apple split-K; this does not isolate the transport/capture change. These are invocation times, not visits/s or playing strength.

| Net / batch | Baseline ms, median [run-median range] | Final ms | Final/baseline |
|---|---:|---:|---:|
| lqo / 8 | 68.09 [66.11–72.13] | 77.82 [77.75–122.47] | 1.143× |
| t3 / 8 | 40.14 [39.56–40.58] | 40.79 [38.17–41.83] | 1.016× |
| bt4 / 8 | 158.00 [148.60–164.57] | 166.14 [154.81–173.99] | 1.052× |
| maia3 / 1 | 10.19 [10.13–12.05] | 11.38 [10.58–11.93] | 1.117× |

Retain the 122.47 ms LQO outlier. This is a shared fanless host; benchlock excludes cooperating benchmarks, not all CPU work or thermal drift. The measured slowdown is a real tradeoff accepted by the owner, not an equal-speed claim. The Maia3 spike's paired experiment is stronger evidence for its marginal cost than this unpaired n=3 comparison.

### Reproduction, verification and remaining work

Build each source with `BUILD_SCOPE=product npm run build:client`; serve it with `scripts/serve_isolated_static.mjs`. Pass a cells JSON containing both per-cell `base` URLs and `build` labels to `scripts/perf_matrix/run_matrix.mjs --cells <file> --out <dir> --benchlock <scratchpad/benchlock>`. Use local pinned model files so the counting proxy sees every model byte; a direct CDN `assetBase` would bypass that evidence. For native-tool Chrome, open a fresh instance with `ForceEagerMeasureMemory`, obtain `get cdp-url`, and set `sessionDefaults.cdpUrl` for that one pair. The runner closes only that leased instance. No direct agent-browser CLI was used by this phase's caller.

Parity: `/lab/kernel-hillclimb/ort-profile.html?preset=<lqo|bt4>&parity=wasm&ortDequantFold=0&warmup=5&iters=15&profileInvokes=0`. Timing uses the same page without `parity=wasm`, and the explicit fold arm. Overlay with `node scripts/kernel_hillclimb/build_lab.mjs` after the product build.

Validation after review fixes: lint; targeted IO/policy/lifetime/worker tests; seven whole-model folding goldens and Small numerical parity; **10 runner regressions**; typecheck (zero errors/warnings); product build; final full suite **2361 passed, 4 existing skips**.

Codex closeout used `~/.pi/agent/skills/autoreview/scripts/autoreview --mode branch --base main --engine codex --model gpt-6.1-sol --thinking xhigh`. The first output failed schema validation for an absolute finding path; the same engine/model was retried with repository-relative-path guidance. Verified and fixed three real harness findings in separate commits: `ca14af0c` retains a completed Chrome cold load if the warm load throws; `33e5d436` reads Chrome's actual `rendererTopMb` done mark; `0d506f13` counts extensionless channel-v2 identity/Brotli model traffic by manifest-pinned raw hashes without counting runtime artifacts. None changes the reported successful load/lifetime peaks. The final same-model branch review (`autoreview-4.log/json`, with only path-format guidance added) exited **0: no accepted/actionable findings**. No speculative production rewrite was accepted, and no extra review was run after that clean result.

Remaining: finish the externally blocked baseline Maia3 warm repeat (resume the saved balanced matrix after Safari is available); investigate Maia3 and LQO's post-disposal Safari compiler/lifetime growth; physical-iPhone Jetsam validation; potentially larger cold cohorts if a 150 MB effect matters. LQO's final desktop load peak remains above 1 GB. BT4's observed desktop load/lifetime peaks are below 2 GB, **not a guarantee of phone safety**. No LQO dimension/capture override, worker reuse, release-before-terminate change, new model format, ORT patch, OPFS, hash redesign, cache-integrity weakening, deployment or push was introduced.

## Phase 3: LQO batch-one capture and Maia3 folding restored

### Mechanism and product policy

The independent `lab/perf-harness` follow-up, `docs/safari_omg_memory_2026-11-17.md`, corrects the “post-dispose” framing: on Safari 26 the OMG compiler burst **starts during search**. Per-node ORT WASM execution triggers it with CPU IO **and non-captured GPU IO**; capture largely avoids that executor path. Those measurements include 90 s laptop peaks of 4.8–9.7 GB, stack samples in `OMGPlan`/B3/Air, and owner-approved JSC flag controls. The flags are not a product option. Safari 27 on the mini behaves differently; neither hardware nor browser version was held constant, and this is not iPhone evidence.

`e879a375` makes LQO's product config explicitly `freeDimensionOverrides: { batch: 1 }`. The same map reaches worker reuse identity, declaration preflight and **only the captured GPU session options**. The shared resolver reads prototype IO getters, validates positive integer values, substitutes named symbols without changing numeric dimensions, and does not mutate the original metadata. Post-create IO must agree with the preflight shape: an unexpected numeric batch 8 is rejected rather than accepted as capture success.

The model is still `lqo_v2.f16.qdq8.onnx`: **95,662,716 bytes**, SHA-256 `4effd3079add6ab04043f066057f987b54e90e7e54cdf62ac0931b300769fc0e`. No model rewrite, alternate URL, hash, manifest, or cache entry is introduced. Normal capture creates exactly **one** native session. If capture/metadata/warmup fails, existing cleanup and verified lazy reload feed the ordinary **unpinned CPU-IO/WASM fallback**; that failure path may create a second session. It is not confused with a speculative normal-load probe.

**Search batch handling:** LQO's PUCT recommendation remains a **logical leaf batch of up to 8**. The effective GPU binding's fixed batch, not symbolic native metadata or `physicalBatchMode=logical`, governs physical execution. A larger logical request is split into serial **batch-1** invokes, preserving order and returning only real rows. Tests submit eight positions and observe one warmup plus eight 14,336-byte f16 input runs. The real parity probe submits the entire ten-position corpus in one logical call and observes **ten b1 readbacks**. Worker diagnostics report effective `ioMode` and `fixedBatchSize`; mini runs confirm `gpu-capture` / `1`.

`f75d6131` restores **Maia3 fold ON**, using the existing shared explicit page/worker config. Small stays on; LQO/BT4/t3 stay off. URL and worker overrides still force either fold arm. No Maia capture implementation, worker reuse, or session/device-release-before-terminate change was made.

### Full-position parity and dedicated speed

An optional `lc0Io=cpu|gpu-capture` mode was added to the existing `ort-profile.html`. It uses the production verified evaluator factory and synchronized b1 evaluation (encoding/postprocess included), not the soak page's eight-visit rate. Timing has no raw-output observer; **after timing only**, the parity path observes the existing binding's CPU readback and applies the page's unchanged gates against shipped, unfurled ORT WASM outputs. It does not patch the model, gate thresholds or exception lists. The ordinary raw-session benchmark is unchanged.

Final Chrome WebGPU capture, full **10/10 fixtures** including rule50/promotions: raw max abs policy/WDL/MLH **1.3359375 / 0.01806640625 / 0.09375**; zero non-finite outputs; max legal-prior difference **0.01649118**; **10/10 best moves**, no tolerated flips needed. All existing raw and legal/top-3 gates pass. Effective backend is `webgpu->webgpu`, IO `gpu-capture`, fixed batch 1; verified loader provenance confirms the original bytes/hash. This establishes the full-corpus release gate, not bit-identical WebGPU/WASM math. The other agent's direct b1 CPU-IO vs capture comparison was bit-identical on four positions; it is not mislabeled as our ten-position bit-identity test.

Dedicated timings: Chrome 157 / Apple M4, benchlock, 3 fresh document loads per arm, 5 warmups + 15 invocations, reversed arm order in round 2. Baseline is the phase-2 source build (`5ac94eb4`); final engine source is `e879a375` + `f75d6131`. Maia uses ordinary raw `ort-profile` b1 with explicit fold=0/1; LQO uses the production-evaluator b1 IO mode with fold off in both arms. Do not compare absolute times across those two benchmark layers.

| Benchmark | Phase 2 ms, median [run-median range] | Phase 3 ms | Final/baseline |
|---|---:|---:|---:|
| LQO production b1, CPU IO → capture | 20.685 [18.810–21.015] | 19.105 [18.765–19.870] | 0.924× |
| Maia3 raw b1, fold off → on | 10.555 [10.530–11.290] | 9.855 [9.565–10.150] | 0.934× |

### Memory protocol and builds

Use `scripts/perf_matrix/`: fresh page process per load, cold new profile/origin, warm fresh window with the old Safari PID gone and **zero model bytes on the wire**, kernel interval/lifetime peaks, normal JSC flags. MB = MiB. The short-run “load peak” includes the two test searches; lifetime includes **10 s** settle. Long rows below separate initial-load and **90 s search** peaks and use a predeclared **70 s** settle, not the short-run lifetime window.

Both builds are local product builds, with identical pinned model/runtime artifacts. Phase 2 was archived under this worktree's ignored `phase2-src`; only the same **lab-only single-load timed-search hook** and optional benchmark overlay were added to both builds. `labSearchSeconds=90` runs one sustained search inside the selected net step, rather than loading a second `heat` engine or contaminating a phase boundary with two loads. A controller regression verifies one load/search/disposal and a 90000 ms search budget. Cohorts are interleaved across builds/models, not selected by measured peak.

Mini transport uses the provided `mini.sh` plumbing with branch-isolated remote scripts in `~/perf-mobile-soak-phase3`, separate local state for each server/tunnel, and the **shared mini mutex** `~/perf-matrix/safari.lock`. Other agents' `~/perf-matrix/scripts` are not overwritten. The mini's existing driver **PID 36158 / port 4446 was never started, stopped or killed**. Secure context, COOP/COEP isolation, Safari UA, effective IO/fold and cache-wire assertions were checked for every completed load. A stale/foreign lock is never recursively removed automatically (avoids racing another stale-lock taker); it requires owner cleanup. The laptop uses the supplied benchlock, not the mini lock.

### Mac mini, Safari 27.0.1: completed short cohorts

M4 / 16 GiB, n=5 per cell, median [min–max]. These are **within-mini** comparisons; not a Safari-26 or iPhone substitute.

| Net / cache | Phase 2 load MB | Phase 3 load MB | Phase 2 lifetime MB | Phase 3 lifetime MB |
|---|---:|---:|---:|---:|
| LQO cold | 705 [689–716] | 625 [607–638] | 705 [689–728] | 625 [607–638] |
| LQO warm | 650 [607–684] | 651 [551–661] | 650 [612–684] | 651 [551–661] |
| Maia3 cold | 338 [329–411] | 460 [440–469] | 366 [355–411] | 468 [460–489] |

All **30 loads valid**; no exclusions. LQO's short warm peaks are indistinguishable at this sample count. On the mini, restoring Maia's expanded f16 weights **raises** the short memory peak, as expected when the Safari-26 burst is absent. The default nevertheless follows the owner's all-device policy for the Safari-26/iPhone target; no claim of uniform cross-version memory improvement is made.

### Mac mini: 90 s LQO searches (two per build)

| Build / repeat | Initial load MB | Search peak MB | Lifetime MB (70 s settle) | Wasm compiler CPU-s during search | Visits/s |
|---|---:|---:|---:|---:|---:|
| phase2 / 0 | 680 | 1822 | 1822 | 4.517 | 17.1 |
| phase3 / 0 | 613 | 1499 | 1499 | 0.549 | 31.4 |
| phase2 / 1 | 606 | 1756 | 1756 | 4.866 | 16.8 |
| phase3 / 1 | 621 | 1511 | 1511 | 0.401 | 31.3 |

All four valid; final runs report actual capture/b1. Final 10 s slices stay about **31–33 visits/s**; baseline slices vary roughly **10–21**. Mini GPU-process peak delta is about **1599–1607 MB baseline vs 136–138 MB final** (shared GPU-process delta, not private VRAM accounting). The large Safari-26 OMG burst was already absent in this mini baseline; these results show lower compiler activity and steady throughput, not proof about the laptop or iOS 26.

### Laptop Safari 26: externally blocked

The coordinator was configured under benchlock for LQO cold+warm n≥5, Maia3 cold n≥5, and the same two 90 s LQO searches per build. **No phase-3 laptop cell completed:** the existing port-4445 driver refused every session creation because Safari was paired with another WebDriver session. Pairing failures yielded benchlock and backed off outside it. Blocking waits continued from roughly **12:06 to 14:56 UTC**, including a final two-hour blocking shell wait. The idle, recorded owned coordinator (Node PID 10722) was then stopped only after verifying its command and that it had no child; it had created **zero Safari sessions**. Own mini server/tunnel PIDs 9073/9074/9164/9165 were stopped after verifying their commands; mini driver **36158 remained running**. No foreign session, driver or WebKit process was terminated. Owner/perf-harness measurements motivate the change but are not relabeled as this phase's baseline/final matrix. This required machine remains an explicit verification gap; saved specs/builds can be resumed after its owner frees Safari.

### Maia3 capture feasibility (inspection only)

The actual graph has three float32 feeds: `tokens [unk__0,64,12]`, `elo_self [batch_size]`, `elo_oppo [batch_size]`; outputs are `logits_move [batch_size,4352]` and `logits_value [batch_size,3]`. Fixed b1 or b11 is structurally possible by pinning **both** symbols consistently. But the current LC0 binding supports **one input**, not Maia's three, and one statically captured session cannot simply alternate b1/b6/b11. A shape-specific session/cache or other explicit capture strategy, extra model/session residency, warmup cost and parity/lifetime testing would be needed. Feed bytes alone are cheap (3080 B at b1, 33880 B at b11), but that does **not** establish cheap overall capture. Nothing was implemented.

Batch 8 remains a follow-up **candidate only**: the other agent measured **52.9 / 62.7 visits/s** (b8 capture) vs **24.9 / 20.7** (b1 capture), max |Δlogit| **0.132**, |ΔWDL| **1.95e-3**, |Δprior| **3.37e-3**, top move 4/4. These differences prevent a bit-identical claim; batch 8 is not this product policy and is not shipped.

### Reproduction and closeout status

Raw evidence: `.local-dev-artifacts/mobile-memory-phase3/{mini-short,mini-long}/`, `parity-lqo-capture.json`, `speed-r{0,1,2}.json`, `speed.json`; laptop specs/queue log are preserved separately. Mini summaries come from `mini.sh fetch`, with original counter time series retained. Its own lock is used by `mini.sh run`; no laptop Safari control is involved.

Build + overlay as before. Full captured gate: `/lab/kernel-hillclimb/ort-profile.html?preset=lqo&batch=1&lc0Io=gpu-capture&parity=wasm&warmup=5&iters=15&profileInvokes=0&ortDequantFold=0`. Baseline speed uses `lc0Io=cpu` on the phase-2 build. Timed memory search uses `/lab/mobile-soak/?steps=lqo&labSearchSeconds=90`; n=2 per build was chosen, stronger than a single long confirmation.

Final validation: lint, 98 relevant capture/controller/protocol/kernel tests and 12 runner regressions; full suite **2368 passed, 4 existing skips**; typecheck zero errors/warnings; product build; real full-corpus WebGPU gate. Branch Codex autoreview used the requested `--mode branch --base main --engine codex --model gpt-6.1-sol --thinking xhigh` with repository-relative-path guidance. One finding was verified: Chrome's pre-sampling cache/worker/start setup could throw before the renderer-refresh timer's cleanup scope. `a909855b` extends the `finally` scope and tests the real setup-failure path, preventing a stranded mutex. It does not affect successful counters or product sessions. The next whole-branch review (`phase3/autoreview-2.log/json`) exited **0, clean: no accepted/actionable findings**; no redundant review followed.

The laptop comparisons and laptop 90 s burst confirmation are **not complete**, and no iPhone safety or Safari-26 burst-removal claim is made from the mini alone.
