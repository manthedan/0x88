# Mobile memory, phase 1 — 2026-11-17

Companion to [mobile soak](mobile_soak_2026-11-16.md). Worktree/branch: `mobile-soak` / `lab/mobile-soak`; baseline `720aaf4a`. Desktop measurements below are **not iPhone validation**. The Carbon analysis is a source of hypotheses, not allocation evidence.

## Scope and invariants

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

## Remaining work / costs

Step 4 and final no-fold controls will be recorded next. The requested sub-1 GB LQO / sub-2 GB BT4 peaks are **not achieved**. Expanded f16 models still enter ORT, and significant native/WebKit cost and reload retention remain. Do not promise an iPhone fix, disable folding by default, weaken integrity, or embark on OPFS/streaming hash work based on these observations.
