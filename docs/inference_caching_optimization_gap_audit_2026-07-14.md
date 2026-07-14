# Inference, caching, and compression optimization gap audit

Date: 2026-07-14

## Executive conclusion

The project has already explored most obvious low-level neural inference levers:

- f32 and f16 ONNX Runtime execution;
- QDQ int8 models;
- uint8 activation and int8 weight `MatMulInteger` graphs;
- fixed and Relaxed SIMD WASM runtimes;
- WebGPU and experimental WebNN execution;
- custom WGSL transformer kernels;
- TVMJS-generated WebGPU runtimes;
- physical batching, speculative pipeline depth, deferred readback, and GPU legal-prior processing;
- int8 and int4 packed-weight experiments.

The remaining highest-value work is primarily system-level rather than another
precision format. After a second repository audit, the strongest opportunities
are:

1. representation-aware, SHA-only artifact delivery and a canonical Artifact
   Worker fast path;
2. elimination of duplicate model, manifest, pack, and evaluator work through
   verification metadata and short-lived single-flight maps;
3. completion of the search-native context work and production cross-search
   evaluation batching;
4. shared immutable WGSL resources followed by genuinely fused compact GPU
   output, gated on end-to-end measurements;
5. resumable content-hashed shards and external/shared model weights;
6. engine worker-copy, UCI-output, and pthread packaging repairs;
7. per-shape WGSL schedule tuning only after synchronization bottlenecks are
   isolated.

The recommended order is to improve delivery and warm startup first, then remove
runtime duplicate work. More MMI, Relaxed SIMD, generic int4, or isolated shader
micro-tuning should not be the immediate priority.

## Scope

This audit covers:

- LC0 transformer inference;
- browser search and neural-evaluation scheduling;
- model and runtime delivery;
- Cache Storage, persistent browser storage, and startup behavior;
- compression, sharding, and artifact deduplication.

It does not propose convolution-model work or changes to model strength/training.

## Existing work and current verdicts

### Stable or usable foundations

- ONNX Runtime remains the stable inference foundation across WASM and WebGPU.
- QDQ provides a substantial model-size reduction with strong numerical quality.
- LC0 search supports batching, evaluation caching, tree reuse, guarded early
  stopping, worker-owned sessions, cancellation, and session reuse.
- The custom WebGPU lane has physical batching, persistent buffers, generated
  kernels, reduced readbacks, and extensive parity and fixed-suite harnesses.
- Model Cache Storage validates expected size and SHA-256 before persisting bytes.
- Large assets can be hosted outside the Netlify application through the R2 asset
  base.
- The service worker caches the application shell while deliberately avoiding
  large ONNX, binary, pack, and Range responses.
- Brotli publishing and verification tooling exists for selected artifact classes.
- LC0web packs and TVM detached-weight artifacts demonstrate sharded and
  tensor-selective loading.

### Explored but not suitable for immediate promotion

#### MMI and Relaxed SIMD

The uint8/int8 `MatMulInteger` lane produced useful Ryzen speedups, especially
for larger transformer models, but:

- accuracy drift was worse than matched QDQ;
- Apple M4 performance regressed;
- Relaxed SIMD integer-dot dispatch was unavailable on tested hosts;
- selective floating-point retention behaved differently between t3 and BT4.

This remains research evidence, not the production default.

#### Generic int4

Symmetric group-wise int4 materially increased drift in tested BT4/TVM
artifacts. It should remain parked unless a different quantization scheme,
training-aware method, or genuinely fused runtime kernel changes the quality
and execution tradeoff.

#### Speculative search pipeline depth

Pipeline depths above one sometimes improved WGSL throughput, but altered search
ordering and reduced fixed-search parity. It is not valid promotion evidence.

#### Standalone GPU legal-prior processing

Moving legal-prior work to the GPU reduced readback bytes but did not reliably
improve wall-clock search. Any future attempt should fuse legal masking,
normalization, compact candidate generation, and readback reduction into one
coherent path.

#### WebNN

WebNN showed strong performance on supported Apple/CoreML paths, including a
large t1 win and a smaller BT4 QDQ win. It remains unsuitable as a stable
dependency while browser support requires flags and model/precision behavior is
inconsistent. It should be monitored rather than rebuilt now.

#### Broad WGSL kernel micro-tuning

Many tile sizes, workgroup shapes, loop unrolls, fusions, transposed layouts,
packed buffers, and smolgen variants have already been tested. Most either lost
throughput or failed drift gates. Future GPU work should focus on synchronized
end-to-end bottlenecks and systematic per-shape autotuning, not another broad
manual micro-tuning sweep.

## Ranked remaining opportunities

| Rank | Opportunity | Expected impact | Effort | Risk |
| ---: | --- | --- | --- | --- |
| 1 | SHA-only identity/Brotli representations plus canonical Artifact Worker caching | Very high | Medium | Low to medium |
| 2 | Model/manifest/pack/evaluator single-flight and duplicate-miss suppression | High | Low to medium | Low |
| 3 | Persistent model-cache fast path | High | Medium | Medium |
| 4 | Search-native inference context and compact neural-input keys | High on fast backends | Medium | Low to medium |
| 5 | Production LC0 evaluation broker | High for concurrent workloads | Medium to high | Medium |
| 6 | Shared immutable WGSL resources and retired-buffer cleanup | Medium to high | Medium | Low to medium |
| 7 | Fused legal-policy/WDL compact GPU output | Medium to high | High | Medium |
| 8 | Resumable shards and external/shared model weights | High bandwidth potential | High | Medium to high |
| 9 | Engine worker-copy, UCI protocol, and pthread packaging fixes | Medium to high | Medium | Medium |
| 10 | Per-shape WGSL tuning or SIMD pre-GPU bridge when profiling justifies it | Medium | High | Medium |

## Detailed findings

### 1. Content-addressed artifact delivery

The repository already documents a sound release structure:

```text
/channels/stable.json
/releases/<release-id>.json
/artifacts/sha256/<decoded-sha256>/identity
/artifacts/sha256/<decoded-sha256>/br/<encoded-sha256>
```

It is not yet the deployed repository structure. The first implementation pass added release generation and explicit Netlify
headers, but production gaps remain:

- the production R2 publisher and Worker must adopt the v2 representation map;
- encoded Brotli bodies are not negotiated in the hashed release path;
- body identity must not retain a logical filename, because equal bytes under
  variant names otherwise create duplicate R2 objects and edge keys;
- logical Worker resolution still performs repeated control-plane reads, HEADs,
  and Cache API lookups before a useful full-body response;
- stable heavy filenames must remain short-lived aliases rather than immutable
  body-cache identities;
- Cloudflare/R2 cache and encoding behavior still needs live end-to-end
  verification.

The intended policy should be:

- immutable content-hashed blobs and release manifests with one-year caching;
- short-lived, revalidated channel pointers;
- revalidated HTML and service-worker responses;
- exposed CDN timing/cache headers for production telemetry;
- write-once content-hashed keys.

This work improves bandwidth, reliability, rollback safety, and cache hit rate
without changing inference numerics.

### 2. Persistent model-cache fast path

The current model cache is correct-first but expensive on warm loads:

- every Cache Storage hit becomes a complete `ArrayBuffer`;
- SHA-256 is recalculated over the complete model;
- progress-enabled downloads retain chunks and then copy them into a second
  contiguous buffer;
- corrupt Cache Storage recovery deletes the entry but can reuse a request with
  `cache: "force-cache"`;
- there is no model-level LRU, quota-aware eviction, manifest-driven retirement,
  or persistent-storage request.

A fast path should store verification metadata in IndexedDB:

- content hash;
- expected and observed byte length;
- content-addressed URL or immutable release identity;
- verification timestamp and schema/runtime version.

For immutable content-addressed URLs, a matching metadata record can avoid
rehashing on every ordinary warm load. Downloads should preallocate when the
decoded byte length is known, and corruption recovery should use
`cache: "reload"` or an equivalent origin-forcing request.

Potential follow-up storage policies:

- request `navigator.storage.persist()` after explicit user model selection;
- quota-aware model LRU;
- per-entry clearing rather than whole-cache clearing;
- release-manifest retirement of unreachable models.

### 3. Search-native inference context

The current search-to-evaluator boundary performs duplicate work:

1. PUCT generates legal moves while preparing an expansion.
2. The legal moves are present in the evaluation context.
3. The LC0 adapter discards that prepared representation.
4. The evaluator regenerates legal moves and converts them to UCI.
5. The adapter rebuilds maps and converts moves back to action/policy indices.
6. Full FEN/history arrays are copied and reparsed even though the encoder needs
   a bounded history representation plus repetition metadata.

The proposed context should carry:

- already generated legal moves;
- action IDs and LC0 policy indices;
- compact history state for the required planes;
- repetition and rule metadata;
- optional canonical neural-input cache key.

This is likely the best low-risk end-to-end search optimization because it
removes CPU work and allocation without changing the neural model or search
algorithm. Benefits should be most visible on fast WGSL/WebNN backends and
low-visit searches where JS preparation is a larger fraction of latency.

### 4. Production LC0 evaluation broker

Broker prototypes already demonstrate:

- in-flight duplicate suppression;
- queue waiting;
- microbatch formation;
- cache integration;
- batch-fill telemetry.

The production LC0 worker does not yet broker requests across concurrent games,
analysis sessions, or arena jobs. A production broker could:

- combine requests into fuller physical batches;
- deduplicate identical positions in flight;
- enforce weighted quotas so interactive analysis is not starved;
- use bounded queue waits;
- preserve cancellation and per-search ordering.

This is primarily an aggregate-throughput optimization. It may not improve a
single isolated search, but it is highly relevant to arenas, simultaneous
games, and multi-line analysis.

### 5. Resumable content-hashed shards

Large ONNX files are monolithic. An interrupted 171 MB, 310 MB, or larger model
download must generally restart, and cache eviction applies to the whole model.

An experimental sharded format should use fixed 16 to 32 MB content-hashed
chunks with:

- per-shard hashes and byte counts;
- independent persistence;
- resume after interruption;
- explicit shard ordering and final model identity;
- bounded concurrent downloads;
- clear eviction semantics.

The first experiment should reconstruct a byte-identical ONNX model before
session creation. Longer term, an external-data or detached-weight runtime could
avoid reconstruction, but that has greater runtime compatibility risk.

### 6. Parity-preserving GPU/readback overlap

GPU queue synchronization and `mapAsync` waiting remain important custom-runtime
bottlenecks. Existing speculative search pipelining is not parity-safe because
it selects future leaves before earlier values are backed up.

A safe overlap design should:

- select a serially valid leaf batch;
- upload and submit it;
- overlap only preparation or transfer work that does not depend on its result;
- preserve result and backup ordering;
- use a bounded readback ring;
- avoid selecting future tree states from stale statistics.

This is a scheduler and evaluator-lifecycle problem, not only a shader problem.

### 7. Fused legal candidate readback

The scoped but unimplemented GPU postprocessing path would:

- consume mapped policy logits already resident on the GPU;
- gather legal policy entries;
- normalize legal priors;
- optionally select a bounded top-K;
- return compact candidates plus WDL in one readback.

The implementation is worthwhile only if it removes enough dispatch, CPU, and
readback work to beat the JS legal-prior control. Byte reduction by itself is
not a sufficient gate.

### 8. Per-shape WGSL schedule tuning

TVM default scheduling improved kernel time but not end-to-end performance.
There was no single dominant kernel, although fused MatMul families represented
most GPU time.

A bounded autotuner should target fixed hot shapes and search:

- tile dimensions;
- workgroup dimensions;
- vector widths;
- shared-memory use;
- output packing;
- fusion boundaries.

It should emit only variants that pass the existing f32/native parity ladder and
improve full evaluation or search, not merely isolated dispatch timing.

### 9. Shared weights across batch variants

Batch-1, batch-8, and batch-16 ONNX models frequently duplicate the same weights
inside separate files. Detached TVM parameter storage demonstrates that
cross-runtime weight sharing is possible, but standard ORT ONNX loading does not
currently exploit it.

This has potentially large bandwidth and storage benefits, but requires either:

- ONNX external data with carefully shared content-addressed files;
- runtime support for detached parameters;
- or a different fixed-batch strategy that avoids separate weight-bearing
  models.

Because session compatibility and browser loading behavior are uncertain, this
should follow the lower-risk caching and sharding work.

### 10. SIMD WASM pre-GPU bridge

CPU-side tasks that could be consolidated into a compact SIMD WASM bridge
include:

- LC0 history and 112-plane encoding;
- bounded history replay;
- legal move generation;
- action/policy-index construction;
- selected PUCT bookkeeping.

The desired shape is one compact bridge from chess state to upload-ready
features, not another heavyweight runtime or repeated large JS/WASM copies.
This should be attempted only after the search-native TypeScript context
establishes how much CPU time remains.

## Compression assessment

Compression is useful but is not the primary remaining bandwidth lever.

- JSON manifests compress extremely well.
- WASM commonly compresses to roughly one quarter of raw size.
- f16 packs and int8/QDQ models are entropy-dense and usually save only about
  10 to 15 percent under Brotli/gzip.
- The audited t1 f16 pack fell from 40.7 MB raw to 35.1 MB Brotli, a 13.9
  percent reduction.
- The measured t3 QDQ and MMI models compressed from about 87.4 MB to 76.6 MB.

The larger wins are:

- preventing duplicate downloads;
- persistent verified caching;
- resumable shards;
- shared weights;
- selecting smaller validated model variants;
- avoiding unnecessary cache rehash and memory copies.

Any production compression work must verify:

- `identity`, `gzip`, and `br` negotiation;
- correct `Content-Encoding`;
- decoded SHA-256 and byte length;
- Range behavior;
- CDN cache status and object metadata.

## Recommended bounded experiments

### Experiment 1: R2 cache and encoding canary

Publish temporary content-hashed examples of:

- one ONNX model;
- one LC0web shard;
- one WASM module;
- one engine data file.

Test cold and repeated requests with `identity`, `gzip`, `br`, and a small Range
request.

Success criteria:

- identical decoded SHA-256 values;
- correct encoding negotiation;
- correct `206` behavior where supported;
- repeat requests served from the edge cache;
- observable transfer reduction;
- no mutable URL marked permanently immutable.

### Experiment 2: Warm model-cache fast path

Use the default 20.6 MB t1 QDQ model and compare:

1. cold network load;
2. current Cache Storage hit;
3. metadata-trusted warm hit;
4. intentionally corrupted cache recovery.

Measure:

- network bytes;
- validation CPU time;
- total startup time;
- peak temporary memory;
- hash time;
- number and size of copies.

Success criteria:

- no warm network transfer;
- at least 50 percent lower warm validation time;
- correct corruption recovery using a forced reload;
- no unbounded storage growth.

### Experiment 3: Search-native context

Implement an opt-in adapter that reuses legal moves, precomputed action/policy
indices, and compact history state.

Test:

- representative FEN and explicit-history fixtures;
- 32- and 128-visit searches;
- batch 1 and batch 4;
- stable ORT and one fast WGSL backend.

Measure:

- move-generation calls;
- FEN parses;
- policy-index conversions;
- allocations;
- inference and search timing;
- visits per second.

Success criteria:

- identical encoded planes, priors, WDL, root visits, best move, and PV;
- meaningful reduction in CPU preparation time;
- no backend-specific regression.

### Experiment 4: Production evaluation broker

Run one, two, four, and eight concurrent searches through a shared evaluator.

Measure:

- aggregate evaluations per second;
- physical batch fill;
- duplicate suppression;
- p50 and p95 per-search latency;
- cancellation latency;
- lone interactive-search regression.

Success criteria:

- exact per-search result parity;
- improved aggregate throughput;
- bounded queueing for a single interactive user.

### Experiment 5: Sharded resume

Split one large t3-class model into content-hashed 16 to 32 MB shards. Abort
after approximately 40 percent, resume, reconstruct, validate, and create an ORT
session.

Success criteria:

- completed shards are not downloaded again;
- reconstructed SHA-256 matches the source model;
- ORT session creation succeeds;
- peak memory and startup cost are documented.

## Revised implementation sequence

### Phase 0: integrate the production baseline — completed

The inference and Artifact Worker work has been reconciled onto current `main`.
The combined baseline passed the full TypeScript/build/test gates before the
production Worker fast path was changed. The original checkpoint branches and
an external reconciliation backup remain available until the integrated branch
is published.

### Phase 1: artifact system v2

1. Run a live identity/Brotli/Range canary before changing publication.
2. Publish identity bodies by decoded SHA only.
3. Publish Brotli bodies by decoded SHA, encoding, and encoded SHA.
4. Record decoded/encoded hashes and lengths in release manifest v2.
5. Make Range requests select identity; send `Vary: Accept-Encoding` for
   negotiated full responses.
6. Cache logical release maps by release ID and revalidate only the channel
   pointer on a short interval.
7. Cache all aliases under one canonical representation-specific synthetic key.
8. Remove full-GET HEAD and duplicate Cache API lookups.
9. Store verified immutable object metadata so normal releases use HEAD rather
   than redownloading every carried-forward body; retain sampled/full audits.
10. Replace variant-level existence probes with one shared release catalog.

### Phase 2: cheap duplicate-work fixes

1. Add short-lived model and manifest single-flight maps without permanently
   retaining large byte buffers.
2. Collapse duplicate LC0 misses within batches, across batch sequences, and
   across concurrent in-flight calls.
3. Stamp final immutable artifact URLs into deployed model manifests.
4. Keep ordinary model-cache verification metadata fast path; retain periodic,
   diagnostic, and post-failure rehashing.
5. Verify production packs at manifest and selected-shard level; reserve
   per-tensor digests for generation tests, diagnostics, and spot checks.
6. Parse UCI output once, retain bounded diagnostics, and remove worker download
   chunk-array/response-clone copies.
7. Benchmark the completed search-native context rather than rebuilding it.

### Phase 3: WebGPU architecture

1. Share shader modules, pipelines, constants, and immutable head/input weight
   buffers across physical/deferred slots. Encoder-layer weights are already
   shared and must remain so.
2. Destroy superseded growth buffers after the relevant queue work completes.
3. Parallelize independent device, pack, helper, and pipeline initialization.
4. Separate GPU execution, queue wait, copy, and map time. Current `mapAsync`
   timing includes synchronization with outstanding GPU work and is not proof
   that mapping alone consumes the measured interval.
5. Fuse legal gather, normalization, compact policy output, WDL, and scalar
   heads only if it beats the existing JS and experimental GPU-legal controls
   in full evaluation/search while preserving all legal probabilities.
6. Add a fair, bounded production microbatch broker for concurrent searches.

### Phase 4: runtime packaging

1. Build external-NNUE Reckless on the faster persistent WASI/UCI path.
2. Externalize Viridithas's existing compressed network and prototype external
   Stockfish networks.
3. Repair Stockfish pthread bootstrap URLs and stage the actual ORT pthread
   sidecar; benchmark one, two, and four threads.
4. Test dynamic-batch and ONNX external-data graphs against fixed exports.
5. Introduce resumable fixed content-hashed shards for the largest models.
6. Compact search history and key evaluator caching by the actual bounded neural
   input while retaining richer repetition/draw state separately.
7. Run controlled Emscripten optimization/memory matrices; do not promote flags
   globally without size, latency, memory, parity, and sustained-NPS evidence.

## Implementation status

Completed in the inference worktree:

- v2 local release materialization with SHA-only identity objects, deterministic
  Brotli representations, decoded/encoded hashes and lengths, and write-once
  release manifests;
- explicit Netlify cache policy for channels, releases, hashed artifacts,
  logical models/engines, HTML, and the service worker;
- IndexedDB model verification metadata, periodic rehashing, preallocated model
  downloads, forced corruption reload, persistent-storage opt-in, model LRU,
  and model-level clearing;
- short-lived same-realm model/manifest single-flight;
- duplicate LC0 miss suppression within batches, across batch sequences, and
  across concurrent calls;
- search-native boards/history, legal moves, action IDs, policy indices, and
  canonical prepared-input propagation through ORT and custom WebGPU paths;
- production pack verification defaults to selected-shard hashes, with explicit
  per-tensor diagnostic verification;
- Reckless browser-API module/NNUE startup now runs concurrently, avoids cloning
  successful streaming-compilation responses, and preallocates decoded NNUE
  downloads when metadata provides the size.

Completed after Phase 0 integration:

- the production Artifact Worker accepts v1 and v2 release/channel manifests,
  negotiates identity/Brotli full responses, forces Range requests to identity,
  emits representation integrity/length headers, and returns `406` when no
  published representation is acceptable;
- body and HEAD Cache API entries now use canonical representation-specific
  keys, so equal bodies and logical aliases share one edge-cache identity;
- immutable release maps and short-lived channel pointers are cached separately;
- full GET misses perform one Cache API lookup and one R2 `get`, without a
  preliminary HEAD or a duplicate body-cache lookup;
- the production R2 publisher accepts v2 representation maps, deduplicates equal
  representation keys, uploads Brotli objects with `Content-Encoding: br`, and
  validates v2 representation metadata with HEAD and retains decoded full-body
  integrity checks until uploads persist a trustworthy R2 verification digest;
- browser model resolution accepts v2 channels/releases, prefers immutable
  Brotli representations for full loads, and continues to verify decoded length
  and SHA-256 metadata.

Still remaining:

- live identity/Brotli/Range canary validation through Cloudflare;
- migration of the default production release-manifest generator and deployed
  app manifests from v1 filename-keyed identity entries to v2 SHA-only entries;
- persisted R2 verification metadata that can safely replace ordinary
  carried-forward full-body integrity downloads with HEAD-only checks;
- replacement of variant-level existence probes with one shared release catalog;
- production cross-worker/session evaluation brokering;
- WGSL shared-head resources and fused final output;
- external-weight persistent runtimes and pthread packaging repairs.

## Work that should remain parked

Unless new runtime capabilities or quality evidence materially change the
tradeoff, do not prioritize:

- additional Relaxed SIMD work;
- automatic promotion of MMI;
- generic symmetric int4;
- speculative `batchPipelineDepth > 1` as a stable search mode;
- standalone GPU legal-prior processing;
- another broad manual WGSL micro-tuning sweep;
- WebNN production integration while browser support remains gated.

## Related repository references

- `docs/artifact_hosting_cache_strategy.md`
- `docs/lc0_pack_serving_compression_audit.md`
- `docs/lc0_fused_legal_topk_readback_plan.md`
- `docs/lc0web_custom_inference_checkpoint.md`
- `docs/lc0_tvm_whole_onnx_webgpu_probe.md`
- `docs/lc0_t3_qdq_webnn_2026-06-10.md`
- `docs/lc0_wasm_relaxed_simd_inference_design.md`
- `src/lc0/modelCache.ts`
- `src/lc0/onnxEvaluator.ts`
- `src/lc0/search.ts`
- `src/search/puct.ts`
- `public/sw.js`
- `netlify.toml`
