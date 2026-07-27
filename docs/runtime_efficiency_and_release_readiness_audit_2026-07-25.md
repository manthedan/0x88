# Runtime efficiency and release-readiness audit

Date: 2026-07-25
Scope: browser runtime throughput, asset delivery, and the work required before
publishing this repository to the chess-engine community.

## Executive conclusion

The project is in good shape. The test suite is green, production is live and
correctly configured (cross-origin isolation and Brotli both verified against
`https://0x88.app`), and the artifact, licensing, and provenance *tooling* is
more rigorous than most projects of this size.

The 2026-07-14 audit (`inference_caching_optimization_gap_audit_2026-07-14.md`)
already covers the neural-inference and delivery-layer opportunities well, and
its ranking still stands. This audit deliberately does not repeat it. It covers
what that audit did not: **CPU parallelism, runtime artifact selection, and the
concrete blockers to a public release.**

### What actually shipped from this audit

| Finding | Outcome |
| --- | --- |
| 1.2 ORT pinned to one thread in built workers | **1.85x** on the CPU-only neural path (Safari/Firefox) |
| 1.3 Asyncify runtime on the CPU path | −44% raw, −38% brotli, −36% session create |
| 1.4 16 MB transposition tables; Stockfish had none | 64 MB everywhere, Stockfish now gets a `Hash` |
| 1.5 Duplicate NNUE `.data` per SIMD variant | −173 MB tracked, no re-download on tier fallback |
| 1.9 Cache-policy gate red, unrun, and bypassable on the build path | Gate green and enforced on every deploy path; header policy de-duplicated to one generated source |
| 2.1 / 2.5 Licence, README | GPL-3.0-or-later, NOTICE, CONTRIBUTING; README rewritten |

### Four of this audit's own claims were wrong

Stated plainly, because a reader should trust the corrections as much as the
findings:

1. **Threading the CPU engines was called the largest remaining win.** It is
   not. A prototype confirmed 3-5x raw NPS and then showed that buys **zero
   extra plies** at fixed movetime. The already-banked SIMD work is worth more.
   See 1.1.
2. **436 MB of dead git history.** Those commits live on an unmerged branch; a
   `main`-only clone never sees them. Real saving from a rewrite: 1.5 MB. See
   2.3.
3. **Asyncify was said to cost per-call execution time.** It does not — the
   difference is ~1%. The win is transfer and startup. See 1.3.
4. **Berserk's network was called unresolved, and the engine was pulled from
   the live site over it.** The engine's own Makefile downloads that exact net
   during a normal GPL build and the binary cannot run without it, which makes
   it Corresponding Source. Restored. See 2.2.

The residual theme is that this project's remaining headroom is in *delivery
and startup*, not in search throughput — which is also where the 2026-07-14
audit landed by a different route.

## Part 1 — Runtime efficiency

### 1.1 Every CPU engine except Stockfish is single-threaded

**Status: MEASURED AND REJECTED. This audit's headline claim was wrong.**

> This section originally called single-threaded engines "the largest remaining
> win" and estimated 3-5x, "dwarfing the SIMD gains already banked". A threaded
> Stormphrax prototype was built to test that. The claim is **confirmed on raw
> NPS and refuted on strength**, and the recommendation is now *do not pursue*.
> The measurements are in
> [`threaded_emscripten_smp_prototype_2026-07-25.md`](threaded_emscripten_smp_prototype_2026-07-25.md);
> the summary is at the end of this section. The description of the current
> state below is still accurate — it is the conclusion drawn from it that was
> wrong.

`src/lc0/engineCatalog.ts` assigns `maxThreads: 1` to Reckless, Viridithas,
Berserk, PlentyChess, and Stormphrax (lines 171, 195, 219, 243, 267). Only
Stockfish gets `maxThreads: 32` (line 141).

This is not an oversight in the catalog — it is accurate. The builds really are
single-threaded:

- `scripts/build_plentychess_emscripten.mjs:134` — `-s USE_PTHREADS=0`
- `scripts/build_stormphrax_emscripten.mjs:92` — `-s USE_PTHREADS=0`
- All three Emscripten engines additionally compile a sync-search macro
  (`-DBERSERK_SYNC_SEARCH`, `-DPLENTY_SYNC_SEARCH`, `-DSP_SYNC_SEARCH`), and
  PlentyChess adds `-DTB_NO_THREADS`, so upstream's real threaded search is
  compiled out rather than merely limited to one thread.
- `src/lc0/recklessEngine.ts:552` and `src/lc0/viridithasEngine.ts:331`
  hardcode `setoption name Threads value 1`.
- `src/lc0/engineProvision.ts` constructs PlentyChess and Stormphrax with an
  explicit `threads: 1`.

On an 8-core laptop, selecting any engine other than Stockfish uses about 12% of
the machine.

Everything needed to fix this on the browser side is **already done**:

- Cross-origin isolation is live in production. Verified directly:
  `cross-origin-opener-policy: same-origin` and
  `cross-origin-embedder-policy: require-corp` are present on `https://0x88.app`,
  so `SharedArrayBuffer` and Emscripten pthreads are available.
- `src/lc0/resourceBroker.ts` already implements per-engine CPU budgeting,
  exclusive/shared policies, an eco/balanced/max dial, and a calibrated-threads
  hook. It currently has no multi-threaded engine to allocate to.

The remaining work is on the build side, and it splits cleanly by language:

- **Berserk, PlentyChess, Stormphrax (C++ / Emscripten)** — these have native
  Lazy SMP. `-pthread -sPTHREAD_POOL_SIZE=N` plus dropping the sync-search macro
  is the well-trodden path; Stockfish.wasm ships exactly this way. This is the
  high-value, moderate-risk work.
- **Reckless, Viridithas (Rust / WASI)** — genuinely harder.
  `docs/reckless_threaded_wasm_feasibility.md` analyses this properly and
  concludes root-split workers before shared-memory threading. That conclusion
  is sound and should stand. Note that the doc is scoped to Reckless only; the
  C++ engines were never given the same analysis, and the conclusion does not
  transfer to them.

#### What the prototype measured

A `-pthread` Stormphrax build (sync-search macro dropped, upstream Lazy SMP
restored, all other flags matched) was benchmarked in Node against the shipped
single-threaded build, `movetime 1500`, median of 5 interleaved repeats:

| Metric | T=2 | T=4 | T=8 | Verdict |
| --- | ---: | ---: | ---: | --- |
| Raw NPS | 2.0x | 3.0-3.5x | 4.0-4.4x | claim **confirmed** |
| Time to fixed depth (256 MiB hash) | — | — | 1.4-2.0x | **weakened** |
| **Plies reached in a fixed movetime** | ~0 | ~0 | **~0, and −1 to −2 on kiwipete** | **refuted** |

Threaded-at-T=1 matched baseline within 1%, so pthreads cost nothing by
themselves. But nodes-to-depth-18 scaled almost linearly with thread count
(kiwipete: 1.3M → 9.6M at T=8) while wall time to that depth *increased*. The
extra cores overwhelmingly duplicate work. Raising the hash from 64 MiB to
256 MiB recovered part of the time-to-depth loss — TT pressure is a real
co-factor — but the plies-at-fixed-time table stayed flat either way.

**The framing this audit used was backwards.** SIMD gains are per-node
speedups, which convert to depth roughly one-for-one. Lazy SMP gains are node
*count* gains that mostly do not. So the **+14% to +64% already banked from
SIMD is worth strictly more than the +330% available from threading** — the
opposite of the ordering this audit originally asserted.

The one regime where threads paid was `endgame-rook`, 19 → 24 plies at T=8,
consistent with endgames rewarding wide shallow parallel search. A
threads-in-endgames-only variant is the only version of this idea the data
supports, and it is not obviously worth the complexity.

Shipping it would also hit a genuine blocker: `src/lc0/stormphraxEngine.ts`
loads the glue via `importScripts` into a Blob-URL worker, and the threaded
glue resolves `_scriptName` to the outer harness blob rather than the
Emscripten glue, so pthread helpers would load the wrong script. All the
measurements above are from Node, which sidesteps this; the browser path is
untested and would need a worker-bootstrap rewrite. Determinism also
disappears at T≥2, which would make this repo's bestmove-pinned gates flaky.

**Recommendation: do not pursue.** Leave the engines single-threaded.

#### A real bug the prototype found — fixed

Building with pthreads exposed an existing latent defect, unrelated to
threading. `patches/stormphrax-emscripten.patch` read the 53 MB NNUE into a
plain `std::vector<std::byte>`, and the shipped single-threaded build satisfied
the loader's SIMD alignment requirement only by allocator luck. The patch now
uses Stormphrax's `alignedAlloc`/`alignedFree` pair with
`util::simd::kAlignment` (16 bytes on the wasm SIMD path), rounds the allocation
size as required by the allocator, and fails explicitly on allocation or read
failure. A rebuilt relaxed-SIMD runtime completed both smoke searches; the
loader would reject startup with `NetworkLoader: Unaligned pointer` if the
invariant did not hold.

### 1.2 ORT WASM is pinned to one thread in production workers

**Severity: high. Effort: low. Status: FIXED — `34377d3`.**

`src/nn/ortRuntime.ts:322-328` defaults ORT's WASM thread count to `'1'` inside
production-built workers:

```js
const builtWorker = !isBrowserMainThread && !isNode && typeof document === 'undefined'
  && (import.meta as unknown as { env?: { PROD?: boolean } }).env?.PROD === true;
```

The comment explains why: ORT's threaded WASM boot deadlocked inside a bundled
worker because the Emscripten pthread helpers were spawned from the chunk's own
`import.meta.url`, re-executing the app's worker module instead of the pthread
stub.

That diagnosis was correct at the time. But the timeline suggests the
workaround is now stale:

- The guard landed **2026-06-11** (commit `2cc2f65`, "Fix ORT production worker
  wasm boot").
- Self-hosted ORT pthread sidecars were pinned **2026-07-11 / 2026-07-14**
  (`180a4a4`, `0157f71`), a month later.
- `scripts/check_ort_runtime_assets.mjs` now explicitly verifies the bootstrap
  markers (`new Worker(new URL(import.meta.url)` and `name: "em-pthread"`) in
  the staged glue, and `browserOrtWasmPaths()` hands ORT explicit same-origin
  glue and wasm URLs specifically so that "helper workers re-import that exact
  staged module".

In other words, the root cause the guard works around appears to have been
fixed by later work, and nobody went back to remove the guard.

This matters because the WASM execution provider is the fallback for **every
browser without WebGPU** — Safari and Firefox users were getting
single-threaded neural inference on machines with 8+ cores.

**Retested 2026-07-25 — the guard was stale, and the fix is not the obvious
one.** Driving the actual built `searchWorker` chunk under COOP/COEP, threaded
ORT boots cleanly at 2/3/4/6/8 threads with no deadlock, and outputs match the
single-threaded run (top-5 policy moves identical and in order; a ~4e-6
difference in the top prior, the expected f32 reduction-order variation).

Measured, median of 20 evals, t1-256x10 qdq8, 10-core:

| Threads | CPU-only pair | Asyncify + WebGPU |
| ---: | ---: | ---: |
| 1 | 61.3 ms | **11.6 ms** |
| 2 | 43.4 ms | — |
| 4 | **40.1 ms** | 18.0 ms |
| 6 | 54.0 ms | — |
| 8 | 44.2 ms | — |

The trap: on the WebGPU path, threads are a **55% regression**, and
`numThreads` is worker-global and fixed at ORT init. A blanket `auto` default
would have slowed down every Chrome user to speed up Safari and Firefox.

The default is therefore conditional on the runtime artifact: a built worker
committed to the CPU-only pair (no `navigator.gpu`, or an explicit `?ep=wasm`
pin) gets `auto`; anything that could still escalate to WebGPU stays at 1; a
thread without cross-origin isolation stays at 1. End-to-end after the change:
**64.6 ms → 35.0 ms, a 1.85x speedup** on the path Safari and Firefox use, with
Chrome/WebGPU unchanged at 1 thread.

`defaultAutoThreads()` also changed from `clamp(hc-1, 2, 4)` to
`clamp(floor(hc/2), 1, 4)`. ORT is classed `resourceClass: 'gpu'` in the
catalog and so never draws down the CPU broker budget; capping it at half the
machine keeps a co-running Stockfish from being over-subscribed, and the old
floor of 2 was wrong on a single-core device.

Still unverified: a genuinely GPU-less browser. `agent-browser` is
Chromium-only, so the CPU-only artifact was reached through the explicit
`ep=wasm` pin rather than the `!webgpuAvailable()` branch — the same decision
and the same downstream code, but the last mile on real Safari/Firefox is
untested.

### 1.3 The CPU-only path uses the asyncify ORT build

**Severity: medium. Effort: low-medium. Status: FIXED — `8fbc108`.**

`browserOrtWasmPaths()` (`src/nn/ortRuntime.ts:336-354`) pins
`ort-wasm-simd-threaded.asyncify.wasm` for every path, and
`ORT_PTHREAD_WASM_FILE` (line 18) hardcodes the asyncify binary.

Sizes:

| Artifact | Size |
| --- | ---: |
| `ort-wasm-simd-threaded.asyncify.wasm` | 24.2 MB |
| `ort-wasm-simd-threaded.wasm` | 13.5 MB |

Asyncify instruments the entire module for stack unwinding. It is genuinely
required for the JSEP/WebGPU path, which suspends across GPU operations, and is
unnecessary for an `executionProviders: ['wasm']` session.

**Correction to an earlier draft:** this audit originally claimed the asyncify
build also costs per-call execution time. Measured, that is not true — Node
steady-state inference differs by about 1% (66.3 ms vs 66.9 ms median). The real
win is transfer and startup:

| | asyncify | CPU-only | delta |
| --- | ---: | ---: | ---: |
| raw (glue + wasm) | 24.30 MB | 13.50 MB | −44.4% |
| brotli (deploy sidecar) | 3.57 MB | 2.21 MB | −38.1% |
| session create (Chrome, t1 qdq8, 1 thread) | 516 ms | 329 ms | −36% |

Policy outputs were bit-identical across the two builds.

This compounds with 1.2: the WASM path is the fallback for browsers without
WebGPU, and it is currently paying both the single-thread penalty and the
asyncify penalty at the same time.

Hazard to respect when fixing: ORT's WASM artifact is worker-global once
initialized — `ortRuntime.ts` already throws *"ORT WASM is already initialized
with a different artifact; select the fallback in a fresh worker"*. A worker
that starts on WebGPU and falls back to WASM must decide its artifact before
first init, not after.

### 1.4 Transposition tables are left at 16 MB

**Severity: medium. Effort: trivial. Status: FIXED — `983390a`.**

- **Stockfish never receives a `Hash` setoption at all.**
  `src/lc0/stockfishEngine.ts` defines `skillLevelCommand` and `threadsCommand`
  (lines 133-137) but contains no hash command — `grep -n "Hash"` on that file
  returns nothing. Stockfish therefore runs at its built-in 16 MB default. Every
  other adapter in the project has a `hashCommand` helper.
- `src/lc0/engineProvision.ts:23` sets `CPU_ENGINE_DEFAULTS = { depth: 4, hashMb: 16 }`
  for all other CPU engines.

The Emscripten builds are compiled with `ALLOW_MEMORY_GROWTH=1`,
`INITIAL_MEMORY` of 256-512 MB, and `MAXIMUM_MEMORY=2147483648`, so there is
substantial headroom. A 16 MB table is a measurable strength loss on analysis
searches past roughly depth 12.

Constraint: Arena and Analysis can hold several engines alive at once, so the
per-engine default must not be sized as though it were the only engine on the
page.

Also note the clamps are inconsistent and should be reconciled:
`viridithasEngine.ts:64` and `recklessEngine.ts:63` clamp to 1024 MB, while
`berserkEngine.ts:26`, `plentychessEngine.ts:26`, and `stormphraxEngine.ts:33`
clamp to 33554432 MB (32 TB), which is not a meaningful bound.

### 1.5 Identical NNUE `.data` sidecars are duplicated per SIMD variant

**Severity: medium (bandwidth). Effort: low. Status: FIXED — `8de36e4`.**

The Emscripten `.data` preload files are byte-identical across SIMD variants.
Verified by hashing:

| Family | Variants | Each | SHA-256 |
| --- | ---: | ---: | --- |
| PlentyChess | 3 | ~60 MB | `691efaca…69d4d4` |
| Stormphrax | 2 | ~53 MB | `04d651e0…33f2cf` |
| Berserk | 3 | ~24 MB | `9b84c340…5c5084` |

Consequence: when the runtime falls back from the relaxed-SIMD variant to
SSE4.1 or scalar — which `resolveDefaultPlentyChessVariantAssetFallback` and its
siblings do automatically on a missing artifact — the browser re-downloads tens
of megabytes it already holds. R2 and the CDN also store two to three copies of
identical bytes.

Fix: point every variant of a family at one canonical `.data` via Emscripten's
`locateFile`. The variant modules already thread an explicit `dataUrl` through
to the adapters, so the change is contained. The build should *verify*
byte-identity rather than assume it, and fail loudly if a rebuilt variant's data
diverges.

Related: Reckless embeds its net in the wasm — `reckless-relaxed-simd128.wasm`
is 64.5 MB — while `reckless-browser-api-simd128-external.wasm` is 1.26 MB with
an external net. Making the external-net variant the default would let all three
SIMD tiers share one downloaded network.

### 1.6 App-origin fallback assets cache at the outer Cloudflare edge

**Severity: low. Effort: dashboard-only. Status: FIXED — Cache Rule `cache_app_origin_engine_fallbacks`.**

The original finding was too broad. Re-measurement after deployment separates
three paths:

| Path | Observed cache behavior |
| --- | --- |
| `https://0x88.app/` | `cf-cache-status: DYNAMIC`, intentionally; HTML is mutable |
| `https://0x88.app/_app/immutable/*` | first request `MISS`, second request `HIT` |
| `https://0x88.app/ort/*` | first uncached `GET` is `MISS`; the second is `HIT` |
| `https://assets.0x88.app/<logical artifact>` | `cache-status: lc0-artifact-worker; hit` |

Engine and model downloads normally use the `assets.0x88.app` Artifact Worker
and R2. Versioned application bundles also cache correctly at Cloudflare. A
zone Cache Rule now closes the remaining app-origin fallback gap for `GET` and
`HEAD` requests under `/ort/*`, `/models/*`, and `/engines/*`.

The rule sets cache eligibility and explicitly keeps browser TTL in
`respect_origin` mode; it does not override the origin's TTL or cache key. This
prevents Cloudflare's default four-hour Browser Cache TTL from inflating the
origin's one-hour ORT policy.
Production canaries for an ORT wasm and a model manifest each changed from
`MISS` to `HIT` on the second
identical `GET`. A separate HTML canary remained `DYNAMIC`, so mutable pages and
channel pointers are outside the rule's scope.

The desired rule is now versioned in
`cloudflare/app-origin-cache-rule.json`. The idempotent
`npm run cloudflare:cache-rule:check` and
`npm run cloudflare:cache-rule:apply` commands compare or reconcile it through
the Rulesets API while preserving unrelated rules. They accept
`CLOUDFLARE_API_TOKEN`; on macOS they can instead read the scoped token from
Keychain service `lc0-cloudflare-cache-rules`, so no credential enters the
repository.

### 1.7 R2 artifacts use stored identity and Brotli representations

**Severity: medium. Status: FIXED — stable release `2026-07-27.stored-brotli.1`.**

The stable channel now uses `lc0_browser.artifact_channel_manifest.v2`, pointing
to a representation-aware immutable release. Its 87 unique decoded bodies have
174 physical R2 objects: one identity representation and one Brotli
representation per body. Object keys include both the decoded SHA-256 and the
representation encoding/hash, so neither the Worker nor Cloudflare recomputes
Brotli.

Across the release, identity representations total 2,912,029,084 bytes and
Brotli representations total 2,434,195,089 bytes: 477,833,995 bytes (16.4%)
less transferred data. Live logical-alias checks include:

| Artifact | Identity | Stored Brotli | Reduction |
| --- | ---: | ---: | ---: |
| Berserk NNUE preload | 25,201,924 | 17,602,380 | 30.2% |
| Stockfish wasm | 113,007,340 | 74,208,176 | 34.3% |
| LC0 QDQ model | 20,558,031 | 18,412,665 | 10.4% |

Production validation confirmed `Content-Encoding: br`, distinct encoded and
decoded SHA-256 headers, warm Worker cache hits, and `206 Partial Content`
identity range responses. A full-body canary downloaded both representations
of the Berserk preload; the identity body and decoded Brotli body each matched
the release's decoded SHA-256.


### 1.8 Move generator: allocation in the scan loops

**Severity: low. Status: FIXED — measured, +49% overall.**

> An earlier draft of this section was titled "The move generator is not a 0x88
> move generator" and rated the credibility of the repository name as **high**
> severity, recommending either a rewrite or renaming the project. That was a
> branding opinion in a runtime-efficiency audit, and it argued for changing
> correctness-critical code on a benefit this same document said was zero. A
> project name is not a specification of its internals. The reasoning is
> withdrawn; what follows is what the measurements actually support.

`src/chess/movegen.ts` is a 64-entry mailbox of two-character strings. The
representation is not the problem. The problem was allocation inside loops that
run on every legality test:

- `isSquareAttacked` built its comparison strings inline (`` `${by}n` ``) — one
  allocation per square probed, and it probes on every pseudo-legal move;
- `kingSquare` built `` `${color}k` `` per call and scanned with `findIndex`;
- queen generation rebuilt `[...BISHOP, ...ROOK]` — an eight-entry array of
  arrays — for every queen, on every call;
- `KNIGHT.forEach(([df, dr]) => …)` allocated a closure and destructured per
  direction;
- `fileOf`/`rankOf` used `%` and `Math.floor` rather than `& 7` and `>> 3`.

The scan-loop allocations were hoisted or replaced. The second pass removed the
larger allocation: `legalMoves` now applies each pseudo-legal move directly to
the existing 64-square array, checks the king, and restores every touched
square in a `finally` block. Castling, en-passant, capture, and promotion state
are restored explicitly; the public immutable `makeMove` contract is unchanged.
The legal-move result also compacts its existing pseudo-move array rather than
allocating another array through `filter`.

Correctness is covered by the existing perft gate (startpos depth 5 =
4,865,609 nodes; kiwipete depth 4 = 4,085,603) and a board-restoration
regression covering castling, en-passant, and promotion candidates.

| implementation | positions/sec | change from baseline |
| --- | ---: | ---: |
| before | 146,226 | — |
| scan-loop fixes | 169,638 | +16% |
| clone-free legality filter | 217,912 | **+49%** |

This still does not move the neural product materially. The neural lane is
GPU-bound at roughly 1.75 ms/position against roughly 4.6 µs for a
`legalMoves` call, so move generation remains well under 1% of search time.
The final change is justified by removing an avoidable board clone per
pseudo-legal move while retaining the same 64-square representation and perft
contract—not by a branding-driven rewrite.

### 1.9 The deploy cache-policy gate was failing, and nothing ran it

**Severity: medium. Effort: trivial. Status: FIXED in this pass.**

`npm run deploy:cache-policy-check` exited 1 on `main`:

```
netlify.toml has one-year immutable Cache-Control outside content-addressed artifacts: /releases/*
public/_headers has one-year immutable Cache-Control outside content-addressed artifacts: /releases/*
```

The config was right and the checker was stale. Release manifests are named
`/releases/<date>.<release-sha>.json` and `publish_content_addressed_release.mjs`
throws `Refusing to overwrite immutable release manifest` on any rewrite, so
they are write-once both by construction and by enforcement — exactly the
condition one-year `immutable` requires. `/releases/*` was added to the
allowlist rather than downgrading the header.

It went unnoticed because the check was referenced only from `package.json` and
one doc: no test, no CI step, and no release step ever invoked it. It is now
called from `netlify_r2_release.mjs` before the build (header policy is a
property of the source config, so it should fail before a build is spent) and
covered by `tests/deploy_cache_policy.test.mjs`, which asserts each negative
case actually trips.

Two further defects surfaced while reconciling the files:

- `netlify.toml` declared `/channels/*`, `/artifacts/sha256/*`, and `/*.html`
  **twice each**. The `/artifacts/sha256/*` pair was not equivalent: the first
  block omitted `Access-Control-Allow-Origin`, `Cross-Origin-Resource-Policy:
  cross-origin`, `Access-Control-Expose-Headers`, and `Timing-Allow-Origin`,
  which the second set. Which of the two won depended on Netlify's merge
  precedence rather than on anything the repository states. In production this
  path is mostly dormant — hashed blobs are served from `assets.0x88.app`, not
  the Netlify origin — but it is the fallback the config exists to describe, and
  a `Cross-Origin-Resource-Policy` that resolves to `same-origin` under the
  site-wide `COEP: require-corp` is precisely the kind of defect that appears
  only once traffic shifts. Duplicates removed, richer block kept.
- `netlify.toml` and `public/_headers` are both applied and had drifted into
  near-duplicates of each other, with no documented precedence between them.

### The gate did not cover the deploy path it claimed to

The first wiring called the checker from `netlify_r2_release.mjs` only. But
`netlify.toml` configures the build command as `npm run build:netlify:r2`, so
every **automatic** Netlify build ran `scripts/build_netlify_r2.mjs` and never
touched the checker. A policy violation could reach production through CI while
the manual release path reported a green gate. The check now runs from the build
script, verified by breaking the policy and confirming the build aborts.

### Duplicated policy was the root cause, so the duplication was removed

Four rounds of adversarial review found fourteen defects, and after the first
two, every one of them was in the *checker* rather than in the config it
guarded. The checker had become a hand-rolled TOML parser, and each round found
another corner of the grammar it got wrong: case-sensitive comparisons, repeated
`_headers` fields concatenated per RFC 7230, `#` comments read as active policy,
quoted `max-age="31536000"` arguments, documented triple-quoted multiline
values rejected as garbage, ordinary `[context.production]` tables absorbed into
the preceding block.

That density of defects was a signal about the design, not the implementation.
The entire cross-file comparison existed only because the policy was stated
twice. So it is now stated once: `netlify.toml` is the single hand-edited
source, `public/_headers` is generated from it by
`scripts/generate_netlify_headers_file.mjs`, and the check fails if the
generated file has drifted. The published tree keeps its `_headers` artifact for
static hosts that read one; the cross-file comparison, the list-header
allowlist, and the second parser are gone. Generation was verified to lose
nothing: every path and header value from the hand-maintained file is present
and unchanged, and `/lab/webgpu-lc0-diag/*` is now covered by both mechanisms
rather than one.

Two lessons worth keeping. A config gate that pattern-matches its own config
format encodes the author's assumptions about that format rather than the
format's rules, and it fails in the direction of confidence. And when a
validator keeps growing to reconcile two sources of truth, the validator is
rarely the thing that needs fixing.

The two remaining caching items, 1.6 and 1.7, are unaffected — neither is
fixable in this repository's config.

## Part 2 — Release readiness

### 2.1 No LICENSE — resolved in this pass

The repository had no licence file at all, which meant default
all-rights-reserved: no community member could legally fork or contribute, while
the project simultaneously distributes GPL-3 engine binaries.

Resolved: the project is licensed **GPL-3.0-or-later**, matching every bundled
engine (Stockfish, Berserk, PlentyChess, Stormphrax, Viridithas) and removing
any argument about whether the app plus its engines constitute a combined work.

### 2.2 Berserk network provenance — REVISED, and the original finding was wrong

This audit originally reported the Berserk network as unresolved and had the
project stop distributing it, on the strength of
`engine_artifact_distribution.md` recording "no standalone license file found in
`jhonnold/berserk-networks` during intake."

That intake note was accurate but incomplete, and the audit inherited its
conclusion without re-deriving it. Nobody had checked how the engine *consumes*
the network. Berserk's own `src/Makefile` names `berserk-9b84c340af7e.nn` as
`EVALFILE`, downloads it from the networks repo during an ordinary build, and
verifies its SHA-256; the binary cannot run without it. The engine is GPL-3.0
throughout (`Copyright (C) 2024 Jay Honnold`), and the author distributes GPL
binaries built with that net. Under GPL-3.0 that makes the network part of the
Corresponding Source of the work he already distributes.

Berserk is therefore tracked and deployed again. The reasoning is written into
the distribution card rather than left implicit, confirmation has been requested
upstream, and the reversal path is documented in case the answer is no.

Two process lessons worth keeping:

- **An absent licence file is a prompt to investigate, not a conclusion.** The
  strict reading cost real product capability — the engine was pulled from the
  live site — on evidence that a five-minute look at the upstream Makefile
  overturned.
- **Do not leave policy and deployment contradicting each other.** The genuine
  defect here was never "we distribute Berserk"; it was that the repository said
  "do not distribute" while production served it. That inconsistency is what a
  community reading both would have caught, and either direction of fix resolves
  it as long as both agree.

### 2.3 Clone size — smaller problem than it first looked

**This section corrects an earlier draft of this audit.** The first pass claimed
436 MB of dead blobs in history, including two 107 MB Stockfish wasm files
(added in `8b82ee8`, removed in `5a39ddf`), and recommended a history rewrite.

That was wrong. Both of those commits live entirely on the unmerged branch
`stockfish-relaxed-full-threaded-variants` — verified with
`git merge-base --is-ancestor 8b82ee8 main` → false. A contributor cloning
`main` never downloads them.

Measured clone cost:

| Clone shape | Before rewrite | After rewrite | Saved |
| --- | ---: | ---: | ---: |
| all refs | 277.9 MB | 197.3 MB | 80.6 MB (29%) |
| `main` only | 198.8 MB | 197.3 MB | **1.5 MB (0.7%)** |

The 80 MB is obtained by simply **not publishing that branch** — no rewrite, no
hash churn. A rewrite buys 1.5 MB on a `main`-only publish while invalidating
all 1029 commit hashes.

Two further corrections to the original inventory: the
`public/ort-experimental/*.asyncify.wasm` files and the four
`public/reckless/*corresponding-source*.tar.gz` archives are **live in the
current tree**, not dead history. The `docs/` benchmark JSONs are 13.1 MB
logical but only 0.4 MB packed.

What actually dominates a published clone is the four Reckless
corresponding-source archives: **178 MB packed, roughly 90% of the pack.** They
are GPL/AGPL-load-bearing and referenced by `reckless-wasip1.manifest.json`,
`package.json`, `public/_headers`, and `NOTICE.md`. Removing them is a
licensing decision, not a cleanup. (Note the inconsistency: `.gitignore:37`
claims these are not committed, but all four are.)

The cheap wins are therefore:

1. publish `main` only and do not publish
   `stockfish-relaxed-full-threaded-variants` (−80 MB, zero risk);
2. document `GIT_LFS_SKIP_SMUDGE=1` in the README — LFS is 528 MB live at
   `main`, dwarfing the pack, and skipping the smudge drops a clone to ~197 MB;
3. `git lfs prune` reclaims 151 MB locally (mostly four superseded 37 MB
   Stormphrax source tarballs); no rewrite required.

The full inventory, the verified rewrite, and the consequences of applying it
are retained in the local-dev doc `.local-dev-docs/git_history_rewrite_plan_2026-07-25.md`
(deliberately not published with the repository).
The rewrite is proven to work — the `main` tree hash is byte-identical before
and after (`9f763b1b…`) — and is kept in reserve rather than applied.

### 2.4 A fresh clone is not a working app

`public/reckless/*.wasm` is gitignored, so Reckless is absent from a clone and
requires a Rust + wasm toolchain build. Several relaxed-SIMD artifacts are
likewise local-only. This is a defensible choice, but it must be stated in the
README so a contributor's first run does not look broken.

### 2.5 README

The README was accurate but written as internal developer notes. For a public
release it needs to state what the project is, link the live site, distinguish
what works from a fresh clone versus what needs building, point at the engine
onboarding path, and explain the licensing and clone-size situation.
`docs/engine_integration_architecture.md` already contains the substance.

## Recommended order

**Before publication**

1. LICENSE (GPL-3.0-or-later) — done.
2. Berserk removal from the repository — done.
3. README rewrite — done.
4. Berserk distribution — restored; see 2.2. Confirmation requested upstream,
   with a documented reversal path if the answer is no.
5. Clone size — publish `main` only and document `GIT_LFS_SKIP_SMUDGE=1`. The
   history rewrite is **not** recommended (see 2.3); it is verified and held in
   reserve should the Reckless source archives ever move.

**Runtime**

The actionable residuals are closed. The Stormphrax NNUE buffer now uses its
explicit SIMD alignment invariant; Play and Arena retain 64 MB CPU-engine
hash tables while Analysis receives 128 MB through the shared
`DisposableVariantPool` factories and its Stockfish adapters; and the
clone-free legality filter is covered by perft and board-restoration tests.

Items 1.2 through 1.8 are done. The threading proposal in 1.1 remains closed as
rejected on evidence; the independent alignment defect it uncovered is fixed.

**Deliberately not recommended**

- **Threading the CPU engines** (1.1) — measured, buys no plies.
- Another quantization format or manual WGSL micro-tuning sweep — the
  2026-07-14 audit's reasoning still holds.
- A representation rewrite pitched as a move-generation performance fix —
  section 1.8 records the measured gain from the narrower make/unmake change.
