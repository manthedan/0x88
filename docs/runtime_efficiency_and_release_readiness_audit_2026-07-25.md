# Runtime efficiency and release-readiness audit

Date: 2026-07-25
Scope: browser runtime throughput, asset delivery, and the work required before
publishing this repository to the chess-engine community.

## Executive conclusion

The project is in good shape. The test suite is green (611 pass, 0 fail, 1
skipped), production is live and correctly configured (cross-origin isolation
and Brotli both verified against `https://0x88.app`), and the artifact,
licensing, and provenance *tooling* is more rigorous than most projects of this
size.

The 2026-07-14 audit (`inference_caching_optimization_gap_audit_2026-07-14.md`)
already covers the neural-inference and delivery-layer opportunities well, and
its ranking still stands. This audit deliberately does not repeat it. It covers
what that audit did not: **CPU parallelism, runtime artifact selection, and the
concrete blockers to a public release.**

The single largest untapped runtime win is not another precision format or
kernel schedule. It is that the project currently uses roughly one core out of
however many the user has.

## Part 1 — Runtime efficiency

### 1.1 Every CPU engine except Stockfish is single-threaded

**Severity: high. Effort: medium. This is the largest remaining win.**

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

Caveats to size honestly before committing:

- `ALLOW_MEMORY_GROWTH=1` combined with pthreads has real Emscripten caveats
  (growable `SharedArrayBuffer`); the current builds use growth plus a 2 GB
  maximum, so the interaction needs checking rather than assuming.
- Lazy SMP scaling in WASM is not the same as native. The measured number
  matters more than the theoretical one. See
  `threaded_emscripten_smp_prototype_2026-07-25.md` for the prototype result.
- Multiple engines can be live simultaneously in Arena and Analysis. Threading
  makes the broker's job real rather than nominal, which is a feature, but the
  default per-engine grant needs to account for it.

### 1.2 ORT WASM is pinned to one thread in production workers

**Severity: high. Effort: low — this is likely just a stale workaround.**

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
browser without WebGPU** — Safari and Firefox users are currently getting
single-threaded neural inference on machines with 8+ cores.

Action: rebuild, boot a production worker with `?ortThreads=auto`, confirm it
does not deadlock, and flip the default if green. Tracked in
`retest ORT threaded wasm boot`.

### 1.3 The CPU-only path uses the asyncify ORT build

**Severity: medium. Effort: low-medium.**

`browserOrtWasmPaths()` (`src/nn/ortRuntime.ts:336-354`) pins
`ort-wasm-simd-threaded.asyncify.wasm` for every path, and
`ORT_PTHREAD_WASM_FILE` (line 18) hardcodes the asyncify binary.

Sizes:

| Artifact | Size |
| --- | ---: |
| `ort-wasm-simd-threaded.asyncify.wasm` | 24.2 MB |
| `ort-wasm-simd-threaded.wasm` | 13.5 MB |

Asyncify instruments the entire module for stack unwinding. It is genuinely
required for the JSEP/WebGPU path, which suspends across GPU operations. For an
`executionProviders: ['wasm']` session it is pure overhead — in download size
*and* in per-call execution cost.

This compounds with 1.2: the WASM path is the fallback for browsers without
WebGPU, and it is currently paying both the single-thread penalty and the
asyncify penalty at the same time.

Hazard to respect when fixing: ORT's WASM artifact is worker-global once
initialized — `ortRuntime.ts` already throws *"ORT WASM is already initialized
with a different artifact; select the fallback in a fresh worker"*. A worker
that starts on WebGPU and falls back to WASM must decide its artifact before
first init, not after.

### 1.4 Transposition tables are left at 16 MB

**Severity: medium. Effort: trivial.**

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

**Severity: medium (bandwidth). Effort: low.**

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

### 1.6 Cloudflare caches nothing on the app origin

**Severity: medium. Effort: trivial (dashboard config, no code).**

Measured against production:

| URL | `cf-cache-status` |
| --- | --- |
| `https://0x88.app/` | `DYNAMIC` |
| `https://0x88.app/ort/ort-wasm-simd-threaded.asyncify.wasm` (24 MB) | `DYNAMIC` |
| `https://0x88.app/models/lc0/manifest.json` | `DYNAMIC` |

`DYNAMIC` means Cloudflare is not caching the response at all — Cloudflare's
default cache is extension-driven and does not include `.wasm`. Every cold
request for the 24 MB ORT binary traverses Cloudflare to the Netlify origin.

The Netlify edge itself *does* warm correctly (`cache-status: "Netlify Edge"; fwd=miss`
on the first request, `; hit` on the second), and `netlify.toml` sets
`Netlify-CDN-Cache-Control` / `CDN-Cache-Control` / `Cloudflare-CDN-Cache-Control`
correctly. The outer Cloudflare layer is simply not acting on them, so it is
currently pure added latency.

Action: a Cloudflare Cache Rule (or "Cache Everything" scoped by path) for
`/ort/*`, `/models/*`, `/engines/*`, and `/artifacts/sha256/*`. No code change.

### 1.7 R2 artifacts are stored identity-only; Brotli is recomputed per miss

**Severity: medium. Already rank 1 in the 2026-07-14 audit; still unshipped.**

Measured against `https://assets.0x88.app/reckless/reckless-relaxed-simd128.wasm`:

```
x-artifact-content-length: 64578859
x-artifact-encoded-length: 64578859
x-artifact-decoded-sha256: a9b4b292…34fcb
x-artifact-encoded-sha256: a9b4b292…34fcb   <-- identical to decoded
content-encoding: br                         <-- applied by the edge, not stored
```

The stored object is identity; Cloudflare compresses on the fly. It works, and
it does save bandwidth (43.3 MB transferred vs 64.6 MB identity, a 33%
reduction), but the cost shows up in latency: on a **warm** cache hit the Brotli
response took 2.30 s against 1.10 s for identity — repeated on-the-fly
compression at a low quality level.

A stored, precompressed `br` representation would be both smaller (higher
quality level, computed once) and faster to serve. This is precisely the
"representation-aware, SHA-only artifact delivery" work ranked #1 in the
2026-07-14 audit, and the `deploy:r2-brotli-assets` tooling already exists — it
is the deployed path that has not adopted it.

### 1.8 The move generator is not a 0x88 move generator

**Severity: low for throughput. High for credibility.**

`src/chess/movegen.ts` backs a project named **0x88** with:

- a 64-entry array of two-character JavaScript strings (`'wk'`, `'bp'`) as the
  board representation;
- `makeMove` (line 147) cloning the entire board object per move;
- `legalMoves` (line 138) implemented as
  `pseudoLegalMoves().filter(m => !inCheck(makeMove(board, m)))` — a full board
  clone, a `findIndex` king scan over 64 squares, and a full attack scan **per
  pseudo-legal move**, with no pin detection and no make/unmake;
- template-literal string allocation inside the inner loops of
  `isSquareAttacked` (lines 98, 102, 122 — `` `${by}n` ``, `` `${by}k` ``), i.e.
  a string allocation per square probed;
- `fileOf`/`rankOf` using `%` and `Math.floor` rather than `& 7` and `>> 3`.

Measured with the project's own harness (`npm run bench:movegen`):
**130,464 legal move generations/second** from the start position.

**Be honest about the impact.** This is *not* currently a bottleneck. The neural
lane is GPU-bound at roughly 14 ms per batch-8 invoke (~1.75 ms/position)
against roughly 8 µs for a `legalMoves` call — movegen is well under 1% of
search time. A rewrite should not be sold as a throughput win.

It matters for two other reasons:

1. It is the first file a chess programmer will open, given the repository name.
   A project called 0x88 whose board is a string array will be the first thing
   the community comments on.
2. It sets the floor for the low-visit and fast-backend cases that the
   2026-07-14 audit's item 3 ("search-native inference context") targets. If
   that work lands and CPU preparation becomes a larger fraction of latency,
   this becomes real.

Recommendation: either rename the project, or make the representation match the
name. Do not do it for the FLOPs.

## Part 2 — Release readiness

### 2.1 No LICENSE — resolved in this pass

The repository had no licence file at all, which meant default
all-rights-reserved: no community member could legally fork or contribute, while
the project simultaneously distributes GPL-3 engine binaries.

Resolved: the project is licensed **GPL-3.0-or-later**, matching every bundled
engine (Stockfish, Berserk, PlentyChess, Stormphrax, Viridithas) and removing
any argument about whether the app plus its engines constitute a combined work.

### 2.2 Berserk network provenance — resolved in this pass

`docs/engine_artifact_distribution.md` states plainly that the Berserk network's
licence and provenance are unresolved, and that the network/data bundle must not
be publicly distributed until that is settled. The production deploy already
prunes Berserk (`scripts/prune_v0_deploy_assets.mjs:31`), so the live site was
clean — but the 24 MB `.data` was committed via Git LFS, and publishing the
repository is itself distribution.

Resolved: the generated Berserk artifacts are removed from the repository and
fetched from upstream at build time. Berserk remains fully buildable; it is no
longer redistributed by this project.

### 2.3 Git history carries 436 MB of dead blobs

`.git/objects` is 278 MB before LFS, with a further 750 MB in `.git/lfs`.
Blobs over 20 MB in history total 436 MB, including two **107 MB** Stockfish
wasm files added in `8b82ee8` and removed in `5a39ddf` — permanently present in
every clone despite being absent from the current tree.

See `git_history_rewrite_plan_2026-07-25.md` for the inventory, the rewrite, and
the consequences (every commit hash changes; existing clones must be re-cloned).

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
2. Berserk artifact removal — done.
3. README rewrite — done.
4. Git history rewrite — planned, requires sign-off (hash-changing, irreversible).

**Runtime, by payoff over effort**

1. Threaded Emscripten builds for Berserk / PlentyChess / Stormphrax (1.1).
2. Retest and likely remove the ORT built-worker thread guard (1.2).
3. Non-asyncify ORT binary for wasm-EP sessions (1.3).
4. Shared `.data` across SIMD variants (1.5).
5. Raise hash defaults and reconcile the clamps (1.4).
6. Cloudflare cache rule (1.6) — config only.
7. Stored Brotli representations on R2 (1.7) — already ranked #1 in the
   2026-07-14 audit.

**Deliberately not recommended now**

Another quantization format, another manual WGSL micro-tuning sweep, or a
movegen rewrite pitched as a performance fix. The 2026-07-14 audit's reasoning
on the first two still holds, and section 1.8 explains the third.
