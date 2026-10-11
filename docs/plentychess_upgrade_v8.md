# PlentyChess 8.0.0 upgrade audit

Branch: `perf/upgrade-plentychess-8`; audit date: 2026-10-10.

## Decision

**Upgrade to the release tag**, retaining all three tiers and `-O2` without
LTO. The promoted relaxed tier is **+6.80% browser wall NPS** and the shared
network is **15.73% smaller raw / 8.51% smaller Brotli**. Native strength is
**+2.78 ±11.32 Elo (95%)**, LOS **68.50%**, after **1,000 games**: no resolved
strength gain or regression. The fallback-tier speed costs (−6.93% base,
−3.23% SSE4.1) are explicitly accepted, not hidden. This is a browser speed /
footprint upgrade, not a claim of a statistically proven native Elo gain.

No deployment or push is authorized or performed. Original artifacts and
raw reports are retained locally; the compact tracked ledger is
[`plentychess_upgrade_v8_evidence.json`](plentychess_upgrade_v8_evidence.json).

## Versions and provenance (checklist 1)

- Old: **7.0.66**, `58d8ba2505ae2b49f48dd410d214a457d15c12c6`.
- Candidate: **8.0.0**, release tag `b-v8.0.0`,
  `50ad9f1b523063cbbbf599fe0547f42870fe2906`.
- Release published **2026-06-27**; tag commit dated **2026-06-19**.
- The live repository has **5**, not 14, commits after the tag, ending at
  `8f0922a`. They include a new network, fused accumulators, allocation and
  search fixes. Upstream still recommends the latest **release** binaries;
  no release-quality endorsement for HEAD was found. Use the tag, not HEAD.
- Re-read the engine's tagged `LICENSE` and the network repository's `LICENSE`:
  both are **GPL-3.0**. Engine license SHA-256:
  `8b1ba204bb69a0ade2bfcf65ef294a920f6bb361b317dba43c7ef29d96332b9b`;
  network license SHA-256:
  `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`.
- Sources: [release](https://github.com/Yoshie2000/PlentyChess/releases/tag/b-v8.0.0),
  [tagged license](https://github.com/Yoshie2000/PlentyChess/blob/b-v8.0.0/LICENSE),
  [upstream README](https://github.com/Yoshie2000/PlentyChess/tree/b-v8.0.0#how-to-use),
  [network license](https://github.com/Yoshie2000/PlentyNetworks/blob/main/LICENSE).

### NNUE layout and shared data

v8 adds pawn-pair inputs and changes the default network to `0178r.bin`.
The processed layout is PSQ i16 rows, pawn-pair i8 rows, threat i8 rows,
input biases, sparse/transposed L1, and f32 L1/L2/L3 biases/weights.
The helper's pawn-pair and threat arrays are contiguous; the engine reads
these as one combined array with the corresponding feature offsets.

| Asset | SHA-256 | Bytes |
|---|---|---:|
| Raw `0178r.bin` | `663e0531ac71c6c7c6d3bc4f1fac923bf69136f9786fa1da5c4ac25382aee54b` | 47,804,242 |
| Processed `processed.bin` / shared `.data` | `5cedef19778f1bb784e181fc013543127d1fdcde227795710b1a48f6e53d79cd` | 53,111,616 |

The processed preload shrinks from **63,023,936** to **53,111,616 bytes**
(**−15.73%**). Every tier must share the same canonical `.data`; a stale
canonical file must be removed in the agent's own output directory before
rebuilding *all* tiers. Do not publish mixed old/new code/network pairs.

**Preprocessing trap:** on x86, `generic` produces dense L1 weights, while
AVX2/AVX512 permute input rows. The WASM build requires the **128-bit sparse
layout**: use `arm64`, `ssse3`, or `fma`. Mac arm64 and Yukon SSSE3 helpers
produce the identical processed SHA-256 above.

## SIMD and vector gates (checklist 2–4)

`inspect_wasm_simd.mjs` on the candidate binaries:

| Build | SIMD opcodes | Relaxed integer dot | Relaxed f32 madd |
|---|---:|---:|---:|
| Base `-O2` | 2,857 | 0 | 0 |
| SSE4.1 `-O2` | 2,853 | 0 | 0 |
| Relaxed `-O2` | 2,767 | 12 | 28 |
| Relaxed `-O2 -flto` | 4,055 | 12 | 28 |
| Relaxed `-O3` | 3,001 | 12 | 28 |
| Relaxed `-O3 -flto` | 6,569 | 12 | 28 |

- `ARCH_X86`, `__SSSE3__` engage vector accumulator updates, activation,
  NNZ extraction, and sparse integer L1 on every tier. SSE4.1 engages the
  single-op signed weight-extension path.
- With the supplied flags, emcc defines `__wasm_simd128__` and `__SSSE3__`;
  SSE4.1/relaxed builds also define `__SSE4_1__`, and the relaxed flag adds
  `__wasm_relaxed_simd__`. It does **not** define FMA/AVX2/ARM.
- All **three f32 gates** in v8 still need our relaxed extension: L1
  conversion/activation, L2 propagation/activation, L3 propagation. The
  preprocessed relaxed evaluator contains all three vector branches and
  **no scalar `std::fma`**. Actual relaxed-madd opcodes corroborate this.
- Base/SSE4.1 deliberately retain the scalar f32 tail to preserve fused
  `std::fma` semantics without relaxed madd. Nonzero aggregate SIMD alone
  would not have caught the silent f32-scalar trap.
- Activation proof rechecked in v8: `INPUT_QUANT=255`, `INPUT_SHIFT=9`;
  the nonnegative first factor is at most 255, the second is at most 255;
  shifted signed multiply-high plus unsigned saturation yields **[0,127]**.
  Negative second factors saturate to zero. The activation is passed as
  the **second (i7)** relaxed-dot operand. Signed weights are [-128,127],
  so pair sums lie in **[-32512,32258]** and cannot saturate.
- Compile-time assertions guard the shift, signed-i16 preshift bound and
  maximum i7 activation. The preload loader checks `sizeof(NetworkData)`.
- Relaxed f32 madd permits unfused implementations in the WASM spec;
  exactness is an empirical gate on the tested FMA-capable machines, not
  a universal proof for pre-FMA hardware. This caveat already applied to
  the promoted 7.0.66 tier and is not weakened here.

## Compiler flags (checklist 5)

Rebuilt the full `-O2/-O3 × LTO off/on` matrix with **Emscripten 5.0.7-git**,
keeping the promoted relaxed tier, source, patch, network, memory, Hash=16,
and Threads=1 fixed. **Keep `-O2`, no LTO, on every tier.** The quiet browser
matrix's best paired median gain is **+2.45%** (`-O3 -flto`), below the 5% bar;
`-O2 -flto` is **+1.34%**. Never infer vector engagement from flags alone.

Final counterbalanced browser run: **8 rounds × 20 positions**, depth 11.
Over eight rounds all four flag cells occupy each slot twice.

| v8 relaxed flags | Wall NPS, thousands [min,max] | Paired gain % [min,max] | WASM raw / Brotli-11 bytes |
|---|---:|---:|---:|
| `-O2` | 1103.0 [932.5,1131.3] | reference | 392393 / 122860 |
| `-O2 -flto` | 1114.2 [983.9,1137.3] | +1.34 [+0.37,+12.09] | 439426 / 124373 |
| `-O3` | 1100.5 [1035.6,1127.5] | +0.33 [-3.30,+13.58] | 398357 / 121745 |
| `-O3 -flto` | 1131.6 [1068.2,1150.7] | +2.45 [-2.26,+15.95] | 461902 / 122896 |

Counterbalanced Node paired medians: **+3.47%, +1.32%, +4.14%** respectively,
with all scheduling outliers retained. No candidate has a repeatable ≥5%
browser win. The preceding quiet 10-round browser run was also below 5%
(+2.21% / +0.54% / +1.84%). Earlier CPU-loaded and depth-12 batches remain
separately recorded; they are too noisy for flag promotion. A subsequent
quiet depth-12 attempt lost its browser session before capture and is not
counted.

**Order trap fixed:** the inherited rotate-every-round + reverse-on-odd
algorithm cancels reversal in two-cell A/B tests, and traps even-N cells in
slot parity. The harness now rotates once per forward/reverse pair; tests
cover N=2/4/5. The final eight-round Node/browser runs use this correction.
Earlier reports are retained separately rather than relabeled counterbalanced.

## Exact parity (checklist 6)

- Mac Node, depths **9/11**, rotated 20-FEN suite: six candidate builds,
  **40/40 each**; five comparisons against relaxed `-O2` = **200/200**.
- Yukon native **unpatched upstream same commit** (GCC 13.3, AVX2), Hash=16,
  Threads=1, same 40 requests: **240/240** exact comparisons against all
  six WASM builds (40 native searches reused across candidates).
- Final counterbalanced Node/browser suite: **1,600/1,600** exact matched
  artifact/round/position rows; flag candidates compare **480/480** against
  v8 `-O2` in each runtime. Earlier valid 10-round batches additionally passed
  **2,000/2,000** cross-runtime comparisons each.
- Canonically named release artifacts retain **40/40 per tier**; rebuilding
  all three from the unpacked archive inside this Git worktree reproduces
  **all seven JS/WASM/data files byte-for-byte**.
- Post-match Yukon Node, same depths/suite: **40/40 per shipped tier**, and
  **120/120** comparisons against the Mac rows; zero mismatches.
- Equality includes **depth, bestmove, score, nodes and full PV**. Missing
  fields fail closed. Old/new versions are *not* expected to match each other.
- Reusable native comparator: `scripts/native_uci_parity.mjs`; reference
  files are produced by `scripts/emscripten_uci_bench.mjs`.

## Native strength (checklist 7)

**1,000 games: v8 W262 / L254 / D484**, **+2.78 ±11.32 Elo (95%)**, LOS
**68.50%**; confidence interval **[−8.54,+14.10]**. Pentanomial (500 pairs):
**[3,119,251,121,6]**. This meets the requested ~±15 Elo precision but does
**not** establish a strength gain. No crash, illegal move or time-forfeit
termination occurred; all 1,000 PGNs end normally or by adjudication.
fastchess logged last-PV/bestmove mismatch warnings on time-limited searches
in **both** upstream versions (old **641**, v8 **534**); fixed-depth full-PV
parity is exact and these warnings did not invalidate games.

Yukon rules were read first. All match processes inherit
`taskset -c 3-5`; builds use `nice -n 10 taskset -c 15-17`.
Both versions use GCC 13.3 native `arch=avx2`, upstream `-O3`/LTO, no PGO.
Ryzen 3900XT is Zen2: avoid the BMI2 target as upstream recommends.

- fastchess `alpha 1.8.2 20261004-a281caf`.
- **TC 8+0.08, Threads=1, Hash=16, concurrency=3**, paired/reversed openings.
- Book: `UHO_Lichess_4852_v1.epd`, randomized seed **8000**; SHA-256
  `7a7f6470615a69c6cf23d565417701d38732876f480af90d67b42abade35644a`.
- Resign: 3 moves at 600 cp; draw: after move 34, 8 moves at 20 cp.
- Fixed **1,000-game observation budget**, chosen before a resolved result;
  no stopping on favorable Elo. Initially launched with a 2,000-game ceiling.
- Native parity briefly shared core 3 during the match; no native timing
  claims come from that run. x86 WASM speed runs completed *after* the match,
  pinned to core 3, with no overlapping match process.

## Speed, footprint and startup (checklist 8)

Final Mac browser results (depth 11, 20 positions, **8 counterbalanced rounds**):

| Tier | Old wall NPS, thousands [min,max] | v8 wall NPS, thousands [min,max] | Paired change % [min,max] | Old → v8 startup median ms |
|---|---:|---:|---:|---:|
| Base SIMD | 639.1 [634.3,654.1] | 597.5 [584.5,603.0] | −6.93 [−7.97,−5.55] | 139.7 → 133.6 |
| SSE4.1 | 691.4 [673.2,696.5] | 670.8 [656.9,677.7] | −3.23 [−3.64,−0.24] | 137.5 → 132.8 |
| Relaxed (promoted) | 1047.2 [1006.0,1063.4] | 1113.4 [1107.4,1130.7] | **+6.80** [+4.33,+10.08] | 138.4 → 132.6 |

**Fallback speed regressions are real** and must not be hidden by the promoted
result. Counterbalanced Mac Node paired medians: base −5.52%, SSE4.1 −2.86%,
relaxed +4.38%. Relaxed startup old range **134.7–143.6 ms**, v8 **129.9–136.6
ms**; this is not a statistically established cold-start win. The prior quiet
10-round browser run yielded relaxed +7.13%; the CPU-loaded run +8.63% with
much wider absolute spreads. Keep these separately labeled; no row was dropped
from any retained batch.

Post-match Yukon Node v26.2.0 / Ryzen 3900XT, core 3, **6 counterbalanced
rounds**, same depth/suite/options (second datapoint, not browser timing):

| Tier | Old wall NPS, thousands [min,max] | v8 wall NPS, thousands [min,max] | Paired change % [min,max] | Old → v8 startup median ms |
|---|---:|---:|---:|---:|
| Base SIMD | 311.4 [310.2,312.6] | 283.6 [281.3,285.0] | −8.79 [−9.84,−8.65] | 294.6 → 280.8 |
| SSE4.1 | 379.4 [371.0,383.5] | 372.3 [368.9,374.3] | −1.76 [−3.80,+0.89] | 294.5 → 280.0 |
| Relaxed | 543.8 [536.7,544.7] | 572.5 [565.3,577.8] | **+5.89** [+3.95,+6.59] | 294.1 → 279.0 |

| Release file | Old raw / Brotli-11 bytes | v8 raw / Brotli-11 bytes |
|---|---:|---:|
| Base WASM | 389983 / 122064 | 395075 / 122865 |
| SSE4.1 WASM | 388663 / 122026 | 394229 / 123465 |
| Relaxed WASM | 386827 / 121491 | 392393 / 122860 |
| Shared NNUE `.data` | 63023936 / 32447720 | 53111616 / 29685001 |

The shared NNUE is **−15.73% raw / −8.51% Brotli**; relaxed WASM adds only
5,566 raw / 1,369 Brotli bytes. JS is ~71 KB raw / ~17.8 KB Brotli per tier.

Mac timing is wrapped in the supplied `benchlock`; browser timing holds the
same mutex via a sentinel for the entire matrix. Config order rotates once
per forward/reverse pair, with fresh subprocesses/Workers and one unreported
warmup per cell.
Report medians **[min,max]**, and paired same-round gains. Wall NPS is total
final nodes divided by total search wall time, excluding setup/reset/startup.

Browser runs use the built product's classic Worker/importScripts/ccall path,
COOP/COEP isolation, no-store fetch, and a unique ignored WASM custom-section
nonce to avoid same-byte module cache reuse. Startup ends at module readiness:
it includes localhost fetch, compile, instantiation, preload and engine init,
not cold-CDN transfer or search warmup.

**Invalid run:** an awaited >30s browser eval hit an upstream IPC timeout and
was retried by the browser machinery, creating overlapping runs. Its captures
are explicitly invalid and are not used. Subsequent runs launch asynchronously
with an already-running guard and poll short commands. A valid but CPU-loaded
batch is retained separately, not silently mixed with the quiet batch.

## Validation, review and release

- Typecheck: **0 errors, 0 warnings**.
- Focused tests: **22 passed**.
- Final full `npm test`: **808 passed, 3 skipped, 0 failures** (811 total).
- Product build: passed. `git diff --check`: passed after patch formatting.
  Final build/typecheck/tests required worktree-local quarantine of six dangling
  legacy Stockfish aliases after their shared node_modules targets disappeared.
  All six symlinks were restored byte-for-byte; no Stockfish change is committed.
  Vite used `--configLoader runner` to avoid shared node_modules config caches.
- Browser promoted-tier lifecycle smoke passed: newgame, repeated search,
  MultiPV=2, abort/recovery, and intentional missing-asset failure. Node ABI
  lifecycle smoke passed on all three tiers.
- A standalone archive rebuild produced all seven release files byte-identically.
  The recipe rebuilds every tier from a fresh copy, requires no npm dependencies,
  and works when unpacked *inside* a Git worktree (see the ceiling fix).
- Native/x86 results are imported and summarized in the tracked ledger. Source
  archive and manifest are rebuilt as a matched set; all release files use the
  existing LFS/tracked mechanism. The final Codex closeout is recorded below.
- Never publish/deploy/push from this task. No large experiment binaries belong
  outside the existing LFS/ignored paths.

### Commits and review closeout

- `e89b363`: fix even-sized benchmark counterbalancing, with N=2/4/5 tests.
- The focused upgrade commit contains the pin/network/patch, metadata, all
  release artifacts, source bundle, native comparator, tests and evidence.
- Required Codex review runs on the committed branch against `main`, using
  `gpt-6.1-sol` / `xhigh`; final status and the reviewed commit are recorded
  after the helper completes. No push is needed or allowed.

```sh
~/.pi/agent/skills/autoreview/scripts/autoreview --mode branch --base main \
  --engine codex --model gpt-6.1-sol --thinking xhigh \
  --prompt-file "$SCRATCH/plentychess-8-review-notes.md" \
  --output "$SCRATCH/perf-upgrade-plentychess-8-review-1.md"
```

**Release/integration needs:** the seven JS/WASM/data files, engine manifest
and GPL corresponding source have been rebuilt here. After integration,
rebuild the product site, global content-addressed release manifests and
compression sidecars; publish the matched artifacts/source/notices and update
the artifact-hosted app shell as one coherent release. Revalidate deployed
hashes, headers, source links and cold-start/cache behavior. Do not publish
only the changed network or stale aliases. No publication is performed here.

## Process retrospective

Manual/fragile steps and concrete automation proposals:

1. **Pin/license/HEAD verification:** release API, tag ancestry, live README,
   and both licenses required manual checks. Add `engine:upgrade-audit
   plentychess` to fetch immutable refs, classify release/HEAD, hash/read the
   licenses, and emit a provenance ledger; do not trust task-provided counts.
2. **Metadata duplication:** lock, builder, archive, manifest, runtime and docs
   repeat pins/hashes. Make generators consume the lock and generate the
   public metadata; add consistency tests until that migration is complete.
3. **Patch rebasing/shared histories:** new Worker construction and shared
   allocation/free paths needed manual inspection. Add an automated native vs
   synchronous-browser lifecycle test including resize, newgame and shutdown.
4. **SIMD layers:** aggregate opcode counts miss a scalar f32 tail. Emit macro
   and preprocessed per-layer evidence and require both dot and madd opcodes;
   retain compile-time activation assertions and same-commit native parity.
5. **Processed NNUE layout:** x86 generic was a latent wrong-layout default.
   Enforce helper architectures, size, hash, and all-tier shared-data identity
   automatically; include cross-architecture helper checksum checks.
6. **Build matrix/browser capture:** source resets, fresh instances, mutex
   acquisition/release, async polling and compacted-output recovery were
   manual and slow. One runner should own all these, reject overlapping runs,
   save reports directly, and log background compiler/test load. The even-sized
   order trap was fixed and the final runs repeated with balanced slot coverage.
   Extend coordination to CPU-heavy builds/tests, not just NPS runs; isolate
   Vite caches when node_modules is shared between worktrees.
7. **Yukon match:** toolchain discovery, slots, book, stopping policy, PGN and
   stats transfer were manual. Script fixed-game/CI-budget matches with slot
   validation, resume support, signed provenance and result import. Never
   push from Yukon. Queue speed benchmarks after Elo, not concurrently.
8. **Source/release:** the archive recipe previously rebuilt only the base
   tier; `git apply` inside an enclosing repo silently skipped every patch,
   and `npm ci` unnecessarily invoked app preparation in a partial source tree.
   Fixes: three fresh tier copies, a Git discovery ceiling, no npm dependency
   step for this built-in-only builder, and a regression test. Require a
   standalone three-tier rebuild, manifest hashes, shared data and smoke as
   one packaging gate.
9. **Optimization gaps:** measure WASM PGO (upstream already recommends native
   PGO), NNUE update tiling for 128-bit WASM registers, and lower initial heap /
   stack sizes. Investigate an exact portable vector-FMA fallback so non-relaxed
   tiers need not leave the f32 tail scalar. Treat these as measured experiments,
   not free speedups; preserve full search parity and cold-start/memory evidence.

## Evidence and reproduction

Ignored local evidence: `.local-dev-artifacts/plentychess-upgrade/` (original
artifacts, six build logs, configs, SIMD/macro/preprocessed dumps, full search
rows, timing summaries, test logs, release API response). Yukon evidence:
`~/upgrades/plentychess/` (native sources/binaries, match script/log/PGN/state,
book/toolchain identity, native parity, completed x86 WASM reports).

```sh
BENCHLOCK=/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad/benchlock
"$BENCHLOCK" node scripts/emscripten_uci_bench.mjs --js <six-v8-js-paths> --depths 9,11 --hash 16 --json wasm-parity.json
# Yukon, from ~/upgrades/plentychess (same-commit native engine):
taskset -c 3 node scripts/native_uci_parity.mjs --engine upstream/new/engine --reference wasm-parity.json --hash 16 --json native-parity.json
"$BENCHLOCK" node scripts/emscripten_uci_matrix.mjs --config <matrix.json> --rounds 8 --depth 11 --hash 16 --json node-matrix.json
```
