# Reckless upgrade audit — 2026-10-10

**Verdict: NO-GO. Production remains pinned to `0010617448bdef4c8cd7d4f4825b7e42c8bc262a`.** The candidate passed the rotated suite but failed the broader native-bench identity gate in both SIMD tiers. There is no demonstrated strength gain to justify that compatibility failure or the scalar throughput loss. Only audit evidence is committed; all attempted pin/runtime/build/manifest/source-archive changes were rolled back. No push, deploy or publication.

Branch: `perf/upgrade-reckless`. Machine-readable summary: [`fixtures/reckless_upgrade_2026-10-10.json`](fixtures/reckless_upgrade_2026-10-10.json).

## 1. Candidate, version, license and corresponding source

- Old: `0010617448bdef4c8cd7d4f4825b7e42c8bc262a`, 2026-06-04.
- Candidate: **`31d9cd6fd2bea6d9f72eeb35e0bac70daa295fb1`**, upstream main, 2026-09-09; **43 commits newer**.
- Both Cargo packages are `0.10.0-dev`; `v60` names the network, not an engine release. Trial artifact identity was `0.10.0-dev-31d9cd6f`; the misleading existing lock terminology is not silently changed by this evidence-only outcome.
- Newest commit has accepted STC/LTC SPRTs and bench **2801205**: BCH Zobrist construction. STC (8+0.08, Threads=1, Hash=2): +2.66 ±1.97 Elo, 33,600 games, LLR 2.94 > 2.89. LTC (40+0.40, Hash=8): +1.96 ±1.54, 46,368 games, LLR 2.90 > 2.89. These are **patch-vs-parent**, not our old-pin comparison.
- Dev tags `v0.10.0-dev-7300f044` (May 1) and `v0.10.0-dev-6abcff05` (April 1) are ancestors of the old pin, not newer candidates.
- Sources: [commit](https://github.com/codedeliveryservice/Reckless/commit/31d9cd6fd2bea6d9f72eeb35e0bac70daa295fb1), [STC](https://recklesschess.space/test/15692/), [LTC](https://recklesschess.space/test/15691/). Read commit messages/tags from a fresh upstream clone.

Upstream LICENSE was re-read: GNU Affero GPL, version 3, unchanged byte-for-byte between pins; SHA-256 `8486a10c4393cee1c25392769ddd3b2d6c242d6ec7928e1414efff7dfb2f07ef`. No newly identified license change blocks the upgrade.

Network **`v60-7f587dfb.nnue`**, 63,266,880 bytes, unchanged SHA-256 `7f587dfb1fe5d74d53909328afa6fd51650c8c7f45907602db7fbb1e52948c61`; L1/L2/L3 **768/16/32**, buckets and threat inputs unchanged. This was verified, not inferred from the filename.

Trial release build generated matching manifests and all **seven** corresponding-source archives: four WASI plus dedicated API scalar/SIMD/external trees. The trial also made the lock the shared default-ref source and stamped version identity from the fetched upstream ref, avoiding copied/extracted sources identifying the surrounding app Git repo. Extracted API external SIMD source rebuilt **offline, locked, byte-identically** to its 1,240,948-byte WASM. Provenance success does not establish inference correctness.

Those production changes and archive updates are **not landed**. Trial archives/manifest/WASM and `attempted-upgrade.patch` remain in ignored local evidence. Future publication needs dedicated API corresponding-source coverage, not merely the WASI adapter's archive.

## 2. Actual SIMD inspection — pass in trial builds

| Candidate variant | SIMD opcode count |
| --- | ---: |
| Scalar WASI | 0 (intentional fallback) |
| Fixed SIMD WASI | 1833 |
| Relaxed SIMD WASI | 1782 |
| External NNUE SIMD WASI | 1833 |
| Scalar browser API | 0 (intentional fallback) |
| SIMD browser API | 1621 |
| External NNUE SIMD browser API | 1621 |

All seven imported only `wasi_snapshot_preview1`, not JS glue. Zero in a SIMD tier would fail; scalar fallbacks must remain scalar.

The relaxed binary has **12 integer relaxed-dot sites** and one compiler-generated `f32x4.relaxed_max` outside NNUE, in the time-management multiplier inlined into `ThreadPool::execute_searches` (offset 0x249b5). Its active lanes clamp `(1.2881 - .0440*pv_stability)` to .7160 and `(3.1838 - 2.6554*fraction)` to .5460. Integer-converted inputs are finite; the searched-node denominator is positive and clamp bounds are positive, avoiding NaN/signed-zero ambiguity. The production-style trial NNUE shim had no relaxed f32 madd/min/max calls. Per-function inspection matters; global totals alone do not prove this.

## 3. NNUE path gates and layout — structurally pass, not a semantic proof

Upstream now explicitly selects `forward::vectorized` and `simd::wasm` on wasm32+simd128. Accumulator add/sub, `activate_ft`, sparse `propagate_l1`, f32 `propagate_l2` and `propagate_l3` call the selected SIMD implementation, not just auto-vectorized scalar code. Trial adaptation reused upstream SIMD `find_nnz`, changed `convert_i8_i16` to its new `u64` interface, and matched recursive-halving reduction order **16→8→4→2→1**. The AVX-512 VNNI split accumulators remain gated out on WASM, which keeps `double_dpbusd`; native gates are Rust target features, not Emscripten C macros.

**Known scalar threat-generation path is unchanged:** `src/nnue/accumulator/threats.rs` selects `threats/scalar.rs` without AVX2. The entire accumulator/threat subtree is byte-identical between pins. Threat generation remains scalar on WASM; applying its deltas remains SIMD.

The upstream JS/WASM runtime needed careful separation: trial WASI patches excluded JS callbacks/web-time/wasm-bindgen modules, retained std clocks/stdout, synchronous single-thread search and persistent UCI stdin, and clamped Threads to one. All seven variants compiled, ran and reset correctly on the tested corpus. The source-shape/quantization tripwires added in the trial were fail-closed structural guards, not proofs of floating-point equivalence.

## 4. Relaxed integer activation proof — pass; float FMA is a different problem

FT lhs is clipped to [0,255]; rhs upper-clamped to 255; signed high-multiply implements the product shift and unsigned packing clamps negatives to zero. Thus activation bytes lie in **[0, `255² >> 9`] = [0,127]**. Only activations occupy the i7 argument of `i32x4_relaxed_dot_i8x16_i7x16_add(i8s,u8s,i32s)`.

Weights can be -128: worst negative i16 pair sum **-32512**, positive maximum **32258**, neither saturating. Standard dot retains shuffles instead of scalar lane extraction. Integer dot remains exact; it does **not** prove the unfused float tail matches native fused inference.

## 5. Rust release profile — pass

Mac and Yukon used Rust **1.95.0**, release **opt-level=3, fat LTO, codegen-units=1, panic=abort**; compiler invocation confirmed the optimization/LTO/codegen values. C/C++ O2/O3×LTO matrix is not applicable. Upstream CI specifies 1.97.0; 1.95.0 nevertheless compiled both revisions and reproduced their native bench signatures (**3634130 old**, **2801205 candidate**).

Yukon native builds: `taskset -c 21-23 nice -n 10 env RUSTFLAGS='-C target-cpu=znver2' cargo build --release --no-default-features -j3`. Match/bench processes ran on cores **9–11**. Live `/proc/.../task/.../status` checks confirmed all six engines and their threads retained affinity 9–11.

## 6. Native/WASM parity — narrow suite passes; broader bench FAILS

Twenty rotated FENs, Threads=1, Hash=16; compare same-commit native **bestmove, normalized cp/mate score, nodes and full PV**. Native and WASM versions were never compared for equality across different refs.

- Mac Node WASI/API: **700/700 exact** sampled comparisons, including depths 7/8/9, dedicated external-NNUE checks, and all seven release artifacts at depths 11/12. Count includes one repeated SIMD reference run.
- Real isolated Chromium, persistent WASI or direct API, reset before each search: **seven × 120 = 840/840 exact**, two rotated passes at depths 7/8/9. Covers warm-state history reset, including upstream's new pawn history.
- Yukon x86 Node: production scalar/SIMD/relaxed **three × 40 = 120/120 exact** at depths 11/12.

**These 1660 passing comparisons were insufficient.** The broader upstream built-in bench under Mac benchlock returned:

| Candidate implementation | Bench nodes |
| --- | ---: |
| Native upstream | **2801205** |
| Scalar WASI | **2801205** |
| Fixed SIMD WASI | **2646409 — mismatch** |
| Relaxed integer-dot WASI, unfused float tail | **2646409 — mismatch** |
| Diagnostic relaxed-float-madd probe | **2801205** |

Replacing only float `mul_add_f32` with `f32x4_relaxed_madd` in an ignored diagnostic build restored the native signature on this Mac. This isolates a fused/unfused float incompatibility, not an i7-range failure or a silent scalar path. **Not a portable fix:** relaxed float madd permits fused or unfused results; one host's match is not an all-platform guarantee. The probe was not rebuilt/revalidated across the full seven-variant matrix and is not shipped. Standard SIMD still needs portable exact FMA or another verified solution.

The failure blocks promotion regardless of the narrower suite and clean code review. A subsequent baseline audit also found an **inherited gap**: old WASI scalar bench **3652376**, old SIMD/relaxed **3744067**, versus old upstream/Yukon native **3634130**. This is not exclusively a new-commit defect. Old native Mac was not sampled here; do not claim the baseline has universally exact native parity. The existing shim's broader floating-point compatibility needs a separate fix/audit, not a waiver for promoting this candidate.

## 7. Strength — precision gate reached, no demonstrated gain

Fastchess alpha 1.8.2 (`20261004-a281caf`), TC **8+0.08**, Threads=1, Hash=16, concurrency=3, UHO_Lichess_4852_v1.epd, seed **31092026**, paired/color-reversed openings. Resign 3 moves/600cp; draw after move 34, 8 moves/20cp. Candidate vs old native, identical toolchain/profile and core slot.

Stopped via graceful SIGINT after reported **600 paired rated games** reached **+1.16 ±13.63 Elo (95%)**, LOS **56.62%**: **149 wins / 147 losses / 304 draws**, pentanomial **[3,61,168,67,1]**. This is statistically consistent with no improvement or a modest loss/gain, not evidence of stronger play. PGN has 601 records: 300 complete paired rounds plus one unpaired game excluded from the 600-game paired rating. Configuration supports resuming if more precision is desired.

Candidate had **two real time forfeits**, games 9 and 134 (119ms and 54ms overruns); **zero crashes**. Both forfeits remain included; no favorable filtering or speculative attribution. Total match time reported 1:24:33. Logs/PGN/config retained under `~/upgrades/reckless/` and copied into local evidence.

## 8. Browser NPS, second platform, size and startup

Mac **Apple M4 / arm64, Headless Chromium 149.0.0.0**, isolated Vite, **depth 11, 20 FENs × 3 rotated repeats**, persistent workers, Threads=1, Hash=16, reset outside timed search, one untimed warm-up per artifact. Old/new interleaved per FEN, order reversed on alternating repeats. Shared `benchlock` held through a bounded browser lease; explicit `reckless-upgrade` browser session on every action, no one-shot fallback.

| Tier | Old median NPS | New median NPS | Change | Old repeat-median range | New repeat-median range | Old→new median search wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar fallback | 186,049 | 163,177 | −12.3% | 168,750–191,972 | 154,851–172,931 | 191.81→174.15 |
| Fixed SIMD | 541,564 | 534,889 | −1.2% | 538,303–550,344 | 530,616–540,740 | 60.72→56.22 |
| Relaxed SIMD | 533,810 | 533,295 | −0.1% | 510,748–576,213 | 509,752–563,068 | 62.54→55.12 |

60 searches per cell. SIMD throughput is essentially flat within spread; scalar loses throughput. Lower fixed-depth wall time here reflects different/fewer searched nodes, not proof of strength. The diagnostic fused-float probe is not part of this A/B table.

Yukon Ryzen Node/WASI second datapoint, after match ended, cores 9–11, depth 11, same 20 FENs × 3 interleaved/rotated repeats (one-shot runtime, not browser/persistent): scalar **124,794→115,977 NPS (−7.1%)**, fixed SIMD **529,979→529,335 (−0.1%)**, relaxed **552,660→566,379 (+2.5%)**. Repeat-median ranges: scalar old 122,335–126,479 / new 114,384–116,730; SIMD old 515,957–538,128 / new 514,605–537,500; relaxed old 545,200–562,358 / new 558,588–580,136. Do not equate Node startup-inclusive wall time with browser steady-state throughput.

Deterministic sizes, bytes, Brotli quality **4**:

| Tier | Old raw | New raw | Old Brotli | New Brotli |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 64,577,670 | 64,557,776 | 43,032,565 | 43,018,885 |
| SIMD | 64,579,221 | 64,556,310 | 43,036,404 | 43,023,340 |
| Relaxed | 64,578,859 | 64,555,953 | 43,037,133 | 43,025,260 |

Size savings are negligible (<0.04%); network dominates. Trial external WASI: 1,293,156 raw / 189,576 Brotli; external API: 1,240,948 / 168,959, **plus** unchanged NNUE 63,266,880 / 42,834,856. External NNUE chiefly improves delivery/cache separation, not compute.

Three fresh-worker prewarms per artifact, initial compilation plus two browser-code-cache passes; local uncompressed assets, not CDN delivery latency. Median [min,max] ms: scalar old **148.81 [73.32,430.49]**, new **117.81 [59.22,176.69]**; SIMD old **147.70 [61.27,154.94]**, new **144.00 [59.23,173.28]**; relaxed old **123.36 [60.41,174.52]**, new **147.55 [64.04,150.61]**. Cache-state spread dominates; no reliable startup gain claimed.

## Validation and review

Attempted upgrade: typecheck **0 errors/warnings**, focused **38/38**, full **809 tests: 806 pass, 0 fail, 3 skip**, diff check clean. Required Codex branch review (`gpt-6.1-sol`, xhigh, base main) returned **clean, no accepted/actionable findings**, confidence .91. These checks did not override the empirical bench failure.

Final evidence-only tree: typecheck **0 errors/warnings**, focused **33/33**, full **804 tests: 801 pass, 0 fail, 3 skip**, diff check clean. Final Codex branch review (`gpt-6.1-sol`, xhigh, base main) was **clean, no accepted/actionable findings**, confidence .95 (`perf-upgrade-reckless-review-2.md` in session scratchpad). No runtime/build/lock/manifest or source-archive delta is retained.

Ignored local evidence `.local_engines/reckless-upgrade/`: original/fused diagnostic bench logs, native parity JSON, `browser-full.json`, `browser-startup-full.json`, `browser-summary.json`, `sizes.json`, x86 reports, match PGN/config/log, trial `release-assets/`, `trial-source-packages/`, source round-trip and `attempted-upgrade.patch`. Yukon scope: `~/upgrades/reckless/`. Large trial binaries/networks are not committed; no LFS publication occurred.

## Process retrospective

Manual/fragile/slow steps and what should change:

1. **Weak corpus coverage was the decisive miss.** Hundreds of rotated/warm comparisons from one opening family were reassuring but missed float differences elsewhere. Run upstream's full native/WASM bench signature and varied tactical/endgame/activation-sensitive FENs **before** costly Elo sampling. Passing SIMD counts/source-shape checks is not semantic compatibility.
2. Upstream selection/SPRT interpretation, tag ancestry and engine-vs-network version naming were manual. Persist machine-readable ref/ancestry/license/network/bench facts; do not borrow parent-only SPRTs for old-pin strength claims.
3. String patch anchors broke when upstream added its JS/WASM runtime; an initial API PV-access edit also needed correction. Replace broad text splices with bounded, fail-closed contracts plus source review. Float operation/reduction changes need dedicated cross-platform kernel tests, not only shape guards.
4. Direct API corresponding-source coverage and version discovery were fragile. Trial packaging produced all seven archives and proved one offline round-trip; automate every variant's extraction/rebuild/hash check. The old release wrappers duplicated scalar builds; cache/reuse trees without mixing variant provenance.
5. Native harness support, incomplete-output rejection and repeated/interleaved timing needed trial additions. Browser lease/acquisition/result extraction was still manual; explicit sessions prevent another agent disturbing the default page. Capture independent evidence, release locks reliably, and report cache state/spread.
6. Three-core Elo sampling took ~85 minutes. Automate a resumable config/seed/book and a stated precision/decisive stop rule; preserve penta statistics, incomplete-pair exclusions and timeout/crash counts. Stop earlier when a correctness gate fails.
7. **Scope incident:** initial bare Yukon `npm install` in an empty benchmark folder walked to `/home/ddbb`, removing 30/changing 2 home-level packages. Reported immediately; unknown prior state was **not restored without permission**. Corrected to an explicit benchmark package.json and `npm --prefix ~/upgrades/reckless/wasmbench`; owner review/restoration remains necessary. Require cwd/package-root/prefix assertions before every package-manager action. Mac node_modules was copied locally before tooling to avoid writing through the main-checkout symlink.
8. Code review checked adapter structure/provenance but did not execute benchmark semantics. Review is complementary to hard gates, never permission to ignore contradictory empirical evidence.

Concrete automation: `engine:upgrade-audit reckless --old-ref ... --candidate ... --slot 9-11` should enforce roots/affinity/benchlock, retrieve and freeze source/license/net metadata, build all seven variants, inspect per-function SIMD, check adversarial NNUE kernels and full native bench **first**, run broad exact parity, then interleaved browser/Node NPS/size/startup and resumable paired fastchess. Emit a gate matrix; absent or failed correctness results prevent publication.

Family-specific opportunities **not applied**: explore portable SIMD f64 intermediates for exact f32 FMA, with proven midpoint/subnormal/overflow fallbacks to strict scalar FMA and adversarial/property tests, then measure cost; do not assume double-rounding is harmless. SIMD threat generation remains untapped; compare upstream's new shuffle/dot against the older standard emulation. Build-cache reuse, dependency pinning and upstream wasm-bindgen threading deserve separate controlled experiments. Blindly restoring relaxed float FMA is not a portable exactness proof.

## Rebuild and publication boundary

**Do not publish the failed candidate artifacts or apply the trial patch as a release.** Production pin and tracked archives/manifests remain unchanged; canonical candidate outputs were removed from `public/reckless/` after preserving research evidence.

To retry: first resolve/prove float FMA parity without unacceptable fallback regression; repeat the entire seven-variant native/WASI/API/external matrix and all gates. Then rebuild WASM/NNUE, all matched source archives (including API), manifests, immutable representation hashes and Brotli encodings together from the approved ref/toolchain; validate extracted source round-trips and WASM MIME/COOP/COEP. Publication/deployment/push needs a separate authorized step; none was performed here.
