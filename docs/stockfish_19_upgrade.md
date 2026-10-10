# Stockfish 19 upgrade audit

Branch: `perf/upgrade-stockfish-19`. No deployment or push is part of this audit.

## Pins, licensing and networks

- Requested source/package upgrade: **18.0.7 → 19.0.1**. Old public lite files and
  source manifest pin `32d4b5ae40c01db88219bfbe2b82dbe6dec93832`. The inherited
  caret dependency had already resolved **18.0.8** in `package-lock.json`, so old
  full/pthread npm timing cells are explicitly labelled 18.0.8, not 18.0.7.
- New npm/source pin: `v19.0.1`, **`0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6`**.
  `scripts/stockfish_release_config.mjs` is the build/provenance identity source.
  `package.json` now uses an exact version, preventing another silent caret drift.
- Native release references: `sf_18` / `cb3d4ee9b47d0c5aae855b12379378ea1439675c`,
  `sf_19` / **`edb0d9db6731067ec50ce619ff372b463bc4dd5d`**.
- Re-read the pinned upstream `README.md`, `Copying.txt` and engine file notices:
  engine GPL v3 with source notices allowing later versions; npm GPL-3.0;
  non-engine Stockfish.js code MIT. Preserve both notices in Corresponding Source.
- SF18 full used `nn-c288c895ea92` plus `nn-37f18f62d772`; SF19 full uses one
  network, **`nn-1a298aa575a0.nnue`**, SHA256
  `1a298aa575a085434d29027978dc36867fe9c5bcea9376654b7a8eba1e52dfc2`.
- SF19 lite uses **`nn-61e7af4bb97d.nnue`**, SHA256
  `61e7af4bb97d51eeeb25d322916f86513b5cd3a827ce189c98c6e31946f99e5b`.
  It has `P_hm` mirrored piece-square features instead of the full engine's
  HalfKAv2_hm + FullThreats + PP_3Wide features. It is NOT just a smaller net that
  unmodified official SF19 can load. Native lite parity uses the exact pinned
  Stockfish.js source with `LITE_NET=yes`. That upstream native CLI has a pointer
  typo (`uci.loop()` on a `unique_ptr`); `patches/stockfish-js-native-audit.patch`
  fixes only that non-WASM line.
- The new source archive includes **both actual network files**, not merely
  download URLs. The builder validates full SHA256s, even for pre-existing files.
  A raw GitHub URL can return an LFS pointer; use Fishtest or the GitHub media URL.

Upstream sources: [Stockfish.js v19.0.1](https://github.com/nmrugg/stockfish.js/tree/v19.0.1),
[official sf_19](https://github.com/official-stockfish/Stockfish/tree/sf_19),
[network provider](https://github.com/official-stockfish/networks).

## Checklist 1–8

1. **Pin/provenance:** exact npm/lock pin, current runtime/asset/Netlify/R2 paths,
   source-archive generator and manifest generator updated. Source/license/net
   provenance is centralized and the archive contains the local runtime adapter,
   pthread bootstrap, build scripts and network assets.
2. **Silent scalar trap:** SIMD inspection results recorded below; zero opcodes
   are a failed gate, not an acceptable fallback for a purported SIMD build.
3. **Engine vector gates:** Emscripten Makefile enables `USE_SSE2`, `USE_SSSE3`,
   `USE_SSE41`, `-msimd128`, with `USE_AVX2`/`USE_PAIR_ACTIVATIONS` off. The actual
   compiled config confirms these, not just command-line intent. Accumulator
   update/refresh uses `SIMD::vec_t` SSE2 128-bit paths. Feature-transformer
   product activation takes the explicit `__wasm__` extmul/shuffle path.
   `fc_0` sparse affine uses the SSSE3 vector/dot gate; `fc_1` (64→32) uses the
   SSSE3 multi-output vector gate; `fc_2` (128→1) uses SSSE3 dot/horizontal sum.
   Both clipped ReLU and squared clipped ReLU take SSE2 vector paths. This is
   true for lite and full, which share the SF19 1024/32/32 architecture.
4. **Relaxed dot proof:** the helper's activation operand is second/i7 (`b,a,acc`
   means signed weights first, activations second). SF19 FT's wasm path saturates
   each accumulator to unsigned bytes, multiplies, takes the high byte then
   shifts by one: `floor(sat_u8(a)*sat_u8(b)/512)`, bounded by
   `floor(255²/512)=127`. Clipped hidden layers clamp to [0,127]; squared hidden
   layers also saturate to [0,127], including the changed SF19 skip-connected
   concatenation into `fc_2`. Padding is zero. Thus every dot activation has its
   high bit clear. The SSSE3 pair sum cannot saturate either:
   `2*127*128=32512 < 32768`; replacing it with the relaxed dot preserves integer
   arithmetic. SF19 already carries the intrinsic, so the local patch enables
   its gate and documents this proof rather than duplicating an old helper.
5. **Build flags:** lite/full O2/O3 × LTO on/off rebuilt with pinned Emscripten
   **6.0.9**. Lite builds run on Mac Docker ARM64; full builds run on Yukon Docker
   x86_64 (`--cpuset-cpus=12-14`, `nice -n 10`). The Mac VM has only 2 GiB and
   full-net clang embedding fails explicitly with `LLVM ERROR: out of memory`;
   it was not silently built scalar or retried after resizing shared resources.
   Full/pthread relaxed artifacts remain local candidates. Matrix results below.
6. **Parity:** Stockfish.js excludes the `bench` command under `__EMSCRIPTEN__`.
   The new harness parses the pinned upstream `benchmark.cpp` Defaults (51
   positions, including high-threat, terminal and Chess960 positions), then
   replays `setup_bench` through UCI, with ONE `ucinewgame` at the beginning.
   Native `bench` is separately checked against the replay's node sum before
   accepting it as the WASM reference. Rotated 20-FEN suite also requires exact
   depth/bestmove/score/nodes/PV; different old/new networks are never conflated
   into a parity comparison. Initial depth-10 signatures: **453169 full**,
   **632754 lite**. All four npm flavors plus relaxed lite single pass **100/100**
   rotated-FEN comparisons against their native variant references.
7. **Strength:** native official sf_18 vs sf_19 built on Yukon, AVX2; fastchess
   pinned to **cores 0–2**, concurrency 3, Threads=1, Hash=16, UHO_Lichess_4852_v1,
   TC **8+0.08**, paired colors, seed 191807; draw 40/8/10, resign 3/600.
   Stop policy: after at least 150 games, stop early only if the nominal 95% Elo
   lower bound is positive and LOS ≥99%; otherwise continue toward ±15 Elo.
   These are ordinary fastchess nominal intervals, not sequentially adjusted CIs.
8. **Speed/size/startup:** Mac browser/Node timing commands are wrapped by the
   shared benchlock, fresh engines, rotated/reversed A/B order, five rounds,
   Threads=1, Hash=16. Browser searches are depth 12 on 20 rotated FENs;
   cold Worker-to-`readyok` startup includes raw local asset load/compilation,
   not internet/CDN latency. Raw/Brotli manifest sizes and median/range below.

## Validation results

### Native full strength (completed)

**+57.86 ±24.82 Elo (nominal 95%), LOS 100.00%, 200 games**: 67 wins,
34 losses, 99 draws; penta `[0,9,54,32,5]`. The predeclared decisive stop fired
at the 200-game paired checkpoint; lower bound +33.04 Elo. Match finished
cleanly after SIGINT (28:35 elapsed); no failures or time forfeits recorded.
A separate native lite-vs-lite check is in progress because its feature
architecture changed; the full-release result is not a claim about lite strength.

### SIMD and native parity

| Relaxed O3/LTO build | SIMD opcodes | Relaxed dot opcodes |
| --- | ---: | ---: |
| Lite single | 5,648 | 48 |
| Full single | 6,086 | 48 |
| Lite pthread | 5,660 | 48 |
| Full pthread | 6,098 | 48 |

The four full flag cells have 3,406 / 3,941 / 4,583 / 6,086 SIMD opcodes
(O2 off/on, O3 off/on); every cell has 48 relaxed-dot opcodes, never zero.
Emscripten also emits one `f64x2.relaxed_madd`; exact parity was checked for
these complete modules, not merely a synthetic dot kernel.

- Initial native comparison: **100/100** depth-10 FEN pairs, 5/5 bench signatures.
- Yukon full matrix + lite pthread: **860/860** depth-12 FEN comparisons across
  five interleaved rounds; all 43 bench comparisons match native. This count
  includes repeat-native consistency checks, not just distinct WASM flavors.
- Mac lite matrix: **480/480** depth-12 FEN comparisons against the native-validated
  npm baseline; all 24 depth-10 replay signatures are `632754`.

### Build-flag NPS (five fresh-process rounds, 20 FENs, depth 12)

Wall NPS medians [minimum–maximum], not peak engine-reported NPS:

| Flags | Mac ARM64 lite | Yukon x86_64 full |
| --- | ---: | ---: |
| O2, no LTO | 1,448,354 [1,306,475–1,535,688] | 441,036 [436,791–445,153] |
| O2, LTO | 1,567,329 [1,474,996–1,586,302] | 443,951 [438,266–452,538] |
| O3, no LTO | 1,407,757 [1,136,525–1,482,658] | 441,355 [436,446–445,338] |
| O3, LTO | 1,502,637 [1,357,367–1,522,092] | 446,308 [442,640–453,773] |
| New npm baseline | 1,244,516 [917,100–1,250,133] | 391,408 [387,713–395,539] |

Yukon full O3/LTO relaxed is 1.140× npm baseline at T=1. Full pthread relaxed
is 450,686 [447,073–458,823], lite pthread relaxed 859,957 [831,479–864,938].
The Mac lite O2/LTO candidate is 1.043× O3/LTO; the browser comparison determines
whether to change the promoted lite flag default rather than extrapolating
this small Node-only difference.

### Transfer size (WASM bytes; Brotli quality 5, matching release staging)

| Flavor | Old raw / Brotli | New raw / Brotli |
| --- | ---: | ---: |
| Lite single | 7,295,411 / 5,486,553 | 1,845,971 / 1,148,651 |
| Lite relaxed (O3/LTO) | 7,287,678 / 5,485,067 | 1,845,006 / 1,150,813 |
| Lite pthread | 7,093,151 / 5,444,502 | 1,697,617 / 1,124,422 |
| Full single | 112,992,459 / 74,489,035 | 99,169,054 / 77,712,620 |
| Full pthread | 113,007,340 / 73,775,853 | 99,076,105 / 77,639,025 |

Lite drops roughly 75% raw / 79% Brotli; full is 12% smaller raw but **4–5%
larger compressed**, not a transfer-size improvement. New relaxed lite JS is
unminified (~159 KB) versus ~28 KB npm baseline; WASM/network savings dominate.
Actual transfer sizes depend on publishing quality; do not compare this Q5
measurement against the old manifest's Q11 estimate.

### Browser/startup and final gates

Mac Chromium old/new tier timing is queued behind the shared benchlock. Final
browser numbers and lite strength will be filled before closeout. Current gates:
`npm run typecheck` (0 errors/warnings), 38 focused tests, full `npm test`
(803 pass, 3 pre-existing skips, 0 failures), Stockfish-only browser asset check,
`git diff --check`. Browser smoke: new lite/full pthread, Threads=2, successful
bestmoves, cross-origin isolation true, no console/page errors.

## Reproducibility and release policy

- `npm run stockfish:build-relaxed-simd:{lite-single,single,lite-threaded,threaded}`
  builds ignored candidates. `--simd baseline --opt O2|O3 --lto on|off` supports
  controlled cells without new ad-hoc patches. Optional `STOCKFISH_BUILD_CPUS`
  passes a Docker CPU set; builds execute at nice 10.
- Default toolchain is `emscripten/emsdk:6.0.9`, not floating `latest`.
  The known downloaded image manifest digest is
  `sha256:96617f27fe16421588241def73908fd348a7f9d260440ed0d00b36dcf7a063cc`.
- SF19 supplies the 4 MiB single-thread search stack, explicit pthread compile
  and link flags, incoming module API list, and Emscripten 6.x `em-pthread`
  same-script bootstrap. The retired SF18 import whitelist, 1 MiB global stack,
  `worker-extern-post.js`/`startWorker` rewrites are deliberately NOT applied.
- Single-thread search still yields via Asyncify so stop/ponderhit can arrive;
  don't remove Asyncify merely because synchronous engines can avoid it.
- Only feature-detected **lite single relaxed** is a release candidate. Baseline
  fallback stays available. Full and pthread flavors keep the npm baseline.
  Chromium `Threads=2` smoke passes for npm lite/full pthread with no page or
  console errors. Broader SF19 Firefox/Safari compatibility is not established
  by the historical SF18 matrix.
- Local detailed evidence is ignored under `.local-dev-artifacts/stockfish-upgrade/`;
  Yukon raw logs/PGN/config/builds are under `~/upgrades/stockfish/`. Source of
  truth is the Mac branch; nothing is pushed from Yukon.
- Before release, rebuild/stage these pinned assets, regenerate source/engine
  manifest plus content-addressed release/channel manifests, precompress and
  validate immutable hash/headers, then publish JS/WASM/source as one coherent
  release. **No publication, deployment or push was performed here.** Existing
  historical release manifests/channel pointers are not rewritten in place.

## Process retrospective (required)

Manual/fragile/slow steps: resolving npm's annotated tag and unexpected old lock
18.0.8 drift; auditing each SIMD gate and activation formula; native lite's
upstream pointer typo; detecting that WASM has no bench command; copying archives
between hosts; flag matrix orchestration; Docker embedding memory; separately
maintained runtime/Netlify/R2/test paths; manually collating Elo and timing reports.
The source archive previously omitted actual Stockfish nets, and its rebuild
instructions needed the pinned source path and local helper imports made explicit.

Concrete automation proposals:

1. One `engine:upgrade-audit stockfish --from … --to … --slot 0-2` command should
   resolve/verify npm integrity + tag commit + license, stage its own dependencies,
   refuse a symlinked shared node_modules, archive hash-verified networks, validate
   the source archive from a clean extraction, and assert no stale release paths.
2. Generate runtime/asset/Netlify/R2 lists and tests from one flavor/pin schema;
   preserve immutable historical manifests while checking the new release end to
   end. Do not let caret packages silently change a GPL binary's source identity.
3. Cache Emscripten SDK/system libraries, preprocess/record SIMD macros and
   instantiated NNUE paths, inspect opcode counts automatically, and hard-fail a
   missing SIMD/relaxed-dot gate. Add adversarial [0,127] × signed-weight dot tests
   and full/native bench replay tests as mandatory CI promotion gates.
4. Persist/resume matrix cells with their JS/WASM hashes, enforce benchlock/core
   assignment in the harness itself, and summarize median/range, raw/Brotli,
   startup and parity counts automatically. Use a memory preflight and a direct
   binary/incbin embedding route rather than a huge generated C++ string where
   possible: the latter is the main avoidable full-build memory cost.
5. Script fastchess fixed-TC/seed/book/core settings plus a documented decisive
   early-stop policy, save raw penta counts and CI convention, and run isolated
   x86 WASM timing only AFTER the match/build workers leave the slot.

Unapplied family optimizations worth testing, not assumed wins: external NNUE
(shared cache instead of re-embedding/downloading per flavor), binary embedding
for build memory, cold-start streaming/precompiled module reuse, lower-frequency
Asyncify yielding, relaxed-dot sparse affine specialization/nnz discovery, and
browser-specific flags/toolchain/PGO. Do not claim Lazy SMP raw-NPS increases buy
extra depth: the runtime audit already rejected that inference for this family.
