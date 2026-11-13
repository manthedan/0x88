# Big-net smolgen split-K (November 2026)

Does the Lc0 Small split-K rewrite
([`lc0_small_splitk_2026-11.md`](lc0_small_splitk_2026-11.md)), restricted to the
smolgen compress GEMVs, pay off in the big nets t3, BT4 and Leela Queen Odds
(LQO) at the product's search level, and on which devices? §7 of that doc left
these as gated candidates that still needed a paired search A/B.

**Verdict.**

| net | Apple M4 (WebGPU) | RTX 3090 (Chrome/Vulkan) | WASM EP |
|---|---|---|---|
| t3 | **ship**: 0.93× per invoke, 0.90-0.94× per move at 64/256/800 visits, search parity clean under the tie protocol | do not ship: GPU busy 0.87× but wall 1.05× (paired) | never served (fallback runs the original) |
| BT4 | do not ship: per-invoke ratio 0.98-1.00, within noise | do not ship: wall 0.98× with the GPU busy at the start (timing invalid) | never served |
| LQO | do not ship: 0.95× per invoke and 0.95-0.97× per move, with gate WDL at 99% of its limit and deterministic search flips that no control could clear (§5) | do not ship: wall 0.99× (noise), busy 0.90× | never served |

So t3 joins Lc0 Small under the same rule: ORT WebGPU sessions on an Apple
adapter load `t3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx`, and
everything else keeps the shipped export (§6). BT4 and LQO keep their shipped
models everywhere. The artifact is in `manifest.json` and the artifact index
but is not published to R2. Publishing and the release manifests are left to
the owner (§7).

Setup is the one in the Lc0 Small doc: Apple M4 (10-core GPU, `apple /
metal-3`), headless Chrome 157, the production `dist-client` plus the lab
overlay, served by `scripts/serve_isolated_static.mjs`, every M4 timing inside
`notes/kernel-hillclimb/benchlock.sh`. Other agents used the GPU outside
benchlock during the session and the load average reached 400+ at times, so
absolute times swing; only paired ratios are quoted. Ratios are A/B with A =
split-K and B = the shipped model, so below 1 is faster; each comes with its
median absolute deviation (MAD) and p10-p90.

## 1. Artifacts

`scripts/kernel_hillclimb/splitk_rewrite.py --include 'smolgen/dense1/w$'
--assume-m 8` rewrites the 15 smolgen dense1 MatMuls of each net (M = 8,
K = 2048, N = 256, S = 16: 8 → 128 workgroups) and nothing else. A second run
into a fresh directory was byte-identical for all three.

| net | split-K file | bytes | sha256 | source bytes / sha256 prefix |
|---|---|---|---|---|
| t3 | `t3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx` | 171,021,068 | `ffb4dbdc37a359b25076465d8b29b9a114846b306094b7b768b3a755e1e7f75f` | 171,016,323 / `fff9de3c` |
| BT4 | `BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.splitk.onnx` | 186,681,088 | `9a54ed7d3e99bde6785e19d15a3f69a59d1572ffc127f33aa63da4029ef8064d` | 186,676,343 / `af5319e8` |
| LQO | `lqo_v2.f16.qdq8.splitk.onnx` | 95,667,477 | `62c67ca195072d5832fd20285eefcc82077be98287b75af137deed7dc14b7b68` | 95,662,716 / `4effd307` |

```sh
.venv-onnx/bin/python -I scripts/kernel_hillclimb/splitk_rewrite.py \
  public/models/lc0/<shipped>.onnx artifacts/kernel-hillclimb/splitk/<shipped>.splitk.onnx \
  --include 'smolgen/dense1/w$' --assume-m 8 --summary <out>.json --check
```

Tools: Python 3.12.13, onnx 1.21.0, numpy 2.4.6, protobuf 7.35.0. `--check`
(onnxruntime 1.26.0, CPU EP, split-K vs original) reports max-abs t3 policy
0.0039 / WDL 0.00049 / MLH 0.031, BT4 0.952 / 0.0513 / 7.4, LQO 0.607 /
0.0215 / 0. The CPU EP runs the QDQ nets in a different precision mix from the
browser, so these are not the gate, but they are one more reason the WASM
fallback loads the original bytes rather than the split-K ones.

## 2. Gate (M4)

`ort-<net>-splitk --gate` (in-page `parity=wasm`: WebGPU candidate vs the
shipped model on ORT WASM, 10 fixture positions, the preset's thresholds). All
three pass. The shipped WebGPU model under the same gate is shown for scale.

| net | arm | policy | WDL | MLH | legal prior | best move |
|---|---|---|---|---|---|---|
| t3 | split-K | 0.0557 | 0.0044 | 0.141 | 0.0049 | 10/10 |
| t3 | shipped | 0.0625 | 0.0034 | 0.203 | 0.0072 | 10/10 |
| BT4 | split-K | 0.302 | 0.0024 | 0.906 | 0.0122 | 9/10 (tie, reference gap 0.00028) |
| BT4 | shipped | 0.294 | 0.0029 | 0.969 | 0.0130 | 10/10 |
| LQO | split-K | 0.898 | 0.0198 | 0.195 | 0.0565 | 9/10 (allowlisted flip) |
| LQO | shipped | 1.297 | 0.0085 | 0.125 | 0.039 | 8/10 (2 allowlisted) |

LQO's split-K WDL deviation, 0.0198 against a limit of 0.02, is 2.3× the
shipped model's. That is the first warning about LQO; §4 shows where it lands.

## 3. Per-invoke paired A/B

`ort-<net>-splitk-ab`: ABAB in one page session, 20 rounds of 10 timed
invokes per arm, per-position wall and GPU-busy ratios. Three repeats on the
M4, one on the 3090.

| device | net | wall ratio, median (MAD) [p10-p90] per repeat | busy ratio |
|---|---|---|---|
| M4 | t3 | 0.927 (0.034) [0.865-1.027]; 0.924 (0.025) [0.870-0.959]; 0.933 (0.014) [0.891-0.990] | 0.924-0.928 |
| M4 | BT4 | 0.996 (0.045) [0.944-1.168]; 0.982 (0.017) [0.964-1.018]; 0.999 (0.016) [0.939-1.059] | 0.986-0.999 |
| M4 | LQO | 0.950 (0.031) [0.862-1.014]; 0.950 (0.038) [0.837-1.006]; 0.961 (0.023) [0.873-0.991] | 0.951-0.957 |
| 3090 | t3 | 1.052 (0.065) [0.885-1.432] | 0.873 |
| 3090 | BT4 | 0.982 (0.024) [0.934-1.056], `timingValid: false` (GPU 100% busy at start) | 0.933 |
| 3090 | LQO | 0.993 (0.041) [0.886-1.106] | 0.899 |

On the M4 the t3 gain is ~7%, about 6.6 → 6.1 ms per position. LQO gains ~5%
and BT4 nothing measurable: at batch 8 BT4 is dominated by its large GEMMs.
On the 3090 the GPU-busy ratio improves (0.87-0.93) but wall time does not, the
same split as Lc0 Small on that card: the extra dispatches cost more than the
saved GPU time. All 3090 runs passed the gate (t3 policy 0.057 / WDL 0.0029,
10/10; BT4 0.235 / 0.0024, 9/10 with a tie at gap 0.0078; LQO 1.0 / 0.011,
9/10 allowlisted). The 3090 exposes `shader-f16` only with a Dawn flag, which
the remote runner sets. Stock Chrome on that Linux box runs the f16 nets on
WASM, so this row describes an NVIDIA WebGPU path, not a common user setup.

### WASM EP

The same paired page with `ep=wasm` (6 rounds, M4 CPU): t3 split-K/shipped
wall 0.989 (MAD 0.029) [0.959-1.035], LQO 0.984 (MAD 0.005) [0.973-1.005];
BT4 was not measured.
The rewrite is neutral on WASM for these nets, unlike Lc0 Small (1.21×
slower), because the 15 rewritten MatMuls are a small share of the work. It
does not matter for the product: WASM sessions are never given the split-K
bytes, and the WebGPU-failure fallback loads the original (§6).

## 4. Search-level paired A/B (M4)

New lab page `lab/kernel-hillclimb/search-ab.html`
(`search-<net>-splitk-ab`). It runs the product search path, `Lc0OnnxEvaluator`
→ `CachedLc0Evaluator` (2048 entries) → `Lc0PuctSearcher` at batch 8, pipeline
depth 1 and `yieldEveryMs` 16, as `searchWorkerCore.ts` builds it, on the
page's main thread. Each search gets a fresh tree and cache (the first search
of a Play game), arm order alternates, and the positions are 12 ECO openings
plus both queen-odds starts (searched as given, without Play's `swapColors`
for LQO as Black). Search options are deterministic: no draw score
or ScLimit (ScLimit samples `Math.random`); LQO keeps Play's cpuct 1.5 and FPU
0.4. Budgets are Play's big-net levels 3-5 (64, 256, 800 visits); 14 positions
× 2 repeats = 28 paired searches per cell.

Parity uses the f16 tie protocol of
[`lc0_tvmjs_bt4it332_onboarding_2026-06-09.md`](lc0_tvmjs_bt4it332_onboarding_2026-06-09.md):
a different best move is tolerated when the two moves have equal root visits in
either tree, or the reference (shipped) prior gap is ≤ 0.01. Anything else is
a flip. (The runs below kept only each tree's top 8 root children for this
check, so a move ranked lower in the other tree would have read as a flip with
an undefined prior gap. The page now keeps them all. No row below was affected:
every non-matching row had both moves in both top 8s and a finite gap, and
every flip's gap is ≥ 0.036.)

| net | visits | wall ratio median (MAD) [p10-p90] | ms per move A vs B (median) | visits/s A vs B | best move: match / tolerated / flip |
|---|---|---|---|---|---|
| t3 | 64 | 0.925 (0.056) [0.294-1.138] | 581 vs 631 | 110.2 vs 101.4 | 26 / 0 / 2 |
| t3 | 256 | 0.900 (0.092) [0.571-1.450] | 2015 vs 2273 | 127.1 vs 112.6 | 28 / 0 / 0 |
| t3 | 800 | 0.941 (0.029) [0.875-0.989] | 3814 vs 3978 | 209.8 vs 201.1 | 28 / 0 / 0 |
| LQO | 64 | 0.951 (0.050) [0.867-1.127] | 1112 vs 1171 | 57.6 vs 54.7 | 24 / 4 / 0 |
| LQO | 256 | 0.947 (0.043) [0.874-1.043] | 4872 vs 5140 | 52.5 vs 49.8 | 26 / 0 / 2 |
| LQO | 800 | 0.971 (0.035) [0.912-1.064] | 17455 vs 17826 | 45.8 vs 44.9 | 26 / 0 / 2 |

The v64 and v256 cells ran while other agents loaded the GPU (a few searches
took 3-7× their pair, hence the wide p10-p90); the medians agree with the
quieter v800 cell and with §3. BT4 was not search-tested, since its
per-invoke ratio is already within noise. Each flip below appears twice
because the search is deterministic: the flipped rows repeat exactly in both
passes.

### The flips, and a noise control

To tell a split-K regression from ordinary f16 drift, the same page ran the
shipped model on WebGPU (A) against the shipped model on ORT WASM (B), the
reference that `npm test` ties to native lc0 (`abEp=wasm`; the timing ratio is
meaningless there).

- **t3, 64 visits: Smith-Morra position**
  (`r1bqkbnr/pp2pp1p/2np2p1/8/2B1P3/2N2N2/PP3PPP/R1BQK2R w KQkq -`).
  - Split-K plays d1b3 with root visits 31 (d1b3) vs 26 (e4e5).
  - The shipped model plays e4e5, 27 vs 30.
  - The reference prior gap is 0.12, so the tie rule does not cover it.
  - In the control, the shipped WebGPU model plays e4e5 again, but the WASM
    reference plays **d1b3 with the same 31 / 26 visits as split-K**.
  - So the split-K model agrees with the reference there; the outlier is the
    shipped WebGPU model. The control's other 13 positions match.
  - Not a split-K regression.
- **t3, 256 and 800 visits:** 56 / 56 match.
- **LQO, 64 visits:** four visit ties, all tolerated. Two are the
  Benko-type position (`b1c3` vs `c4b5`, 31 / 31 in the shipped tree). The
  other two are the human-White queen-odds start (`e2e4` vs `d2d4`): 13 / 13
  in the shipped tree but 13 / 8 in the split-K tree.
- **LQO, 256 visits: human-White queen-odds start**
  (`rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -`, Black's queen
  removed, White to move). In Play this is the human's move. LQO plays Black
  with `swapColors` and opens from its book, so the page searches a root
  that Play never gives LQO. The other start (LQO as White) matched at every
  budget.
  - Split-K plays e2e4 with root visits 85 (e2e4) vs 54 (g1f3).
  - The shipped model plays g1f3, 23 vs 80.
  - The shipped model prefers e2e4 on prior by 0.036, so this is a genuine
    flip under the tie rule.
  - The split-K tree values e2e4 at Q 0.652 against the shipped tree's
    0.526.
- **LQO, 800 visits: same position.**
  - Split-K plays b1c3, 214 (b1c3) vs 97 (g1f3).
  - The shipped model plays g1f3, 46 vs 164. Reference prior gap 0.045.
- **LQO noise control: not obtained.** All three attempts at the shipped
  WebGPU vs WASM run crashed the page's renderer during the LQO WASM arm:
  14 positions at 64 visits, 14 at 256, and only the two queen-odds starts.
  The page was left on `about:blank`; `bench.mjs` now fails fast on a lost
  page instead of polling until its timeout. The t3 WASM control ran
  cleanly. So it is unknown whether the WASM reference sides with split-K
  or with the shipped model in that position.

## 5. Decision

- **t3: ship on Apple WebGPU only.**
  - Per invoke: 0.93× on the M4, consistent across three paired repeats.
  - Per move at Play's big-net budgets: 0.90-0.94×, about 50-260 ms saved per
    move (581 vs 631 ms at 64 visits, 3814 vs 3978 ms at 800).
  - The gate passes with smaller policy and MLH deviations than the shipped
    model.
  - Search parity is clean once the one flip is traced to the shipped model
    itself (§4).
  - The 3090 shows the Lc0 Small pattern: GPU time down, wall time up. So
    NVIDIA, AMD and Intel stay on the shipped export.
  - The cost is a 171 MB re-download for Apple users who already cached t3.
- **BT4: do not ship.** The per-invoke ratio is 0.98-1.00 on the M4 and
  0.98 on the 3090, inside the noise. At batch 8 BT4's time is in its large
  GEMMs, not the smolgen GEMV.
- **LQO: do not ship (marginal; the owner may weigh it differently).**
  - The gain is real but small: 0.95× per invoke, 0.95-0.97× per move.
  - Its WDL deviation sits at 99% of the gate limit (0.0198 vs 0.02), 2.3×
    the shipped model's. One unlucky device or ORT update would fail the
    gate.
  - Its search moves away from the shipped model's more than t3's does.
    There are two deterministic flips, both in the human-White queen-odds
    start, and the root Q moves by up to 0.13. The WASM-reference control
    that cleared t3's flip could not be run for LQO (§4).
  - That position is not one LQO searches from in Play. There the human
    moves first, LQO plays Black with colours swapped, and LQO's own
    opening moves come from `lqoOpeningBook.ts`. So no shipped Play
    behaviour is known to change. The 12 ECO positions and the LQO-White
    start match or tie at every budget.
  - For about 4% per move, Apple users who have LQO cached would download
    96 MB again, and the gate margin would be gone. Not worth it now. A
    re-export with a tighter WDL deviation (for example f32 accumulation in
    the rewritten MatMuls) would be the way to revisit it.

## 6. What shipping changes

The Lc0 Small mechanism is reused as is; t3 is one more row in its table.

- **Selection.** `LC0_BIG_NET_SPLITK_FILES` in
  `src/lc0/lc0SmallModelSelection.ts` maps the shipped t3 file to its split-K
  file, and `lc0SplitKUrl` covers Lc0 Small plus that table.
  `selectLc0OrtModelUrl` keeps its one rule: split-K only when the first
  resolved ORT provider is `webgpu` AND `GPUAdapterInfo.vendor` is `apple`.
  NVIDIA, AMD, Intel and WASM keep the shipped export, and so do BT4 and LQO
  on every device.
- **Who opts in.** `Bt4WorkerSearcher` (every big net in Play, Analysis
  and Arena; t3 is offered in Analysis and Arena, not in v0 Play) now sends `deviceModelSelection: true` to the search worker.
  Big-net URLs are fixed (there is no `?model=` override), so the device rule
  always applies; for BT4 and LQO it is a no-op.
- **Fallbacks.** Unchanged from Lc0 Small. If the split-K bytes cannot be
  loaded, for example before the R2 publish, the original is served. If the
  WebGPU session fails, the WASM attempt loads the ORIGINAL URL through
  `wasmFallbackModel`, so WASM never runs the split-K bytes. Covered by
  `tests/lc0_small_model_selection.test.mjs`.
- **Caches.** The bytes are cached and sha256-verified under their own URL.
  The worker reports `servedModelUrl`, which `Bt4WorkerSearcher` now keeps
  (`servedModelUrl`, reset on invalidate). The Analysis evaluation-line cache
  key for a big-net row now includes it, so a line computed by one model is
  never shown as the other's. Because the URL is known only after init,
  `analyzeCurrent` first initializes each selected big net with the options
  its search passes (`ANALYSIS_BIG_NET_SESSION`), including a pooled session
  that Play or Arena left with another eval-cache size. A stopped or left run
  makes no further init attempt. The run reads nothing else before that init
  and its position, MultiPV, engine rows, settings and keys only after it,
  and it ends in favour of a fresh run if anything was queued for after it
  meanwhile (an Lc0 runtime reload, engine disposal, a settings re-analysis)
  or the engine rows changed, so it never searches a backend the new settings
  replace. The run then records each net's served URL
  once (`analysisBigNetServedModel`, as `lc0ServedModelUrl` does for Lc0
  Small), and every key reads that record, never the live worker. A worker
  crash mid-run therefore changes no key, and if a search finds its session
  no longer serves the recorded URL, that engine's row fails. So the URL in
  the keys is the URL of the session that ran the search
  (`tests/analysis_bignet_served_model.test.mjs`). The Arena status line
  names the served URL.
- **Download cost.** An Apple user who already cached the 171 MB t3 export
  downloads 171 MB again on the next t3 load. The old entry stays until the
  model cache's LRU bounds (8 entries, 1 GB) evict it. Users who never loaded
  t3 pay nothing extra.
- **Lab.** `search-<net>-splitk-ab`, `ort-<net>-splitk` and
  `ort-<net>-splitk-ab` candidates in `scripts/kernel_hillclimb/candidates.mjs`
  for all three nets, and the new `search-ab` page
  (`scripts/kernel_hillclimb/README.md`).

## 7. Release steps left for the owner

1. **Publish the artifact to R2** at
   `/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx`
   (171,021,068 bytes, sha256 `ffb4dbdc…7f75f`). The local copy is in
   `../models/lc0-bestnets/onnx/`, and `npm run lc0:prepare-model-assets`
   links it into `public/models/lc0/`. Until it is published, Apple sessions
   log `split-K model unavailable` and load the original, so the code can ship
   first.
2. **Add the entry to the release manifests** `r2-v0-present.manifest.json`
   and `r2-stable-present.manifest.json`, as the Lc0 Small release did
   (commit `1647ac47`). This branch adds it only to `manifest.json` and
   `public/artifact-index.json`.
3. **Deploy and check on an Apple device.** Analysis with t3 should fetch
   and cache the split-K URL. With `?ep=wasm` the big nets are unavailable
   (`probeBt4Support`), so the WASM side of the rule is reached only through
   the WebGPU-failure fallback, which is unit-tested.
4. **Built-preview smoke (done on the M4).** On this branch's product build,
   choosing Lc0 t3-512 in Analysis and accepting the large-net prompt left
   exactly `/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx`
   in the `lc0-browser-models-v1` cache. The original was only HEAD-probed for
   availability. Lines rendered, with no error text on the page.
5. **Not yet run:** a non-Apple browser smoke of the t3 path (the rule is
   unit-tested), an Arena smoke, and a contempt-on (`contempt=1`) search A/B.
   Contempt only changes the search, not the evaluator, so the per-move ratio
   should carry over.

## Reproduce

```sh
# artifacts (byte-identical on rerun)
.venv-onnx/bin/python -I scripts/kernel_hillclimb/splitk_rewrite.py public/models/lc0/<shipped>.onnx \
  artifacts/kernel-hillclimb/splitk/<shipped>.splitk.onnx --include 'smolgen/dense1/w$' --assume-m 8 \
  --summary <out>.json
# build: product + lab overlay (copies artifacts/kernel-hillclimb/splitk/*.onnx to /models/lc0-splitk/)
BUILD_SCOPE=product npm run build:client && node scripts/kernel_hillclimb/build_lab.mjs
node scripts/serve_isolated_static.mjs dist-client <port>
# M4, inside benchlock
node scripts/kernel_hillclimb/bench.mjs --candidate ort-t3-splitk --gate --base-url http://127.0.0.1:<port>
node scripts/kernel_hillclimb/bench.mjs --candidate ort-t3-splitk-ab --gate --repeat 3 --base-url ...
node scripts/kernel_hillclimb/bench.mjs --candidate search-t3-splitk-ab --gate --query 'repeats=2&visits=800' --base-url ...
# noise control: shipped WebGPU (A) vs shipped WASM (B)
node scripts/kernel_hillclimb/bench.mjs --candidate search-t3-splitk-ab --gate \
  --query 'model=/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx&abEp=wasm&visits=64' --base-url ...
# RTX 3090 (one queued job at a time; retry after 10+ min on exit 4)
node scripts/kernel_hillclimb/remote/run.mjs --host yukon --dist dist-client --with-models --timeout 900 \
  --page 'lab/kernel-hillclimb/ort-profile.html?preset=t3&parity=wasm&model=/models/lc0-splitk/<split-K file>&ab=t3&abRounds=20&profileInvokes=0&iters=10' \
  --include 'lab/kernel-hillclimb/**,assets/**,ort/**,fixtures/lc0/**,models/lc0/<shipped file>,models/lc0-splitk/<split-K file>' \
  --out <result>.json
```
