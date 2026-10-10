# C/C++ CPU engine compiler-flag matrix — October 2026

## Decision

**Promote `-O2 -flto` for the next Berserk promoted-tier build only.**

- Berserk: browser paired wall NPS **+29.42%** at depth 11, **+29.09%** in a separate depth-12 confirmation. Engine-reported aggregate NPS corroborates this: **+30.00% / +28.77%**. Every comparison preserves exact depth, bestmove, score, nodes, and full PV.
- `-O3 -flto` is +30.99% at depth 11 but +28.94% at depth 12: no repeatable extra benefit over `-O2 -flto`. Choose the smaller `-O2 -flto` artifact.
- PlentyChess: best browser paired wall gain **+3.96%** (`-O3 -flto`), below 5%. **Keep `-O2`, no LTO.**
- Stormphrax: best browser paired wall gain **+2.31%** (`-O2 -flto`), below 5%. **Keep `-O2`, no LTO.**

The Berserk builder defaults LTO on only when `BERSERK_WASM_RELAXED_SIMD=1`, the already-promoted tier measured here. Its optimization level remains `-O2`; scalar and regular SIMD fallback defaults remain unchanged. `BERSERK_EMSCRIPTEN_LTO=0` restores the previous build. All three builders accept explicit `<ENGINE>_EMSCRIPTEN_LTO=1` independently of their existing `<ENGINE>_EMSCRIPTEN_OPT` knob. `-flto` is passed at compilation and linking in the single Emscripten invocation; this is full LTO, not ThinLTO.

**Release pending, not published:** no JS/WASM/data binary, source archive, or existing release manifest was replaced or committed by this experiment. The manifest generator now records Berserk's recipe defaults in `build.defaultCompilerFlags`, and `public/berserk/README.md` marks the pending rebuild. The existing manifest and source archive still correctly describe the existing no-LTO binaries. Before release, rebuild the artifact pairs, regenerate the corresponding-source archive and manifest, and publish that matching set through the normal release process. Do not regenerate only a manifest against old binaries and call that a new release.

## Environment and controls

Measured on 2026-10-10, branch `perf/cpu-o3-lto`, starting repository commit `5761490ef5533d82870bbaf0b0d3529a3298ddd3`.

- Apple M4 MacBook Air, 10 cores (4 performance / 6 efficiency), 24 GB RAM; macOS 26.4.1, Darwin 25.4.0, arm64.
- Node **v24.16.0**, V8 **13.6.233.17-node.49**.
- Native `agent_browser` session **`cpu-o3-lto-5193`**, headless Chromium **149.0.7827.54**, arm, macOS 26.4.1. Reduced UA: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/149.0.0.0 Safari/537.36`.
- Production application built with `BUILD_SCOPE=product npm run build:client`; served by `node scripts/serve_isolated_static.mjs dist-client 5193`. `crossOriginIsolated=true`; the repository's relaxed integer-dot feature probe validates. Product home-page snapshot/title, console, and page-error checks passed.
- Every engine uses its **existing promoted default tier**, `*-emscripten-relaxed-simd128`. SIMD flags, source patch, NNUE, memory limits, tablebase settings, and synchronous search configuration remain fixed within each engine's matrix. No thread or SIMD implementation changes are part of this work.
- **Hash=16 MiB, Threads=1, MultiPV default 1**, `ucinewgame` and `isready` before every search; no time/node limit. The exact 20-FEN rotated suite previously embedded in `scripts/emscripten_uci_bench.mjs` is now shared through `scripts/emscripten_uci_protocol.mjs` without changing its positions/order.
- Emscripten is fixed **per engine**, including every one of its four candidates: Berserk and PlentyChess use the installed **5.0.7-git** toolchain; Stormphrax uses **6.0.2**, commit `7a2d97d627ff4945eae28847ce0387ac52b92c09`, local Docker image `emscripten/emsdk@sha256:644883f58ca15c38c8be59b3a727ba0eff347729bc31d50a3348a6c9ed92bc07`. These match the versions recorded by their existing manifests. No toolchain upgrade is mixed into a flag comparison.
- Sources and raw networks came from the current worktree's existing corresponding-source archives, without downloads. Each candidate gets a fresh upstream-source copy, an independent `git init` (so `git apply` is not filtered by the enclosing worktree), the same repository patch, and the same verified network. The local 5.0.7 cache was copied into this worktree; Docker builds use `--network none`.

### Pinned sources, patches, and models

| Engine | Upstream commit | Repository patch SHA-256 |
|---|---|---|
| Berserk 14 | `8ae895a6151695be4a50d4fb65b0c131659c513a` | `0e74e4da26a82b6c1886c14dbad29afef9320ceb2dda29dfb58e89a1c52247a7` |
| PlentyChess 7.0.66 | `58d8ba2505ae2b49f48dd410d214a457d15c12c6` | `c064512b84ab881e638638737288e337e883d15ed840c1c858b51379683ccd0a` |
| Stormphrax 8.0.0 | `582965517ed2032d41a6b4cd6c2e66b1b934e2ad` | `721917be224cd65f025cc88eb8894b3d47cd26d74296da5842caf893f30e41ca` |

| Engine / network | Raw network SHA-256 | Runtime preload SHA-256 / bytes |
|---|---|---|
| Berserk `berserk-9b84c340af7e.nn` | `9b84c340af7e45f6e07f0046235ccb327f4ae0840c8ee2c4b97b99121e5c5084` | Same SHA; 25201924 bytes |
| PlentyChess `0134-2r24-s0.bin`, processed with `tools/process_net false` | `550a0b664b68113fd228f501524b25e0cea1be500a608bb0f26d42a6255c8061` | `691efaca9d6b32c85be9256d55d852559f470c3ee67d8d4bdeaf8e113169d4d4`; 63023936 bytes |
| Stormphrax `undertown.nnue` | `04d651e078b7c7334709dbd772d40a23c0a5480e93e19521a03020c7d633f2cf` | Same SHA; 55914848 bytes |

All five artifact sets of each engine have the identical runtime-preload digest above. Corresponding-source archive SHA-256:

- Berserk: `fe42589368640366b667181f990614a84fe04616f957875406c380b4c3945a3d`
- PlentyChess: `822a26f9c33b392418502e46cf5ffa46dbb60468bee66ca8145bc09b055f2d23`
- Stormphrax: `f6e043ae1f7dcef7f798f71614abb8cca2a272a6a3acb86a9b26fe7bc7d1252a`

### Protocol and interpretation

1. Original harness parity gate: current artifact first, then freshly built `-O2`, `-O3`, `-O2 -flto`, `-O3 -flto`, all 20 positions at depths **9 and 11**.
2. Node screen and browser product gate: **10 rounds**, five artifact sets per engine, **20 positions at depth 11** per cell. Artifact order rotates and reverses on alternating rounds; over ten rounds each artifact occupies every slot twice. Each Node cell is a fresh subprocess; each browser cell a fresh classic Worker. One unreported start-position search at the measured depth warms the search before the measured suite.
3. Independent Berserk browser confirmation: **6 rounds**, current / `-O2 -flto` / `-O3 -flto`, **20 positions at depth 12**. Each artifact occupies each slot twice. This checks the LTO win and whether the larger `-O3 -flto` build adds repeatable value.
4. All engine execution/timing occurred under the supplied machine-wide mutex:
   `/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad/benchlock`.
   The Node parity-plus-screen runner was wrapped directly. Browser automation ran only while a `benchlock node .../hold-browser-lock.mjs` sentinel held the mutex, after its acquisition marker appeared; its release marker was written only after both browser batches completed and were captured. No overlapping timing runs were used.
5. Browser search follows the product's classic-Worker / modularized factory / `ccall('command')` path and posts each UCI stdout/stderr line to the parent, matching product transport overhead. The harness measures search inside the Worker, not rendered UI response latency. It is staged locally beside the production build, not shipped as a new product route.
6. **Wall NPS** is `sum(final nodes) * 1000 / sum(ccall go wall milliseconds)`; **UCI NPS** is the same numerator divided by the sum of final engine-reported `time` milliseconds. **Suite TTD** is the summed search wall time for all 20 positions, excluding reset, position setup, startup, and warmup. Printed per-position NPS and all per-position times remain in the raw reports.
7. Tables report **median [minimum, maximum]** across complete rounds, not percentiles or confidence intervals. Gains are medians of **same-round candidate/current ratios**, not ratios of medians. No sample was discarded. Large Node outliers remain visible, including the PlentyChess `-O3` 4.2-second suite and Stormphrax fresh `-O2` control tails. The byte-identical `-O2` controls and interleaved ratios help distinguish a broad frequency/runtime drift from a flag win.
8. The shipping decision requires exact parity, browser NPS gain at least 5%, repetition, and acceptable size/startup cost. Berserk `-O2 -flto` wins in **all 10 depth-11 rounds and all 6 depth-12 rounds**, with both wall and UCI NPS gains above 5% even at the minimum. Its depth-11 first/last five-round wall-gain medians are **30.44% / 28.00%**. PlentyChess and Stormphrax do not clear 5% at the median and are not promoted.

UTC batches:

| Batch | Start → finish | Cells / measured searches |
|---|---|---:|
| Node screen | `2026-10-10T20:57:05.421Z` → `2026-10-10T21:00:03.886Z` | 150 / 3000 |
| Browser gate | `2026-10-10T21:02:54.378Z` → `2026-10-10T21:05:44.377Z` | 150 / 3000 |
| Browser confirmation | `2026-10-10T21:06:24.426Z` → `2026-10-10T21:06:41.254Z` | 18 / 360 |

## Exact parity evidence

- Initial depths 9/11: **40/40 per candidate**, four candidates × three engines = **480/480** exact comparisons.
- Repeated depth 11: **200/200 per candidate** in Node and browser, **2400/2400 candidate comparisons in each runtime**.
- Independent Berserk depth 12: **120/120 per LTO candidate**, **240/240** comparisons.
- Node versus browser, matched engine/artifact/round/position at depth 11: **3000/3000** exact comparisons, including the current controls.
- Depth-11 suite final-node totals are fixed across every build and repetition: Berserk **513540**, PlentyChess **719251**, Stormphrax **865981**.
- Missing depth/score/nodes/PV output fails closed; full PV is compared and retained in mismatch diagnostics. A PV-only mismatch cannot pass the gate.
- Fresh `-O2` Berserk and PlentyChess glue **and wasm** are byte-identical to current. Fresh Stormphrax `-O2` is not byte-identical (432 additional wasm bytes), despite matching the documented toolchain version, pinned source/patch/network and all search outputs. That reproducibility delta is unresolved; neither its current baseline nor fresh control was silently substituted. Every Stormphrax candidate is compared against the actual current artifact, and no Stormphrax default changes.
- A final Berserk build with **no optimization/LTO overrides**, only the already-promoted tier selected, reproduced the measured `-O2 -flto` **JS, wasm, and data hashes exactly**. This verifies the new default really emits the tested artifact, not just a source-string expectation.

## Node depth 11

| Engine / flags | Wall NPS, thousands | UCI NPS, thousands | Suite TTD, ms | Paired wall NPS gain, % | Paired UCI NPS gain, % |
|---|---:|---:|---:|---:|---:|
| berserk Current (-O2) | 1497.9 [1249.4, 1526.4] | 1503.8 [1252.5, 1537.5] | 342.8 [336.4, 411.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| berserk -O2 | 1496.2 [1117.2, 1510.2] | 1497.2 [1114.0, 1533.0] | 343.2 [340.0, 459.7] | -0.3 [-21.0, 3.1] | -0.3 [-21.9, 3.5] |
| berserk -O3 | 1796.4 [1438.5, 1893.0] | 1790.4 [1454.8, 1916.2] | 286.1 [271.3, 357.0] | 23.0 [-4.5, 31.5] | 21.4 [-3.7, 31.8] |
| berserk -O2 + LTO | 1930.1 [1627.4, 1969.1] | 1923.4 [1625.1, 1998.2] | 266.1 [260.8, 315.6] | 29.2 [15.5, 31.7] | 28.0 [16.0, 32.3] |
| berserk -O3 + LTO | 1931.1 [1651.5, 1967.7] | 1930.7 [1651.3, 1967.6] | 265.9 [261.0, 310.9] | 29.7 [23.7, 32.2] | 29.3 [24.9, 31.8] |
| plentychess Current (-O2) | 1028.3 [861.3, 1045.8] | 1033.4 [862.4, 1045.4] | 699.5 [687.8, 835.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| plentychess -O2 | 1025.9 [874.6, 1053.8] | 1026.1 [877.1, 1053.1] | 701.1 [682.5, 822.4] | -0.8 [-3.1, 3.9] | -0.8 [-3.4, 3.9] |
| plentychess -O3 | 1032.0 [170.5, 1052.5] | 1037.9 [170.6, 1053.1] | 697.0 [683.4, 4219.4] | -0.0 [-81.9, 1.5] | -0.3 [-82.0, 1.5] |
| plentychess -O2 + LTO | 1057.5 [878.3, 1092.7] | 1058.2 [876.1, 1091.4] | 680.5 [658.3, 818.9] | 4.2 [-7.6, 6.9] | 4.2 [-7.5, 6.6] |
| plentychess -O3 + LTO | 1036.3 [906.5, 1076.4] | 1038.7 [905.9, 1078.3] | 694.6 [668.2, 793.5] | 3.1 [-7.5, 5.2] | 3.2 [-7.0, 5.0] |
| stormphrax Current (-O2) | 742.1 [287.5, 750.9] | 748.5 [288.6, 758.3] | 1167.0 [1153.2, 3011.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stormphrax -O2 | 731.2 [151.4, 742.8] | 738.6 [151.9, 749.8] | 1184.3 [1165.9, 5720.0] | -1.0 [-47.3, 2.9] | -1.0 [-47.4, 3.0] |
| stormphrax -O3 | 721.7 [597.1, 734.2] | 727.7 [600.5, 740.2] | 1199.9 [1179.5, 1450.2] | -3.1 [-11.4, 107.7] | -3.2 [-11.3, 108.1] |
| stormphrax -O2 + LTO | 752.2 [536.2, 761.6] | 759.3 [539.9, 770.4] | 1151.3 [1137.0, 1615.1] | 1.4 [-7.1, 86.5] | 1.4 [-7.0, 87.1] |
| stormphrax -O3 + LTO | 749.4 [499.1, 757.6] | 756.0 [502.6, 765.0] | 1155.6 [1143.0, 1735.2] | 0.6 [-7.1, 73.6] | 0.6 [-7.3, 74.2] |

## Browser depth 11

| Engine / flags | Wall NPS, thousands | UCI NPS, thousands | Suite TTD, ms | Paired wall NPS gain, % | Paired UCI NPS gain, % |
|---|---:|---:|---:|---:|---:|
| berserk Current (-O2) | 1326.3 [1125.5, 1542.5] | 1334.0 [1123.7, 1537.5] | 387.2 [332.9, 456.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| berserk -O2 | 1306.8 [1119.2, 1532.4] | 1305.1 [1121.3, 1528.4] | 393.0 [335.1, 458.9] | -1.1 [-5.8, 2.6] | -1.1 [-6.2, 3.2] |
| berserk -O3 | 1653.9 [1462.8, 1940.9] | 1675.5 [1475.7, 1952.6] | 310.5 [264.6, 351.1] | 25.9 [21.5, 30.0] | 26.8 [22.6, 31.3] |
| berserk -O2 + LTO | 1714.3 [1496.7, 1980.6] | 1714.7 [1510.4, 1990.5] | 299.6 [259.3, 343.1] | 29.4 [13.1, 33.8] | 30.0 [12.8, 34.4] |
| berserk -O3 + LTO | 1747.7 [1477.8, 2002.6] | 1755.7 [1471.5, 1982.8] | 293.8 [256.4, 347.5] | 31.0 [25.1, 39.2] | 30.5 [24.6, 41.4] |
| plentychess Current (-O2) | 885.6 [741.0, 1050.7] | 889.1 [740.7, 1054.6] | 812.2 [684.5, 970.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| plentychess -O2 | 882.8 [708.1, 1053.7] | 884.2 [707.9, 1060.8] | 814.8 [682.6, 1015.8] | -0.6 [-4.4, 1.7] | -0.7 [-4.4, 1.6] |
| plentychess -O3 | 902.5 [760.8, 1063.7] | 904.7 [762.7, 1068.7] | 796.9 [676.2, 945.3] | 0.7 [-4.7, 10.0] | 0.7 [-4.2, 9.9] |
| plentychess -O2 + LTO | 920.9 [761.9, 1109.2] | 922.7 [763.5, 1113.4] | 781.0 [648.5, 944.0] | 3.4 [1.1, 11.2] | 3.3 [0.5, 11.0] |
| plentychess -O3 + LTO | 928.7 [766.8, 1093.3] | 932.9 [769.3, 1096.4] | 774.5 [657.9, 938.0] | 4.0 [1.6, 8.9] | 3.8 [1.6, 8.6] |
| stormphrax Current (-O2) | 648.0 [596.2, 759.7] | 653.6 [601.4, 765.0] | 1336.4 [1140.0, 1452.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stormphrax -O2 | 645.4 [554.6, 765.8] | 650.4 [558.7, 772.5] | 1341.8 [1130.8, 1561.5] | 0.3 [-20.5, 1.2] | 0.1 [-20.6, 1.3] |
| stormphrax -O3 | 629.7 [537.8, 740.3] | 635.1 [541.2, 746.5] | 1375.1 [1169.8, 1610.3] | -2.2 [-18.0, -0.4] | -2.2 [-18.0, -0.5] |
| stormphrax -O2 + LTO | 656.6 [594.0, 781.6] | 661.6 [598.1, 788.7] | 1318.9 [1107.9, 1457.9] | 2.3 [-9.5, 6.6] | 2.3 [-9.5, 6.4] |
| stormphrax -O3 + LTO | 652.5 [616.8, 769.8] | 658.0 [620.8, 776.7] | 1327.2 [1124.9, 1403.9] | 1.6 [-0.1, 3.5] | 1.7 [0.2, 3.2] |

Half-batch wall-gain sensitivity (first five / last five rounds): Berserk `-O2 -flto` **30.44% / 28.00%**, `-O3 -flto` **31.30% / 30.69%**; PlentyChess `-O2 -flto` **2.82% / 4.02%**, `-O3 -flto` **3.90% / 4.03%**; Stormphrax `-O2 -flto` **-0.62% / 3.15%**, `-O3 -flto` **1.34% / 1.96%**.

## Browser depth 12 confirmation

| Engine / flags | Wall NPS, thousands | UCI NPS, thousands | Suite TTD, ms | Paired wall NPS gain, % | Paired UCI NPS gain, % |
|---|---:|---:|---:|---:|---:|
| berserk Current (-O2) | 1518.0 [1514.3, 1547.9] | 1526.1 [1511.2, 1538.2] | 510.7 [500.9, 512.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| berserk -O2 + LTO | 1966.7 [1931.4, 1994.5] | 1965.1 [1933.3, 1977.7] | 394.2 [388.7, 401.4] | 29.1 [26.1, 30.8] | 28.8 [26.4, 30.1] |
| berserk -O3 + LTO | 1955.1 [1950.9, 1980.7] | 1962.7 [1938.1, 1987.8] | 396.5 [391.4, 397.4] | 28.9 [26.0, 29.4] | 28.8 [26.0, 30.1] |

## Size and cold compile/startup

Cold here means a fresh subprocess/Worker and engine instance, **not** a flushed OS filesystem cache or a cold CDN. Both harnesses explicitly use `WebAssembly.compile` and instantiate that module, rather than mixing streaming and non-streaming paths. Node compiles the original bytes; the browser appends a unique ignored custom section to avoid identical-wire-byte module-cache hits, cache-busts glue URLs, and fetches wasm/data with `cache: 'no-store'`. The section changes no executable code and is not included in the size/hash ledger. Compile time is the compile-promise latency, not a claim that every function has completed optimized-tier JIT compilation. Startup ends at factory/module readiness, includes byte loading, glue loading, instantiation, preload and engine initialization, and excludes search warmup. Worker creation/dispatch and rendered UI latency are not included in startup. These numbers are a controlled localhost module-startup comparison, not field download performance.

Sizes are the original wasm bytes, compressed with Node Brotli **quality 11**, no executable custom-section nonce. All values are median [min, max].

| Engine / flags | Wasm raw / Brotli-11 bytes | Node compile, ms | Browser compile, ms | Node startup, ms | Browser startup, ms |
|---|---:|---:|---:|---:|---:|
| berserk Current (-O2) | 131480 / 50719 | 0.304 [0.274, 0.394] | 0.260 [0.250, 0.425] | 497.3 [441.2, 566.1] | 529.3 [428.2, 592.4] |
| berserk -O2 | 131480 / 50719 | 0.300 [0.270, 0.351] | 0.267 [0.240, 0.370] | 460.9 [435.8, 515.5] | 531.9 [447.5, 614.1] |
| berserk -O3 | 137856 / 52610 | 0.325 [0.295, 0.449] | 0.285 [0.235, 0.555] | 459.6 [435.9, 526.4] | 534.9 [476.3, 609.0] |
| berserk -O2 + LTO | 142181 / 53719 | 0.314 [0.279, 0.347] | 0.273 [0.250, 0.340] | 461.6 [445.5, 532.3] | 533.3 [464.5, 632.8] |
| berserk -O3 + LTO | 146839 / 54987 | 0.324 [0.306, 0.348] | 0.280 [0.255, 0.295] | 474.6 [449.1, 507.2] | 518.0 [448.5, 571.3] |
| plentychess Current (-O2) | 386827 / 121491 | 0.603 [0.538, 0.857] | 0.505 [0.450, 0.705] | 111.5 [110.2, 166.3] | 165.7 [141.2, 195.6] |
| plentychess -O2 | 386827 / 121491 | 0.594 [0.547, 0.681] | 0.490 [0.425, 0.605] | 116.3 [108.9, 145.9] | 164.4 [135.0, 382.8] |
| plentychess -O3 | 392390 / 120078 | 0.595 [0.552, 0.697] | 0.488 [0.415, 0.725] | 114.2 [109.0, 157.1] | 167.5 [136.1, 210.7] |
| plentychess -O2 + LTO | 434804 / 123066 | 0.617 [0.568, 0.872] | 0.522 [0.480, 0.735] | 116.2 [110.9, 181.2] | 172.2 [146.2, 199.4] |
| plentychess -O3 + LTO | 456875 / 121195 | 0.629 [0.613, 0.969] | 0.555 [0.505, 0.695] | 111.6 [107.7, 154.6] | 173.8 [139.4, 256.7] |
| stormphrax Current (-O2) | 697337 / 201411 | 0.830 [0.763, 4.912] | 0.685 [0.620, 0.820] | 33.6 [31.9, 69.8] | 71.0 [54.2, 85.9] |
| stormphrax -O2 | 697769 / 201406 | 0.869 [0.776, 1.786] | 0.683 [0.625, 1.255] | 33.6 [32.1, 204.7] | 68.8 [55.1, 117.0] |
| stormphrax -O3 | 698380 / 197736 | 0.819 [0.767, 0.882] | 0.712 [0.610, 2.850] | 32.7 [31.7, 51.7] | 69.9 [59.8, 177.3] |
| stormphrax -O2 + LTO | 632266 / 192868 | 0.779 [0.730, 1.253] | 0.635 [0.585, 0.655] | 34.6 [30.9, 59.8] | 69.9 [54.3, 155.9] |
| stormphrax -O3 + LTO | 648881 / 194923 | 0.877 [0.817, 1.538] | 0.717 [0.695, 0.865] | 32.7 [31.8, 60.3] | 72.1 [53.7, 90.2] |

Selected Berserk cost: **+10701 raw bytes (+8.14%)**, **+3000 Brotli bytes (+5.92%)**, against an unchanged 25.2 MB NNUE preload. Browser median compile changes by about **0.013 ms**; median startup changes from **529.3 to 533.3 ms (+0.76%)**, well inside the observed spread. This is an acceptable one-off cost for roughly 29% higher search NPS; no statistically resolved startup improvement is claimed. The depth-11 suite median falls from **387.2 to 299.6 ms**, and the independent depth-12 suite from **510.7 to 394.2 ms**.

## Artifact identity ledger

`Current` lives under `public/<engine>/`; fresh variants live locally under `.local-dev-artifacts/cpu-flags/artifacts/<engine>/<o2|o3|o2-lto|o3-lto>/`. Each uses `<engine>-emscripten-relaxed-simd128.{js,wasm}` and canonical `<engine>-emscripten.data`. Full SHA-256 values, before the browser-only cache nonce:

| Engine / flags | Wasm SHA-256 | Glue SHA-256 |
|---|---|---|
| berserk Current (-O2) | `0add80acd95319f51c60f152a2d4f44ae1f6c622f5b2752cc5235fe1709418bb` | `87805af5371be3fc0cb6a32ec844458a409a86b1878f48ae74279be433d88d6f` |
| berserk -O2 | `0add80acd95319f51c60f152a2d4f44ae1f6c622f5b2752cc5235fe1709418bb` | `87805af5371be3fc0cb6a32ec844458a409a86b1878f48ae74279be433d88d6f` |
| berserk -O3 | `27c6760cc79053ac60b7234d3a3afce25bcb6afd7306e5a860e89cebe327ef5d` | `2529254b4ef55a3237bef88e4ea8d07d3dab2f7909d418f7abf485ac1d871c9c` |
| berserk -O2 + LTO | `0b5ebb07a098c3404c21b69b03d68823fe6c6933a8c6a75c5605de865df77493` | `20c5907ba8edcc554f7d39e91f2444dc34e51fae0100cdae3ff326cd4eef1c0c` |
| berserk -O3 + LTO | `7e75b81bd61ccdce23256263bfd983a9cac7d3eb5337f0573af4a7998e96e9c1` | `f68f9da737dfcebb78044372064338b4d9324c3592313db206b5f4e465a162ae` |
| plentychess Current (-O2) | `c030812786c33a56f706b6c84ae36ca7934c7156542dcc9b6f36e9ca2f384dc4` | `850b884c518b23cd6d84e15f408a1fa0edad02f28dd5a4ea24199cc39fe8668d` |
| plentychess -O2 | `c030812786c33a56f706b6c84ae36ca7934c7156542dcc9b6f36e9ca2f384dc4` | `850b884c518b23cd6d84e15f408a1fa0edad02f28dd5a4ea24199cc39fe8668d` |
| plentychess -O3 | `2afa54316837822ca7fa08aeef3a597b281c5a79c609387663462e5bf46e5d87` | `dd4158afa24c2dbf899c7deb5e3ad1560099a65a8f0629580d074da9adaf1a77` |
| plentychess -O2 + LTO | `2488ea0051f32b7308f550792c994628809e1a40935a6d34b94b5eed5610cc5a` | `f7277d77855d7df191d1d18d4eb913961752160d52c8d4e3a6d300165173bdd9` |
| plentychess -O3 + LTO | `370f2165412840127f422a4750e031e0ed2542bd647460ed36fc576eef2dd19b` | `e2933b17f88ad17572e420a3cfc47cfd3e791ade4460df70bffefeec73345ea9` |
| stormphrax Current (-O2) | `cb39b34a7302b969ebe13c3137f00b0e950232fecc62d6c0c0b2296cf3faa4a6` | `1a57e31fdfe4df209f0bfd1bddf250e5a9f621ea74c033091dfa7376a88f9f0f` |
| stormphrax -O2 | `7421774998091388d163d9a260216bd26f169d7a5844cc0f3ae052ff992d730d` | `f274a404a00b004cf2d03477c8ed63b9d0518ce499284efb3d403525efd91f53` |
| stormphrax -O3 | `8f7b8834d3c2f1f58015d9e2125aeb807937a6a2d653e880bbeeb31fb3d3d2ad` | `1eb8b9f7dccb6a069e9e9ad5c90d24eb7ec6c9b2fde224f4aba4f70d7ca92093` |
| stormphrax -O2 + LTO | `4e93542b22befa9d05a26335809c76b077ceafe083940e8d46298cdf8e30ac4c` | `828cb209c856f7731b5473ea10837fc4aeac76815192190d4d17399354e1486f` |
| stormphrax -O3 + LTO | `c159232146fa3f3c9b8d5f4e53d0df813707aa5cd4b188a15d7c67ae73c0a414` | `a22c45a9d0e8aa7ad4e5100adb1579185b35d5573b1a3803184c622eca5c266b` |

## Reproduction and local evidence

The reusable harnesses are committed; binaries and full generated row reports remain local/gitignored. Config cells are `{engine,label,js,data,factory}`, with `label=current` identifying the actual shipped artifact. Example cell:

```json
{"engine":"berserk","label":"current","js":"public/berserk/berserk-emscripten-relaxed-simd128.js","data":"public/berserk/berserk-emscripten.data","factory":"Berserk"}
```

For every engine add `o2`, `o3`, `o2-lto`, and `o3-lto` cells with the local paths above. Preserve the current-first config ordering; rotation/reversal is performed by the harness. Build each from a fresh archive upstream copy, initialized as an independent git repository, with `*_SKIP_GIT=1`, `*_BUILD_DIR`, `*_NET_DIR`, `*_EMSCRIPTEN_JS_OUT`, `*_SHARED_DATA`, the promoted tier knob, and **explicit** `*_EMSCRIPTEN_OPT=-O2|-O3` / `*_EMSCRIPTEN_LTO=0|1`. Explicit `0` is important now that Berserk's promoted tier defaults to LTO. Pin the toolchain versions above; never use a floating Docker tag for an independent reproduction without verifying its local digest/version.

Node commands (substitute each engine's five JS paths in `--js`):

```sh
BENCHLOCK=/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad/benchlock
"$BENCHLOCK" node scripts/emscripten_uci_bench.mjs \
  --js <current.js>,<o2.js>,<o3.js>,<o2-lto.js>,<o3-lto.js> \
  --depths 9,11 --hash 16 --json <parity.json>
"$BENCHLOCK" node scripts/emscripten_uci_matrix.mjs \
  --config .local-dev-artifacts/cpu-flags/matrix.json \
  --rounds 10 --depth 11 --hash 16 \
  --json .local-dev-artifacts/cpu-flags/node-matrix.json
node scripts/summarize_emscripten_uci_matrix.mjs <report.json>
```

Browser staging, after the production build:

```sh
mkdir -p dist-client/cpu-flags/harness
cp scripts/emscripten_uci_protocol.mjs scripts/emscripten_uci_matrix_worker.js \
  scripts/emscripten_uci_browser_matrix.mjs dist-client/cpu-flags/harness/
ln -s "$PWD/.local-dev-artifacts/cpu-flags/artifacts" dist-client/cpu-flags/artifacts
node scripts/serve_isolated_static.mjs dist-client 5193
```

Map config `public/...` paths to `/...` and local artifact paths to `/cpu-flags/artifacts/...`, then drive the built product at `http://localhost:5193/` with native `agent_browser`, unique session `cpu-o3-lto-5193`. Hold the supplied benchlock with a sentinel command for the whole browser run, verify acquisition, import `/cpu-flags/harness/emscripten_uci_browser_matrix.mjs`, and call `runMatrix(config,{depth:11,hash:16,positions:20,rounds:10,onProgress})`. Capture the returned report before releasing the lock. Confirmation uses only Berserk current/LTO cells, depth 12 and six rounds. If native output is compacted, copy/parse the reported `fullOutputPath`; the small `outputPath` file may contain only a compaction pointer, not the measurement rows. Terminate Workers through the harness, release the mutex, close the browser session, and kill the assigned-port static server when done.

Evidence under `.local-dev-artifacts/cpu-flags/` includes all build logs, pinned config, complete stdout-derived rows, cold-start metrics, summaries, browser identity, production-build/test logs, and default-build hash proof. Report SHA-256:

| Local report | SHA-256 |
|---|---|
| `node-matrix.json` | `acf968c4929560bf764b9d33778b0fb2850b423dc9d5b5671422bbdaafabf558` |
| `browser-matrix.json` | `adff4628e7ae42eb42496666f9399d47da27ffe4334f06d87c0a33f56f3cce76` |
| `browser-confirm.json` | `5157ef591db53d5858a1a379ebf95f2ff3c1c769c4d6a5d7161b29bc05cd2682` |
| `parity-berserk.json` | `23b0af460c86b1f06661ee4b414e32179469af1cbaafca9be0213d6c48fefca2` |
| `parity-plentychess.json` | `59e93435b881d1923bed44f38e27c296bae45aadf0512692c7a62bfc054efdb0` |
| `parity-stormphrax.json` | `9c7c209ee77ed7c35adb2eea18daaa1687408cfbd011c9ebe8231576b7564e36` |
| `matrix.json` | `e2ba4324716ccb56d86931d9c21ed8f2b14b66a2c798d5a040dc9dfac9e3b2d8` |

## Verification and limits

- `npm run typecheck`: zero errors/warnings.
- Focused protocol/parity/ordering/default-flag tests, engine variants, canonical preload mapping, artifact registry and manifest totals: **67 passed**.
- Full `npm test`: **776 passed, 0 failed, 3 skipped** (779 total); the existing three skipped tests remain skipped.
- Production `BUILD_SCOPE=product npm run build:client`: passed, SvelteKit runtime-ID check passed; real native browser checks passed on port 5193.
- `git diff --check`: clean. Branch Codex review is a required closeout gate; its final outcome is reported with the commit, not inferred from benchmark parity.

Open questions are limited to portability of this compiler win beyond M4/Chromium, true field cold-download/startup costs, and the unresolved Stormphrax byte-reproduction delta. There is no evidence here to promote PlentyChess or Stormphrax flags, no promise of a universal hardware speedup, and no reason to change the unmeasured Berserk fallback-tier defaults. Release artifact/source/manifest rebuilding and publication remain intentionally outstanding.
