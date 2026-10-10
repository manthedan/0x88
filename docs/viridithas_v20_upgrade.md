# Viridithas v20.0.0 tag upgrade

Branch `perf/upgrade-viridithas-20`. The previous pin was the 2026-06-03 pre-release commit `20d7402065cae084715183e019fdd18089e2dfac`, which already called itself v20.0.0 and embedded `atlantis-b800`. This upgrade moves to the real tag.

| | Old pin | v20.0.0 tag |
| --- | --- | --- |
| Commit | `20d7402065cae084715183e019fdd18089e2dfac` | `0631113e2215714c54d397cce9a8caebc5e6f8ee` |
| Date | 2026-06-03 | 2026-06-27 |
| Cargo version / licence | 20.0.0 / MIT | 20.0.0 / MIT |
| Network | `atlantis-b800.nnue.zst` v106 | `sandhi-s2-b200.nnue.zst` v109 |
| Network sha256 | `2d387407b926df4dbda441cdc3e2288fee2e6a2afa8e1bd22262309ec0fb668a` | `066c172e22c25ee3d3b4084ad44ff761426f906a9c536cabd0127d17413d0585` |
| Official `bench` nodes | 3684965 | 3749873 |

The tag commit is "Update WDL model. (#449)". It changes the WDL normalisation (`NORMALISE_TO_PAWN_VALUE` 222 to 101) on top of the sandhi retrain. Sandhi is required: the pawn-pawn feature layout (`PAWN_TUPLE_FEATURES = 4560`, threat features 60144 to 59808, `l0_aux`) does not match atlantis. Yukon native `bench` reproduced both node totals exactly.

## Why not v21 or current master

The next commit, `2001ac64f3a28d2008dca81c54b8d1d480da3d24` (2026-07-06, "Relicense to AGPL-3.0-only (as of v21)"), is the parent-side successor of the tag. Its `Cargo.toml` becomes version 21.0.0 and `license = "AGPL-3.0-only"`, and `LICENSE` is replaced with the AGPL-3.0 text. That commit's own bench signature is still 3749873, the same as the tag, so it is not a strength change.

Later networks on the AGPL-era code are small:

- `acolyte` v110: +3.69 ± 2.35 Elo at 40+0.4, 19226 games (viridithas-networks release, test 745).
- `hyperion` v111: +6.52 ± 3.70 Elo at 40+0.4, 7940 games (test 749).

Those gains are not a reason to leave MIT. The published v20-versus-v19 claim (+60.84 ± 5.92 at 60+0.6, 3000 games) is against v19, not against our previous pin. The strength result that matters is the Yukon match below.

Engine `LICENSE` at the tag was re-read: MIT, copyright 2022-2025 Cosmo Bobak. The network repo's `LICENSE` is CC0-1.0, added in `1040c3ee` on 2026-06-08, before the v109 sandhi release.

## Checklist

### 1. Pin

`engines/viridithas/upstream.lock.json` ref is `0631113e2215714c54d397cce9a8caebc5e6f8ee`, version `v20.0.0`, licence MIT, network sandhi v109 with the sha256 above. The same pin is in `scripts/build_viridithas_wasi.mjs`, both manifest writers, `NOTICE.md`, `public/artifact-index.json`, and the engine docs. `patches/viridithas-wasip1.patch` applies and compiles with no rewrite. The corresponding-source archive was regenerated (60,442,439 bytes, sha256 `d7eeae4a84e13fb2b30494adccc3bcc407d1ae616fa2f87be727ba52122288b2`). The release manifest is regenerated after the browser timing run so its brotli pass does not share the machine with a timed benchmark.

### 2. Silent-scalar trap

`scripts/inspect_wasm_simd.mjs` on the new artifacts:

| Artifact | Bytes | SIMD opcodes | Relaxed ops |
| --- | ---: | ---: | --- |
| `viridithas.wasm` | 55763309 | 0 | none |
| `viridithas-simd128.wasm` | 55739644 | 3437 | none |
| `viridithas-relaxed-simd128.wasm` | 55739115 | 3390 | `i32x4.relaxed_dot_i8x16_i7x16_add_s` × 40 |

Scalar having zero SIMD opcodes is the scalar tier. The SIMD and relaxed builds are not silent-scalar.

### 3. Vector gates

`src/nnue/simd.rs` selects `simd_wasm.rs` only for `wasm32` plus `target_feature = "simd128"`, and `simd_scalar.rs` for wasm without that feature. `layers.rs` calls those `simd::` kernels for the feature-transformer activation and the L1 dot. There is no separate scalar matmul behind a flag that wasm would miss. The wasm weight load in `network.rs` always `permute(true)` and returns. The native `use_simd = x86_64 || neon` gate is on the mmap path, which wasm does not take. AVX2 / AVX-512 / VBMI cfgs stay off on wasm32, so those layers use the non-VBMI implementations, which the wasm SIMD module then vectorises.

### 4. Relaxed-dot range

Unchanged from the pre-release pin: `QA = 255`, `FT_SHIFT = 9`, and `layers.rs` documents keeping the left-hand activation in `0..127`. The stored activation is `(a * b) >> 9` after a saturating narrow, so the maximum is `255 * 255 >> 9 = 127` and negatives become 0. That is the i7 operand range of `i32x4.relaxed_dot_i8x16_i7x16_add`. `QA`, `FT_SHIFT`, and the layer code are not in the pin-to-tag diff. The relaxed tier stays because fixed-depth parity against scalar and against native also held (item 6).

### 5. Build flags

This is a Rust engine. The C/C++ `-O2`/`-O3` × LTO matrix does not apply. Tag `Cargo.toml` `[profile.release]` is `lto = true`, `codegen-units = 1`, `panic = "abort"`, `strip = true`. Release opt-level stays the Cargo default of 3. WASM builds use `--no-default-features` (no syzygy, no `final-release` zstd) and `RUSTFLAGS` `-C target-feature=+bulk-memory`, plus `+simd128` or `+relaxed-simd` for those tiers. Native Elo binaries used the same feature set plus `-C target-cpu=native` (Zen 2 AVX2 on Yukon).

### 6. Parity

Harness: `scripts/reckless_wasi_node_bench.mjs`, 20-position rotated Ruy Lopez suite, depths 7 and 8, Hash 16, Threads 1. Exact means best move, score, nodes, and PV.

| Comparison | Pairs | Exact |
| --- | ---: | ---: |
| SIMD wasm vs scalar wasm | 40 | 40 |
| Relaxed wasm vs scalar wasm | 40 | 40 |
| SIMD wasm vs Yukon native `viridithas-new` (`0631113e` + sandhi) | 40 | 40 |

The native binary identified itself as `Viridithas 20.0.0-dev` (the tag's Cargo version was already 20.0.0). Node counts match across wasm scalar, wasm SIMD, wasm relaxed, and AVX2 native, so the relaxed dot is value-exact on this suite and the wasm patch does not change fixed-depth search.

### 7. Strength

Yukon fastchess, physical cores 6–8, concurrency 3, `Threads=1`, `Hash=16`, UHO Lichess 4852, `tc=8+0.08`, 1000 rounds with `-repeat` (2000 games), draw 40/8/10, resign 3/600, seed 200063.

Engines: `v20pre-20d74020` (atlantis) vs `v20.0.0-0631113e` (sandhi). Elo is reported for the old pin relative to the tag, so a negative number means the tag is ahead.

Stopped after the 480-game rating interval, once the 95% half-width was about ±16 Elo. 518 games had finished; fastchess was then stopped (pid 3576767 only). The printed line is the result.

**The tag leads by 33.40 ± 15.90 Elo.** LOS that the old pin is stronger was 0.00%. At 480 games the old pin had 97 wins, 143 losses, and 240 draws (45.21% score). Pentanomial (0–2) was [2, 83, 114, 41, 0]. nElo for the old pin was −65.73 ± 31.08.

Earlier intervals, tag lead: 80 games 65.92 ± 36.93 (LOS 0.02%), 160 games 54.74 ± 27.37, 240 games 42.19 ± 22.33, 320 games 37.05 ± 19.60, 400 games 35.74 ± 17.95. The lead stayed positive while the interval tightened.

fastchess logged "Illegal PV move" warnings from the old pin only. Games continued. The tag contains "Prevent invalid PV extensions" (`801281b`), which is a plausible reason those warnings are one-sided.

Official native `bench` NPS is not a strength or speed comparison: old 694064 nps at 3684965 nodes, new 614058 nps at 3749873 nodes. The search and the network both changed.

### 8. Speed, size, startup

Browser NPS was measured on the lab page under the shared Mac bench lock, persistent mode, startpos, movetime 1000 ms, Hash 16, hash cleared between runs, three interleaved pairs of the new artifacts and the previous pin's artifacts, all three tiers. Startup is one-shot depth 6 cold wall on the same page.

| Artifact | Raw bytes |
| --- | ---: |
| New scalar / SIMD / relaxed | 55763309 / 55739644 / 55739115 |
| Previous scalar / SIMD / relaxed | 57442854 / 57419901 / 57419372 |

The new binaries are smaller because sandhi's feature layout is smaller than atlantis, not because the search was stripped.

Browser measurement (headless Chromium, Apple Silicon, `crossOriginIsolated`, Relaxed SIMD validated, persistent WASI available). Three interleaved rounds of new then old. Persistent startpos, movetime 1000 ms, Hash 16, hash cleared. The figure is the median of the three round-level average NPS values, with the min–max of those rounds.

| Tier | New NPS | Old NPS | New / old |
| --- | ---: | ---: | ---: |
| Scalar | 68306 (67645–68507) | 78458 (78157–78600) | 0.871 |
| SIMD | 457878 (430448–469164) | 488501 (488215–491165) | 0.937 |
| Relaxed | 531667 (524440–532439) | 573298 (568656–573436) | 0.927 |

On the new pin, relaxed is +16% over SIMD (532k / 458k) and SIMD is 6.7× scalar. The new SIMD round spread is wider than the old pin's (430k–469k versus 488k–491k); the median is still below every old SIMD round. The tag is slower per second and stronger in the match. Native `bench` NPS moved the same way (694064 → 614058) and is not itself the speed comparison.

One-shot depth 6 cold wall, median of two rounds: new scalar 632 ms, SIMD 599 ms, relaxed 598 ms; old scalar 603 ms, SIMD 583 ms, relaxed 588 ms. Every new tier searched 1765 nodes; every old tier searched 1529. Startup is still dominated by instantiating the ~53 MiB module.

Brotli quality 11 (Node zlib, same quality as `scripts/r2_brotli_publish_assets.mjs`): scalar 52,722,486 bytes, SIMD 52,718,628, relaxed 52,718,504. The three artifacts sum to 158,159,618 brotli bytes against 167,242,068 raw (ratio 0.946). Almost all of each file is the already-compressed network, so Brotli barely shrinks it. Gzip level 9 totals 158,473,373 bytes.

## Process retrospective

Manual or fragile:

- The "v20.0.0" string in the old lock was a pre-release commit, not the tag. An upgrade audit has to resolve the ref with `git rev-parse <tag>` and compare it to the lock, then re-read `LICENSE` and `Cargo.toml` at that commit.
- The embedded network is not named in `Cargo.toml`. The right file was the one whose published bench signature matched the tag (`3749873`), confirmed by a native `bench` on Yukon. A wrong net still loads and searches.
- The WASI patch applied cleanly only after a real build. `git apply --check` on a freshly checked-out tree is the right first probe; it should be one command.
- SIMD proof required reading `simd.rs` cfg selection, the wasm `permute(true)` path, and `layers.rs`, then `scripts/inspect_wasm_simd.mjs`. Opcode count alone does not prove every layer uses the vector module, but a zero count does prove the silent-scalar failure.
- Fixed-depth parity already had a harness (`scripts/reckless_wasi_node_bench.mjs`) but no native side. The native half was a one-off Python UCI loop.
- Browser NPS could not see the relaxed tier, which is the product default, until a lab checkbox existed. Old-versus-new needed same-origin URL overrides because the page's artifact paths are constants and the static server caches `.wasm` for an hour.
- The Elo match, the Mac bench lock, and Brotli-q11 of a 53 MiB wasm fight over the same cores. The manifest writer always compresses unless told not to, so a provenance rebuild is accidentally a timing hazard.
- Launching fastchess over `ssh` with `bash -lc '... &'` left a login shell in `do_wait` for the life of the match. `setsid` plus `nohup` kept the match alive, but the ssh session did not return.
- A fresh worktree had `node_modules` linked and no `.svelte-kit/tsconfig.json`. Lab mode turns the SvelteKit plugin off, so Vite never generates that file, and oxc then refuses to transform the lab module (`Tsconfig not found`). The HTML still shows `Ready.`, which looks like a harness timeout rather than a missing generated tsconfig. `svelte-kit sync` (also what `npm run typecheck` runs) fixes it.
- A match watcher that searched for `Elo:` also matched `nElo:` and would have waited on the wrong interval. The pattern needs a letter boundary.

Concrete automation: one `engine:upgrade-audit <engine>` that reads the lock, prints tag versus pin, licence files, network sha, release-profile LTO and codegen-units, wasm SIMD opcode counts per tier, and the relaxed-dot precondition (`QA`, `FT_SHIFT`, activation max). A second command, `engine:upgrade-parity <engine>`, runs the rotated suite for scalar, SIMD, and relaxed, then the same suite against a native binary of that commit. Browser A/B should take artifact URLs as arguments so old and new tiers can be interleaved without swapping files. fastchess should be started with `systemd-run --user` or a double-fork that exits the ssh, and the watcher should be the thing that stops at ±15 Elo.

Not yet applied to this family: the C-engine `-O3`/LTO matrix is the wrong tool. The Rust release profile already enables fat LTO and one codegen unit. The open speed work is the wasm NNUE itself (the relaxed dot is already on) and persistent-session startup, which still dominates one-shot wall time. No further upstream commit is available under MIT.

## Republish

Do not publish from this branch. After merge, rebuild is not required again if these artifacts ship: the three `public/viridithas/*.wasm` files, `viridithas-wasip1-corresponding-source.tar.gz`, and `viridithas-wasip1.manifest.json`. `scripts/r2_brotli_publish_assets.mjs` is what uploads them. Historical files in `public/releases/` stay as the records of earlier publishes.
