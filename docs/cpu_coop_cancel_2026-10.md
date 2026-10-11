# Warm CPU cooperative cancellation — October 2026

## Decision

Enable persistent **Viridithas v20 only** in Analysis when SharedArrayBuffer and
cross-origin isolation are available. Retain one-shot execution otherwise.
Do not start a Berserk rebuild: this pilot is the scoped deliverable, and its
review must finish before a second engine is considered.

The initial kill criterion did **not** fire: shipped relaxed-SIMD depth-12 wall
was 932 ms persistent versus 1,734 ms one-shot, a 46.2% saving over three positions.
This is a local runtime/initialization result, not an Elo or network-transfer claim.

## Before any WASM/app build

Measured the actual `ViridithasEngine` and `recklessWasiWorker` using Node workers,
the browser WASI shim and **the shipped WASM**, under the shared `SP/benchlock`.
An untimed repetition preceded five alternating A/B repetitions, reversing order
on alternating repetitions. Hash=128 MiB, Threads=1, MultiPV=1; root=startpos,
then `moves e2e4`, then `moves e2e4 e7e5`; depth=12. Each three-request sequence
starts with a new adapter; persistent TT survives between positions, as on the
Analysis continuity path. The probe omits HTTP transfer/preload and Analysis's
initial depth-1 preparation. It instruments first `info depth` in the one-shot
worker; the product one-shot adapter actually publishes only completed results.

Node v24.16.0, macOS arm64, Apple M4. The initial harness bundled the TS adapters
for Node; the checked-in `scripts/cpu_cancel_probe.mjs` now uses Node's native TS
loading, so reruns need no bundling/build. The worker fetch substitutes local file
reads for HTTP and counts actual `WebAssembly.instantiate` calls. No engine source,
WASM, or product build occurred before this baseline.

All figures are median [min–max], milliseconds, excluding the untimed repetition:

| Shipped mode | Instantiates | First info: position 1 / 2 / 3 | Total wall |
| --- | ---: | --- | --- |
| One-shot, no cancel | 3 | 654 [635–784] / 423 [419–433] / 434 [416–437] | 1734 [1693–1926] |
| Persistent, no cancel | 1 | 657 [633–677] / 0.208 [0.194–0.211] / 0.158 [0.150–0.170] | 932 [900–948] |
| One-shot, cancel middle at 200 ms | 3 | 644 [630–776] / none / 654 [615–921] | 1654 [1615–2202] |
| Persistent, cancel scheduled at 200 ms | 1 | 666 [631–1009] / 0.242 [0.221–0.255] / 0.163 [0.155–0.179] | 942 [894–1318] |

**Important:** all five persistent middle searches finished before 200 ms. Their
cancel timers were cleared, not acknowledged. Conversely, one-shot cancelled
while initializing; its worker instantiated before cancellation but had no info.
Neither row proves warm search cancellation. A separate depth-20 middle search,
cancelled at 200 ms after a completed depth-12 search, supplies that proof below.
Do not compare the different completed work in the cancelled baseline rows as a
cancellation speedup.

## Implementation and lifecycle

- `patches/viridithas-wasip1.patch`: at the existing `SearchInfo::check_up`
  boundaries (normally every 1024 nodes, also iteration boundaries), poll one
  byte from optional **nonblocking WASI fd 4**. This fd represents a separate
  host-owned shared stop word, not stdin. No per-node JS import, WASM shared
  linear-memory requirement, thread change, or search-heuristic change.
- `recklessWasiWorker.ts`: optional `stopBuffer` adds fd 4 only for this adapter.
  Other WASI engines' descriptor layouts remain unchanged. The poll is sticky;
  only the host resets it. Argv-driven UCI runs bypass it; built-in `bench` has no
  stop descriptor, so EBADF means “not stopped.”
- `viridithasEngine.ts`: abort sets the flag, queues `stop`, and keeps the request
  queue occupied until its `bestmove` (or the sync request's `readyok`) is drained.
  The cancelled promise rejects with AbortError; it cannot publish its info or
  bestmove. A 100 ms watchdog terminates on missing acknowledgement, including
  older/custom WASM and cold initialization. Timers/listeners are cleaned up on
  completion, failure, and disposal. Incarnation checks reject late worker errors
  or messages after replacement.
- `analysisBrowser.ts`: remove only Viridithas's `forceOneShot`; explicitly use
  `newGame` before its existing depth-1 preparation at intentional session resets.
  Normal same-game positions do not reset. No other engine's policy changes.

Focused tests cover graceful drain before replacement, discarded cancelled info,
stop-word reset, queued abort, watchdog fallback, stale worker messages/errors,
disposal while draining, sync cancellation, and non-isolated/forced one-shot.

## Gates

Interleaved shipped-versus-candidate, ten pairs per tier, one untimed pair;
persistent startpos/e4/e4-e5 at depth 18 (SIMD tiers), depth 16 (scalar). NPS is
**total nodes / summed engine-reported search time**, excluding startup; not the
median of unrelated position rates. Each pair has identical nodes/bestmove/score/PV.
The throughput test actually exercises the new poll, unlike argv-only testing.

| Tier | Baseline NPS median [range] | Candidate NPS median [range] | Median throughput cost | Stop median / p95 (range), ms |
| --- | --- | --- | ---: | --- |
| Scalar | 70,994 [70,414–71,115] | 70,889 [69,233–71,136] | 0.15% | 8.10 / 15.33 [3.16–15.33], n=5 |
| SIMD128 | 394,735 [378,159–466,473] | 396,656 [388,704–464,983] | −0.49% | 1.23 / 3.54 [0.64–3.54], n=5 |
| Relaxed SIMD128 | 527,611 [504,552–528,996] | 528,570 [481,205–529,851] | −0.18% | 0.70 / 1.99 [0.34–1.99], n=10 |

Stop samples use depth-12 first/third positions and a depth-20 middle request,
aborted at 200 ms. **Every** persistent cancelled sequence used exactly one
instantiate and retained its worker; no watchdog fired. Relaxed-SIMD sequence
wall was 966 [953–1025] ms persistent versus 1595 [1585–1703] one-shot; third-request
first-info was 0.242 [0.229–0.265] versus 616 [613–656] ms. This one-shot timer again
interrupted initialization, not useful search, so report the accounting explicitly.
P95 is nearest-rank; five-sample fallback-tier estimates are necessarily weak.

The machine was shared and noisy despite serialization: SIMD rates drifted
substantially, and scalar cancellation controls had 8–13 s outliers. The paired
median costs all clear <1%; they are **not** evidence of a meaningful NPS gain or
a tight confidence bound below 1%. The initialization saving is much larger than
these fluctuations. No single-run timings were used for the decision.

### Full bench and fixed-depth equivalence

Ran `npm run engine:upgrade-audit -- viridithas` on separately retained shipped
artifacts and on the rebuilt public artifacts, under benchlock. All tiers executed
the **full built-in bench**; their total signatures are unchanged:

| Tier | Shipped nodes | Rebuilt nodes |
| --- | ---: | ---: |
| Scalar | 3749760 | 3749760 |
| SIMD128 | 3750315 | 3750315 |
| Relaxed SIMD128 | 3750315 | 3750315 |

The audit labels the native comparison SKIP when no native reference is supplied;
an independent strict JSON comparison establishes **shipped/rebuilt equality**,
not native equality. An initial audit using the documented native 3749873 reference
failed (−113 scalar, +442 SIMD/relaxed). Running the retained shipped artifacts
confirmed the same inherited discrepancy. This task does not waive or fix that
native correctness gap, and does not claim an upstream upgrade/native-parity pass.
Viridithas's built-in audit exposes a total signature, not position rows.

`reckless_wasi_node_bench.mjs`, the fixed 20-position Ruy Lopez suite, depth 9 and
12, Hash=16: **40/40 exact bestmove, score, nodes and PV** for each rebuilt tier
against shipped relaxed SIMD (120 comparisons). The separate persistent throughput
runs add 30/30 exact rows per tier against its own shipped tier (90 comparisons).

## Build, license and artifacts

Exact source is still MIT v20 `0631113e2215714c54d397cce9a8caebc5e6f8ee`; no v21+
source was fetched or built. Networks came from the checked-in source archive;
`sandhi-s2-b200.nnue.zst` SHA-256 is
`066c172e22c25ee3d3b4084ad44ff761426f906a9c536cabd0127d17413d0585`.
The exact pin alone was fetched to create the local Git object needed by the
existing source-archive script. Both Viridithas source-fetching scripts now fetch
only that pin, not all tags/history. Other engines' archive behavior is untouched.

Commands, all from this worktree:

```sh
# Preserve shipped artifacts before rebuilding.
mkdir -p .local_engines/cpu-cancel-baseline
cp public/viridithas/*.wasm public/viridithas/*.json public/viridithas/*.tar.gz .local_engines/cpu-cancel-baseline/
# Extract the shipped archive's v20 source and network to the build/net dirs.
# Initialize the source repo, then fetch only this object (no clone/tag fetch):
git -C .local_engines/viridithas-wasi-src fetch --depth=1 --no-tags origin 0631113e2215714c54d397cce9a8caebc5e6f8ee
# Existing scripts, installed Rust 1.95.0 / cargo 1.95.0, wasm32-wasip1.
VIRIDITHAS_SKIP_GIT=1 CARGO_NET_OFFLINE=true npm run viridithas:build-relaxed-simd-wasi
VIRIDITHAS_SKIP_GIT=1 CARGO_NET_OFFLINE=true npm run viridithas:build-wasi
VIRIDITHAS_SKIP_GIT=1 CARGO_NET_OFFLINE=true npm run viridithas:build-simd-wasi
npm run viridithas:source-archive
npm run viridithas:release-manifest -- --skip-compression-estimates
```

The release script retains Cargo's release LTO, codegen-units=1 and normal flags:
`+bulk-memory`, plus `+simd128` / `+relaxed-simd` for the respective tiers.
The new source archive includes the resolved Cargo.lock and the shared worker;
compression estimates were skipped, not reused from old bytes. Artifacts are
**Git LFS tracked**, contrary to the old WASI note; no force-add or publication.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `viridithas.wasm` | 55763466 | `058707d8941e01a27a85274f8ebdc4bda98ce865fb94d0a23066baea405e5683` |
| `viridithas-simd128.wasm` | 55739801 | `78904e7c6ed20a3cb4cc27552f2faea5f86d95d832372262798aaa822f6c214d` |
| `viridithas-relaxed-simd128.wasm` | 55739272 | `8f8680be8c5f06500179a1421d639e0b74c4b1be03f237e3edcaef3a4f6cc93b` |
| corresponding-source `.tar.gz` | 60464623 | `f1587c79d7d39fa7a12f42598ca6bad0fdee498ebbfd3fd7d6793589ffe2eeb5` |

Each WASM grew only 157 bytes. The release manifest is the authoritative digest
record if a subsequent review fix requires regenerating the source archive.

## Reproduce and locate evidence

`SP` is the delegated task's scratchpad. All timing commands are prefixed with
`"$SP/benchlock"`; do not rebuild while holding it.

```sh
"$SP/benchlock" node scripts/cpu_cancel_probe.mjs --repeat 5 --out "$SP/probe.json"
"$SP/benchlock" node scripts/cpu_cancel_probe.mjs --repeat 10 --cancel-depth 20 --out "$SP/stop.json"
"$SP/benchlock" node scripts/cpu_cancel_probe.mjs --repeat 10 --depth 18 \
  --compare-wasm .local_engines/cpu-cancel-baseline/viridithas-relaxed-simd128.wasm --out "$SP/nps.json"
"$SP/benchlock" npm run engine:upgrade-audit -- viridithas \
  --artifacts .local_engines/cpu-cancel-baseline --lock engines/viridithas/upstream.lock.json --json "$SP/before.json"
"$SP/benchlock" npm run engine:upgrade-audit -- viridithas --json "$SP/after.json"
"$SP/benchlock" node scripts/reckless_wasi_node_bench.mjs \
  --wasm .local_engines/cpu-cancel-baseline/viridithas-relaxed-simd128.wasm,public/viridithas/viridithas-relaxed-simd128.wasm,public/viridithas/viridithas-simd128.wasm,public/viridithas/viridithas.wasm \
  --depths 9,12 --positions 20 --hash 16 --json "$SP/suite.json"
```

Raw task artifacts: `cpu-coop-cancel-baseline.json`, `cpu-coop-cancel-candidate.json`,
`cpu-coop-cancel-throughput.json`, `cpu-coop-cancel-{simd,scalar}-{stop,throughput}.json`,
`cpu-coop-cancel-{baseline,final}-audit.json`, `cpu-coop-cancel-full-bench-comparison.json`,
`cpu-coop-cancel-fixed-suite.json`, and `cpu-coop-cancel-gates-summary.json`.

## Built product and verification

`BUILD_SCOPE=product npm run build:client`; isolated static server on **5183**;
agent-browser Chromium on `/app/analysis/`. Wrapped Worker construction/termination
and observed real UCI completion messages. Rapid e4/e5/Nf3/Nc6 navigation at depth
20, slower completed depth-12 navigation, and an explicit Stop followed by a
replacement request retained Viridithas worker **id 2**, with no termination or
persistent errors. Final PV corresponded to the final board, not a cancelled one.
With `SharedArrayBuffer` disabled before page code loads, only `run` messages were
sent; Stop terminated worker 2 and replacement worker 3 completed successfully.

Screenshots in SP, saved and read: `cpu-coop-cancel-fast.png` and
`cpu-coop-cancel-slow.png` (refreshed on the final built code, in one session).
Final transcripts are `cpu-coop-cancel-browser-final-fast.json` and
`cpu-coop-cancel-browser-final-slow-stop.json`; earlier transcripts include
`cpu-coop-cancel-browser-fast.json`, `cpu-coop-cancel-browser-slow.json`,
`cpu-coop-cancel-browser-explicit-stop.json`, and `cpu-coop-cancel-browser-no-sab.json`.
Browser sessions and server were stopped.

`npm run typecheck` clean; full `npm test` green (1014 pass, three expected environment skips).
Focused tests green. Biome is clean for the adapter, worker, build script, probe,
and focused tests. `analysisBrowser.ts` and `write_engine_source_archive.mjs` retain
only their pre-existing lint/format findings; no unrelated reformatting was done.
Codex review found one probe-only bug: a deeper request that finished before the
cancel timer was validated against the normal depth. The completed-result boundary
now validates the actual requested search depth; focused pure unit tests cover
that missed-cancel case and rejection of incomplete output. A real depth-1/depth-2
CLI recheck waited for an unrelated shared mutex holder until the foreground
timeout, without starting any measurements; no additional timing is claimed.
The mandatory final branch review and commit IDs are recorded in the task report.

Remaining limits: no mobile/other-browser performance claim; cold NNUE initialization
can still hit the watchdog; old/custom WASM safely restarts; p95 samples are small;
existing native full-bench divergence remains. No Berserk experiment, publication,
push, merge, deploy, or other worktree change.
