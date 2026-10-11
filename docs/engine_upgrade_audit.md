# Browser-engine upgrade correctness audit

```sh
npm run engine:upgrade-audit -- viridithas --native-bench 3749873 --json audit.json
npm run engine:upgrade-audit -- reckless --native-bench 3634130
npm run engine:upgrade-audit -- berserk --upstream
npm run engine:upgrade-audit -- stormphrax --upstream
npm run engine:upgrade-audit -- stockfish \
  --native-cmd "ssh yukon \"bash -lc 'taskset -c 3 ~/upgrades/stockfish/sf19-native'\"" \
  --native-lite-cmd "ssh yukon \"bash -lc 'taskset -c 3 ~/upgrades/stockfish/js19-lite-native'\""
```

This is a correctness/provenance tool, **not a timing benchmark, builder, Elo
runner or release command**. No new dependencies. `--help` lists all options.
All WASM runs are foreground, sequential and bounded (default `--timeout
600000`, milliseconds per subprocess). Temporary clones, glue copies and output
files are under `os.tmpdir()` and removed after completion, failure or a handled
interrupt. Engines and shell commands run as trusted local code, not in a
security sandbox. `--native-cmd` is intentionally a trusted shell command.

## Candidates and pin selection

Artifacts default to `public/<engine>/`. Select a candidate with `--artifacts`;
the tool reads that public directory's sibling `engines/<engine>/upstream.lock.json`,
not this checkout's old lock. The tier inventory merges required tiers, on-disk
WASM files and manifest-declared WASM files; a missing declared tier is a FAIL,
including nonstandard external/API variants. Stockfish also requires its five
supported versioned release flavors (threaded/full, single/full, threaded/lite,
single/lite and relaxed single/lite), deriving filenames from the candidate,
not the current checkout. A changed layout requires adapter review. Stockfish
and Stormphrax use the artifact manifest's upstream pin. `--lock` and `--manifest` override this discovery. An arbitrary
artifact directory without a pin fails closed: provide explicit metadata. Multiple
manifests require `--manifest`. Existing files are checked against manifest hashes;
regenerate candidate metadata rather than mixing old manifests and new binaries.
The input directory is read-only, including when it is another worktree.

Example, PlentyChess **v8 candidate**, not main's 7.0.66:

```sh
npm run engine:upgrade-audit -- plentychess \
  --artifacts /path/to/upgrade-plentychess-8/public/plentychess \
  --native-cmd "ssh yukon \"bash -lc 'taskset -c 3 ~/upgrades/plentychess/upstream/new/engine'\"" \
  --json pc8-audit.json
```

## Gates, in execution order

### 1. Provenance (cheap local checks; optional upstream query)

Reports locked repository, commit/ref, version, license and network metadata from
the lock/manifest. Checks lock/manifest ref and license consistency, available
manifest artifact/corresponding-source archive SHA-256s, and separate/preloaded
network SHA-256s. Archive digests are rechecked before extracting a bench corpus. PlentyChess
`.data` contains the **processed** net, so compare `processedSha256`, not the raw
training-network digest. Each variant-specific `.data` is checked. Berserk's lock
uses a 12-hex filename prefix; this limitation is recorded (the manifest also
checks the entire preload digest). Embedded-only networks explicitly get
`SKIP`: a WASM artifact digest is not an independent embedded-network digest.
Reckless external prototypes can specify `--nnue <path>`; its digest is checked
before execution against the recorded network. External WASI receives exactly
that provenance-verified file, selected from candidate metadata, never a hardcoded
historical network.

Without `--upstream`, the upstream subcheck is `SKIP` and **no network is used**
(other than an explicitly supplied remote native command). With `--upstream`:

- `git ls-remote --tags` checks the literal locked tag/version against the commit,
  including peeled annotated tags. Missing/mispointed tags are `WARN`, not invented
  release identities. Alternative `v`/`b-v` spellings remain evidence, not a silent
  rewrite of the locked string.
- Highest version/name-sorted tags are reported with their commits. This is not
  chronological tag-date ordering: `ls-remote` does not include dates.
- Shallow-fetch the locked ref into a fresh temporary Git directory and read its
  commit message, including an anchored OpenBench `Bench: N` line when present.
  No existing source checkout or Git configuration is modified.

A network-query error is `FAIL`, not a successful provenance check. Local
provenance/SIMD failures prevent network queries and slow benches.

### 2. SIMD

Imports the shared `scripts/inspect_wasm_simd.mjs` decoder; scans **code**, not
byte patterns in embedded nets/data. Every tier reports total SIMD, total relaxed
SIMD, integer relaxed-dot count and relaxed f32 madd/nmadd/min/max counts. Complete
relaxed opcode counts are in JSON. A tier named `simd`, `sse41` or `relaxed` with
zero SIMD is a **FAIL** (silent scalar trap). Stockfish and Stormphrax also
explicitly require SIMD in their otherwise unnamed baseline flavors. Intentional
scalar/base fallbacks in other engines may legitimately have zero SIMD. Relaxed f32 madd/nmadd in a relaxed tier is **WARN**: a CPU/runtime
may choose fused or unfused results. A single-host exact bench does not establish
cross-platform floating-point equivalence. SIMD presence alone is not semantic
correctness.

### 3. Full bench — mandatory before costly stages

Native reference precedence is **`--native-cmd` → `--native-bench N` → locked
commit `Bench:`**. A supplied command that fails cannot be replaced with a more
convenient numeric reference. Supply a command for the **same source and net**;
its command and reference rows are recorded, but this tool cannot independently
attest the executable's build provenance. No supplied/discoverable native reference
produces `SKIP(native unavailable)` with a build suggestion; nothing is built
automatically. An OS-attested spawn/ENOENT lookup failure also skips. Failures
inside supplied shell/SSH commands remain FAIL: diagnostic text such as "network
not found" is not proof that an executable is unavailable.
Runtime failures, incomplete native output and invalid summaries fail. The final
summary record is selected only after validating every recognized summary's
complete format and numeric tokens on both stdout and stderr; stream ordering
or earlier valid output cannot hide a malformed summary.
Replay likewise validates complete UCI bestmove records and safe-integer node
tokens before accumulating each search, including the final node-bearing line.
Supported
Emscripten benches that time out or omit a valid total also fail; only an explicit
unsupported-command response skips. Replay must exit cleanly, not crash on quit
or require forced termination.

For built-in benches the native command receives `bench` then `quit` over stdin
(works with quoted `ssh ... bash -lc ...` commands). Replay adapters drive the
same suite through native UCI and WASM UCI. Threads=1, Hash=16; replay has **one**
`ucinewgame` before the suite, not one reset per position. Final node totals and ordered per-position nodes/bestmoves must match when the
native reference provides position rows; omitted candidate rows cannot disable
that comparison. Recognized position records are validated completely, with
contiguous coverage and a sum matching the summary; Berserk requires all 50 rows.
`EXACT` is `PASS`; `DIFF(+/-n)` is **FAIL**, even when another narrower suite passes.
A zero total delta with different position signatures still fails.

`--baseline-diverged` changes bench mismatches to **WARN**, retaining the delta,
reference, rows and mismatches. It does not waive provenance, SIMD, runtime or
parity failures. Use only to record a known inherited gap; it is not release
approval. With no reference, the tool still measures all tiers but the comparison
is **SKIP**, never a silent cross-WASM/native pass.

Adapters:

| Engine | Full-bench protocol |
| --- | --- |
| Viridithas | Node `WASI`, preview1, args `['viridithas', 'bench']`, `returnOnExit`; parse `N nodes in ...s`. |
| Reckless | Node WASI built-in bench via stdin `bench`/`quit` with no bench argv, matching native UCI mode and retaining all 50 table rows. Parse table footer or `Bench: N nodes ... nps`. External WASI supported; browser API facades explicitly skip (no `_start`/bench). |
| PlentyChess | Never invoke deadlocking single-thread `bench`. Replay **all 46** `BENCH_POSITIONS` at depth 13, preserving `moves ...` suffixes; enable `UCI_Chess960` immediately before **zero-based index 44**. Sum the last UCI nodes before every bestmove. |
| Stockfish | Reuse `stockfish_upgrade_bench.mjs`: Stockfish.js compiles out bench, so replay **all 51** pinned `benchmark.cpp` Defaults at depth 10, including options/terminal positions. Native built-in depth-10 bench must equal native replay before reference acceptance. Full/lite references are separate: use `--native-lite-cmd` / `--native-lite-bench` for lite. |
| Berserk | `command("bench")` works in shipped tiers; parse `Results: N nodes`, including position signatures. |
| Stormphrax | `command("bench")` works in shipped tiers; parse `N nodes ... nps`. |

Bench sources are read offline from the **checked-in corresponding-source archive
for the locked ref**. `--bench-source <path>` supplies a reviewed source file;
`--upstream` permits fetching it at the locked ref when absent locally. Corpus
shape changes fail closed and require adapter review. No unpinned fetches.

Stockfish commit-message benches use a different default depth; PlentyChess's
built-in bench constructs boards from the FEN and ignores `moves` suffixes.
Therefore those commit signatures are reported as provenance but **not used as
replay references**. Supply same-replay native commands or explicit replay totals.

#### PlentyChess historical oracle discrepancy

The supplied `pcbench/script.txt` from the v8 upgrade has **153 searches**, skips
alternating bench entries (starts at Kiwipete, not startpos), and incorporates
`SPEEDTEST_POSITIONS`. Its **9,558,434** is a matching native/WASM signature for
that historical script, **not** the canonical full 46-position bench replay.
This audit deliberately extracts only `BENCH_POSITIONS`. The v8 locked ref
`50ad9f1b523063cbbbf599fe0547f42870fe2906` gives **3,735,787** for native and all
three WASM tiers, including exact per-position nodes/bestmoves. The native
built-in bench is another protocol (**3,939,470** in the checked Yukon run), not a
substitute reference for the prescribed replay. Do not embed the erroneous
153-search oracle as a version constant.

### 4. Optional rotated parity

Add `--parity` after bench is exact (or explicitly baseline-waived). Wraps the
existing `emscripten_uci_bench.mjs` / `reckless_wasi_node_bench.mjs` at depth 9,
20 rotated FENs, or the existing Stockfish harness with separate full/lite
references. This is **cross-WASM-tier** parity; it does not imply native parity.
Compare bestmove/score/nodes/PV. The wrapper independently validates row coverage,
requested depth, complete fields and Reckless exit codes; matching failed searches
cannot pass via the harness's aggregate totals. Producing harnesses validate raw
UCI numeric tokens, bestmove/ponder records and PV moves before forming rows, so
matching malformed records cannot become exact parity. Canonical and tier-specific
Emscripten preload aliases are preserved by both bench and parity loaders.
Unsupported Reckless API/external variants are
listed as excluded; Viridithas has no existing rotated harness and explicitly
skips. No new parity protocol implementation. Earlier FAILs and unavailable
required full-bench corpora or skipped required bench executions block this stage
with SKIP. Optional browser API facades are excluded from that barrier. A measured
WASM bench without a native reference may still get cross-WASM parity; its native
comparison remains SKIP and is not release approval.

## Output and exit codes

Stdout: human Markdown table. Progress goes to stderr. `--json <path>` writes:

```json
{"engine":"viridithas","ref":"...","gates":[
  {"name":"bench","status":"FAIL","detail":"DIFF(-113) ...",
   "data":{"tier":"scalar","delta":-113,"candidate":{"nodes":3749760},
           "reference":{"nodes":3749873,"source":"--native-bench"}}}
]}
```

Gate names repeat across tiers; use `data.tier` / `data.check` to identify rows.
Statuses are `PASS | WARN | FAIL | SKIP`. Exit **1** for any FAIL, **0** for a
matrix without FAIL (which may still contain SKIPs), **2** for CLI/setup/output
errors. Consumers must inspect SKIPs; exit zero alone is not release approval.
Cheap failures skip slow stages. Bench differences still collect **every tier**
to expose scalar/SIMD splits, then stop before parity. Bench totals intentionally
omit timing claims.

## Upgrade order and Yukon hygiene

1. Prepare same-ref/same-net candidates and metadata; **audit** full bench first.
2. **Parity** across tiers and native on a broader/adversarial corpus.
3. Interleaved **browser A/B**, warm-state, cold-start and transfer-size checks.
4. Only then a resumable paired **Elo** match with predeclared precision/stop rules.

**Stop the upgrade on the first FAIL.** Investigate explicitly documented WARNs
and required SKIPs before proceeding. Passing unit tests or an Elo gain never
waives a failed correctness gate. v1 does not build, compress, play games, publish,
push or deploy.

Yukon native binaries currently live under `~/upgrades/<engine>/`. Inspect paths
read-only, use `bash -lc` and **`taskset -c 3`** for this audit. Never build a missing
native engine automatically; print its build command instead. Never run
`npm install` in an empty scratch directory: it can walk to a home-level package.
For separately authorized package work, assert the package root and use an explicit
`npm --prefix /absolute/package/root ...`. This worktree's `node_modules` is a
symlink: do not install into it.

For later, separately authorized fastchess work, detach with `setsid`/`nohup`,
redirect stdin/stdout/stderr, and `disown` so SSH can exit, for example:

```sh
setsid nohup taskset -c 3 fastchess <reviewed-match-options> \
  >match.log 2>&1 </dev/null &
disown
```

Preserve configurations, seeds, PGNs, paired statistics and crash/time-forfeit
counts. **Never run Brotli/compression concurrently with timing**, builds or Elo
sampling; serialize expensive work and use the project's timing lock for browser
A/B. Audit node counts are correctness evidence, not comparable NPS observations.

Fast tests (no real engine runs): `node --test tests/engine_upgrade_audit.test.mjs`.
Full validation: `npm test` and `npm run typecheck`.
