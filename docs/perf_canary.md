# Performance canary

`npm run perf:canary` is a local tripwire for CPU-engine speed. It is not a CI
gate and it does not promote a runtime. Hardware noise on a shared machine is
wider than the regressions this repo cares about, so the script writes JSON,
diffs it against the previous run, and exits 0 unless the bench itself fails.
Pass `--strict` only when you explicitly want a non-zero exit on a regression
call.

## What one run does

The default CPU leg calls `scripts/emscripten_uci_bench.mjs` on the scalar
Berserk wasm build (`public/berserk/berserk-emscripten.js`). The suite is the
existing 20-position rotated FEN list, at fixed depth 8, 1 thread, and 16 MiB
hash. The canary repeats that pass three times and keeps the median and the
median absolute deviation (MAD) of engine-reported NPS and engine-reported
time. Node wall time is not the comparison metric: process startup and the
copy of the Emscripten data file sit outside the engine's own clock.

`--browser` also runs `scripts/lc0_browser_hybrid_input_bench.mjs` with the
wasm input backend, one iteration, on port 5188. That leg needs a working
browser and a GPU. Leave it off for the CPU canary, including on Yukon. The
default JSON records the browser leg as skipped.

Results land in `.local-dev-artifacts/perf-canary/` (gitignored):

- `canary-<timestamp>.json` — one run, schema `browser-engine.benchmark-artifact.v1`
- `latest.json` — a copy of the newest run

The next run diffs its medians against the newest stamped `canary-<UTC>.json`
whose host, artifact, depth, position count, and hash match this invocation.
A different machine or a different suite starts a new baseline instead of a
false regression. `latest.json` and names that are not a full UTC stamp are
ignored. A swing is called `regressed` or `improved` only when it is larger
than both 10% and three times this run's MAD/median. A run whose own MAD/median exceeds 10% is
`noisy-run` and is not treated as a regression. Fewer than three repeats is
`insufficient-repeats`.

## How to run it here

Hold the shared timing lock. Do not build while you hold it.

```sh
/path/to/scratchpad/benchlock npm run perf:canary
```

Run it twice. Read the second invocation's `npsMedian` and
`engineTimeMsMedian` lines. Those are the same-machine run-to-run deltas.

## When the machine is too loud

If the same-machine run-to-run spread on NPS or engine time is above about
10%, this host is not a useful canary. Take the CPU leg to Yukon instead of
widening the threshold or gating CI.

Yukon is the `yukon` ssh alias (Ryzen 12C). Use your own directory and do not
run `npm install` in `$HOME`. The CPU canary only needs Node and the committed
Berserk js/wasm/data files; it does not need `node_modules`.

```sh
rsync -a --exclude node_modules --exclude .git --exclude dist-client \
  --exclude .local-dev-artifacts \
  ./ yukon:~/agents/chore-test-infra-perf-baseline/
ssh yukon "bash -lc 'cd ~/agents/chore-test-infra-perf-baseline && taskset -c 2 node scripts/perf_canary.mjs'"
```

`~/tools/chess/AGENTS.md` assigns physical-core slots to upgrade agents
(stockfish 0–2, plentychess 3–5, viridithas 6–8, reckless 9–11). SMT siblings
12–23 are for builds. Pin the canary to one idle physical core in a slot
nobody else is using. `taskset -c 2` is the example above, inside the
stockfish slot; pick another core if that slot is busy. Threads stay 1 inside
the bench.

## Measurements

Recorded 2026-10-10 with the default CPU canary: Berserk scalar wasm, depth 8,
20 rotated FENs, 3 repeats, 16 MiB hash, 1 thread. The browser leg was off.
`engine time` below is the median of per-position engine-reported times inside
one pass, then the median of those three pass medians. That median sits on a
1 ms grid near 24 ms, so NPS is the sensitive signal. Node counts were
identical across repeats.

### Yukon (quiet machine)

Two back-to-back runs on the Ryzen 9 3900XT, Node v26.2.0, pinned with
`taskset -c 2` (physical core in the stockfish slot; that core was idle).
Wall clock was 3.68 s and 3.61 s (`/usr/bin/time -p` real).

| Run | NPS median | NPS MAD | NPS samples | Engine-time median (ms) | Engine-time samples | Nodes |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 189185.5 | 780.5 (0.4%) | 187323, 189185.5, 189966 | 24 | 25, 24, 24 | 89193 |
| 2 | 191057 | 0 (0%) | 189719, 191057, 191057 | 24 | 24.5, 24, 24 | 89193 |

Run-to-run NPS delta was +1.0%, inside the 10% floor (`within-noise`).
Engine-time median delta was 0. Yukon is quiet enough for this canary.

### This Mac, under benchlock

Two back-to-back runs on the Apple M4 (darwin 25.4.0, Node v25.9.0) while
`benchlock` was held, at 01:59 local on 2026-10-11. Wall clock was 2.30 s and
2.28 s. Nothing else was in the timing lock.

| Run | NPS median | NPS MAD | NPS samples | Engine-time median (ms) | Engine-time samples | Nodes |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 413157 | 0 (0%) | 415829, 413157, 413157 | 9.5 | 9, 9.5, 9.5 | 89193 |
| 2 | 415829 | 2672 (0.6%) | 423350, 415829, 413157 | 9.5 | 9.5, 9.5, 9.5 | 89193 |

Run-to-run NPS delta was +0.6%, inside the 10% floor (`within-noise`).
Engine-time median delta was 0. On a quiet hold of the lock this Mac is usable.

Do not compare the M4 medians to the Yukon medians. Both hosts searched the
same 89193 nodes; the NPS gap is the machine. Diff a run only against the
previous JSON on the same host. Earlier the same evening this Mac's load
average was in the teens and thirties while other agents worked, which is the
case the 10% rule is for: if a later pair spreads by more than about 10%,
move the CPU leg to Yukon instead of widening the threshold.
