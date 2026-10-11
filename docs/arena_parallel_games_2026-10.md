# Parallel CPU Arena games — October 2026

## Verdict

**Ship K=2 as the conservative desktop Auto setting.** All three gates passed on this 10-core Apple M4: K=2 retained **95.1% Stockfish / 91.9% Berserk** per-engine throughput, deterministic games agreed, and the built-app round robin saved **49.2% median wall time**. K=3 remains explicit opt-in, not the default.

Base: `5c161a9`; branch: `perf/arena-parallel-games`. Compact numerical evidence: [`evidence/arena_parallel_games_2026-10.json`](evidence/arena_parallel_games_2026-10.json).

## Source checks and implementation

The schedule loop in `src/lc0/arenaBrowser.ts` was serial. Its page broker was exclusive, **but only Stockfish's move wrapper actually acquired that broker**. `src/lc0/resourceBroker.ts` already had a bounded policy; no broker or engine-adapter modification was necessary. The consultant's old “reset every constructed family” claim is no longer true at this base: the existing loop already resets only participating Viridithas/Berserk/PlentyChess/Stormphrax engines.

- `src/lc0/arenaScheduleRunner.ts`: bounded entry dispatch, distinct child AbortSignals, fail-fast sibling cancellation **with settlement/cleanup before returning**, CPU-only allowlist and conservative admission.
- `src/lc0/arenaCpuGame.ts`: game-local board, history and tree; leased CPU searches; fresh-game reset for both private seats.
- `src/lc0/arenaBrowser.ts`: each scheduled CPU game constructs **two separate engine instances**, even for mirror seats; namespaced leases (`arena:<schedule index>:<color>:<pid>`) register with maxThreads=1. Instances are disposed in `finally`; route/page teardown also disposes the live set. Old surface pools are disposed before the private workers start. At most 2K workers are resident, rather than one worker per tournament participant per lane.
- Finished games update the existing standings reducer immediately. PGN records sort by schedule index, not completion order; cancelled partial games do not count or export. Finished trails retain their schedule indices. The single board/charts follow the earliest ready active game; background games retain UCI telemetry without repainting their outputs onto that board.
- `src/routes/app/arena/+page.svelte`: visible **Concurrent CPU games** selector, explanatory effective-K note; URL override `arenaConcurrency=auto|1|2|3`.

All participating seats must be in the explicit CPU family allowlist. LC0 (including small and big nets), Centipawn, Maia/unknown future families always force K=1, regardless of a CPU/WASM inference fallback. Threaded Stockfish remains on the original serial path. Its existing thread setting is not silently changed.

Admission: unknown/nonfinite memory or `deviceMemory <= 4` gives K=1, even with an override. Auto needs at least eight logical cores and memory above 4 GiB to select K=2. Explicit requests are capped at `floor((cores - 2)/2)` and at three; a memory hint below 16 GiB caps them at two. K=3 therefore requires both adequate cores and a >=16 GiB hint. Safari/other browsers without deviceMemory stay serial.

### Deterministic state boundary (intentional decision)

The private-worker CPU path applies to **K=1 as well as K>1**. Both seats receive `ucinewgame`, and no TT/search state crosses game boundaries. Previously Stockfish/Reckless could retain state across games, while other families reset; mirror seats could also share one worker. Historical tournaments using that asymmetric state are **not** promised unchanged. The parity claim is K=1 versus K=2/3 with the same private-seat/fresh-game semantics, not preservation of historical TT-dependent games. Neural/threaded pools retain the old path.

Fixed depth and fixed nodes are deterministic in the measured **Threads=1, Skill=20, same build/hash/opening/settings, fresh-game state** envelope. PGN moves/results and, in the initial lab, per-move nodes/depth/score/PV all agreed. Fixed movetime/clock limits are **not deterministic**: concurrency changes the number of nodes searched and may change moves/results. Multi-threaded search, handicapped random skill selection and neural timed searches are outside this parity claim.

## Cheapest throughput gate, before product changes

Built research harness: `lab/arena-parallel-games.html` / `.mjs`, using the existing adapters/artifacts. Separate Stockfish Lite relaxed and Berserk relaxed workers, Threads=1, Hash=64 MiB each. Each trial played six CPU-vs-CPU games, alternating two opening positions (three of each), eight plies per game, 500 ms per move. Startup and readiness warmup excluded; `ucinewgame` precedes each game.

Order **1,2,1,3**, repeated three times under the shared `benchlock`. Metric is nodes at the requested fixed movetime, divided by requested time; each trial averages all six games. Thus the solo baseline has six observations and K=2/3 each have three. An earlier unmatched-opening exploratory run was not used for the gate.

| Engine | K | Median nodes/s | Trial min–max | Fraction of solo |
|---|---:|---:|---:|---:|
| Stockfish | 1 | 1,373,561 | 1,342,519–1,386,116 | 100% |
| Stockfish | 2 | 1,305,695 | 1,274,538–1,334,447 | **95.1%** |
| Stockfish | 3 | 1,180,684 | 1,162,155–1,187,975 | 86.0% |
| Berserk | 1 | 1,142,876 | 1,128,362–1,184,216 | 100% |
| Berserk | 2 | 1,050,166 | 1,024,332–1,059,578 | **91.9%** |
| Berserk | 3 | 951,316 | 887,993–1,035,112 | 83.2% |

Neither K=2 engine approached the <75% kill threshold. K=3's wider Berserk spread and additional memory argue against making three the default.

Initial deterministic game gate: six full games per trial (300-ply draw cap), two openings, order 1,2,1,3 for each of depth=6 and nodes=10,000. All 24 games per limit, including the reference games, had identical complete traces and outcomes by schedule index. Nodes are a **lab-only UCI transport substitution**: the adapters/product UI expose depth/movetime, so the harness substitutes `go nodes 10000` at Worker.postMessage. No product nodes budget was added.

## Memory

`performance.measureUserAgentSpecificMemory()` in the isolated built harness, warmed private Stockfish/Berserk pairs, one GC-backed sample per K:

| K | Live workers | Total reported memory | Increment vs prior K | TT payload across seats |
|---|---:|---:|---:|---:|
| 1 | 2 | 412.19 MiB | — | 128 MiB |
| 2 | 4 | 821.55 MiB | 409.36 MiB | 256 MiB |
| 3 | 6 | 1,230.90 MiB | 409.35 MiB | 384 MiB |

These are browser-attributed JS/WASM allocations, **not measured process RSS or physical committed RAM**; TT payload is included, not additional to the total. Keep Hash=64 MiB at every K: shrinking it only for concurrent games would invalidate deterministic-limit parity. The machine has 24 GiB physical RAM; Chrome reports deviceMemory=16. Full Stockfish and Stormphrax have larger nets/heaps and their per-K footprint was not measured here. The low/unknown-memory gate, >=16 hint for K=3 and two-workers-per-active-game residency bound are intentional safeguards, not a claim of mobile validation.

## Built-app verification

`BUILD_SCOPE=product npm run build:client`, served with COOP/COEP on assigned port 5181; driven with agent-browser (Headless Chrome 149). The small round robin used **Stockfish Lite relaxed, Berserk relaxed, PlentyChess SSE4.1**, each depth=10, Threads=1, Hash=64 MiB, start position plus `1. e4 c6 2. d4 d5`, one game/opening/pairing, six games, delay=0.

Warm-cache button-to-finished wall clock includes **new worker initialization, search, rendering and disposal**. Three interleaved AB pairs under benchlock:

| K | Runs (s) | Median (s) | Min–max (s) | Peak / final workers |
|---|---|---:|---|---|
| 1 | 12.550, 12.533, 12.484 | **12.533** | 12.484–12.550 | 2 / 0 |
| 2 | 6.346, 6.364, 6.438 | **6.364** | 6.346–6.438 | 4 / 0 |

**49.2% median saving**, above the 30% gate, on the final reviewed-fix build. Before review fixes, three AB pairs gave 13.136 / 6.612 s (49.7%); initial separate autoruns also agreed (13.509 / 6.669 s). These earlier samples are retained in the evidence but not mixed into the final medians. All exports matched byte-for-byte, including order and opening replay plies. Fixed-depth PGN SHA-256: `9cdf94d55864bc6c290854063deb36eb470b3131951c3122b2e2fa6dfb55474b`.

Standings for both K: PlentyChess W2 D2 L0, 3/4 points; Berserk W1 D2 L1, 2/4; Stockfish W0 D2 L2, 1/4. Scores, ordering and displayed Elo/error text matched.

Built fixed-node round robin: `go nodes 10000` substituted by the research browser init script, K sequence 1,2,1,2. All four six-game exports and standings agreed; final worker count was zero. After the review fixes, both K=1 and K=2 were repeated and matched that same export/standings. PGN SHA-256: `4490ea0f748e874d4970fe8825e1b9e5279fcb9887fc32557d258aab50a590d5`.

Stop test: 60-game schedule, K=2, 1,000 ms/move; stopped with two games running and four live workers. Both child searches settled, no partial games counted, all four workers terminated, Start re-enabled, `Stopped after 0 game(s).` (observed warm stop-to-ready 11.5 ms; a single functionality check, **not** a latency benchmark). Auto displayed effective K=2 on this machine.

Timing runs held benchlock in <20-minute chunks; no builds were run while holding it. Other agents' timing loads were serialized. The desktop was not an OS-level clean room: load average roughly 3, modest Brave/WindowServer/VM background activity, no sustained external engine/GPU load observed. Final built wall-clock spread was under 1.5%; no core pinning or thermal/device-portability claim.

Screenshots (opened/visually checked), under the delegated scratchpad root `/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad/`:

- `arena-parallel-games-gate.png`
- `arena-parallel-games-built-K2-standings.png`
- `arena-parallel-games-built-stop.png`
- `arena-parallel-games-built-live-replay.png` (final fix build, two active games and a growing review trail)

Raw traces/PGNs and results there: `arena-parallel-games-throughput-matched.json`, `arena-parallel-games-parity-summary.json`, `arena-parallel-games-memory.json`, `arena-parallel-games-built-depth-K1-1.json`, `arena-parallel-games-built-depth-K2-1.json`, `arena-parallel-games-built-depth-repeats.json`, `arena-parallel-games-built-nodes.json`, `arena-parallel-games-stop.json`, `arena-parallel-games-built-depth-final.json`, `arena-parallel-games-built-nodes-final.json`, `arena-parallel-games-prep-stop.json`, `arena-parallel-games-replay.json`. The compact committed evidence is sufficient to recompute gate medians/spreads; raw evidence includes the full exports.

## Review-driven regression checks

The first Codex review identified two valid integration bugs. Both were fixed before final remeasurement:

1. **Preparation cancellation:** abort now immediately disposes both private seats, including during readiness (Stockfish's initialization itself does not observe its signal). The listener is removed on settlement, and ordinary `finally` cleanup remains. Built fault injection dropped the Stockfish `uci` handshake: two private workers stayed unready, Stop terminated both and re-enabled Start in 12.8 ms rather than awaiting the adapter's 120-second initialization timeout. This is a single functional check, not a stop-latency benchmark.
2. **Live replay identity:** one trail object and entry array are maintained per game, appended in place and passed unchanged to the finished-game log. Built K=2 check entered review at move 1/4; while the game advanced, the same review reached move 1/11 and Last jumped to its current end. Four workers were running; stopping then left zero.

Final depth/node exports matched the pre-fix golden exports. Review artifacts and verdict are recorded in the delegated final report.

## Verification and remaining scope

- Final `npm test`: **1,011 pass, 3 expected environment-dependent skips, zero failures** (1,014 total); included a clean typecheck.
- Nine focused new tests: CPU/memory admission, out-of-order completion/standings/export equivalence, individual child signals and cancellation, sibling failure/drain, pre-aborted runs, private board/reset/lease accounting, search-failure lease release, abort during signal-ignoring readiness, and stable replay trail/entry-array identity.
- Existing tournament and resource-broker suites pass. Biome has no new issues; pre-existing arena/Svelte unused-variable warnings remain untouched.
- No GPU parallelism, SMP, adapter cancellation redesign, individual-game Stop UI or TT resizing. K=3 product round-robin wall time, full-net/Stormphrax memory and real constrained/mobile hardware were not measured. K=3 is explicitly gated and opt-in; fixed-node support remains research-only.
