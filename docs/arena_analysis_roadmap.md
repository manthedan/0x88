# Arena & Analysis capability map and backlog

Current source check: 2026-11-12. This is the delivered/backlog map for
`/app/arena/`, `/app/analysis/`, and `/app/play/`, not a list of missing June
features. The original design comparison is retained below.

## Current capabilities

| Area | Delivered | Remaining boundary |
| --- | --- | --- |
| Tournament modes | Match, round-robin, gauntlet, N seats, W-D-L standings, approximate Elo/error bars, per-opponent crosstable | Swiss, Cup/knockout, Ladder, byes, seeded brackets and format-specific tiebreaks |
| Budgets | Fixed visits/depth or equal movetime; warmup hooks | Real clocks/increments, asymmetric time odds, move-overhead/minimum-time controls |
| Parallel games | Bounded private CPU-only games, memory admission, conservative desktop Auto K=2 | Neural/threaded-engine schedules remain serial; not shared-tree GPU concurrency |
| Openings | Built-in/custom FEN, UCI and PGN/SAN replay with real LC0 history; color alternation and opening-pair results | Full book tooling, seeded random order, out-of-book filtering, reference-line enforcement |
| Persistence | Completed games saved locally; compatible tournament resume and PGN export | Not a Swiss/Cup/Ladder bracket persistence system |
| Charts | Eval, move time, NPS/nodes series and live PUCT root visit/Q detail | Desktop-style per-engine PV boards and configurable streamer layouts |
| Review | Mainline re-evaluation, approximate per-side accuracy, move classes, critical moves, annotated PGN | Batch review and Brilliant/Great sacrifice detection |
| Analysis/import | Variation tree, PGN/FEN load/export, MultiPV, PV preview, multi-engine compare, opening explorer, Lichess/Chess.com username import | Restricted-root-move UI, UCI script loading, consolidated settings |
| Rules/protocols | Normal chess, browser engine adapters and chess-rule outcomes | Chess960, Winboard/XBoard, arbitrary native executables, Syzygy/eval-streak adjudication |

Source boundaries: `src/lc0/tournament.ts`, `src/lc0/arenaBrowser.ts`,
`src/lc0/arenaOpenings.ts`, `src/lc0/gameReview.ts`, `src/lc0/charts.ts`,
`src/lc0/gameImport.ts`, and `src/lib/nav.ts`. Elo and review accuracy are
approximations, not Ordo or full Lichess review equivalence.

## Delivered stages (original design: 2026-06-10)

### Stage 1 — Tournament infrastructure (delivered)

`src/lc0/tournament.ts` generalizes A-vs-B into round-robin/gauntlet pairings,
expanded over openings and games-per-opening with alternating colors. Arena
has add/remove seats, mode selection, standings and per-game results. The
Elo-vs-pool estimate uses logistic score fraction and normal-approximation
~95% error bars; it is labeled approximate.

The original single-game concurrency boundary changed in October. Eligible
CPU schedules now create two private single-threaded engines per active game,
reset both at game boundaries, and dispose them on completion/cancellation.
See [`arena_parallel_games_2026-10.md`](./arena_parallel_games_2026-10.md) for
K admission, memory limits, measurements and the changed historical TT-state
semantics. Fixed movetime is not deterministic under concurrency. The neural
simul feasibility record is a separate, non-shipping experiment:
[`arena_simul_gate_2026-11.md`](./arena_simul_gate_2026-11.md).

### Stage 2 — Charts (delivered)

`src/lc0/charts.ts` provides dependency-free SVG time series and root-visit
bars. Arena/Analysis use reported eval, time, NPS/nodes and real PUCT root
statistics, rather than inferring visits from UCI text. This is no longer a
proposal to add all charts.

### Stage 3 — Game review (delivered; batch review remains backlog)

`src/lc0/gameReview.ts` computes move classes, win-probability losses,
approximate per-side accuracy and critical moves, and produces annotated PGN.
Analysis re-evaluates loaded mainlines. Brilliant/Great sacrifice detection is
intentionally absent. The per-move curve follows the published Lichess formula;
the game aggregate is a per-side mean, not an identical review pipeline.

### Stage 4 — Remaining product backlog

- Restricted candidate-move/focus UI (`rootMoves` exists in search).
- Puzzle/policy/value-head testing UI: top-N, KLD and rating/theme groups;
  ERET visualization. Dev-side suites are not this product surface.
- Full opening-book import/tooling, book-exit evaluation/filtering and
  reference-PGN deviation control. Existing line replay and color pairs are
  delivered, not prerequisites still to build.
- Seeded random opening order, additional tournament formats and Ordo export.
- Batch game review, per-engine PV boards, configurable layouts and a
  consolidated settings page.

Real clock + increment controls remain deferred; equal per-move budgets are
already available. Winboard, tablebases, Chess960 and desktop-native process
management are not current browser milestones. The old threading dependency
was a planning assumption, not proof that clocks require another threaded port.

## Equal movetime: current behavior and retained design rationale

The earlier time-parity proposal is delivered. Arena defaults to equal
movetime (500 ms in the current page) and can switch to fixed visits/depth.
Equal time compares resources, not strength: CPU and neural engines can gain
different amounts from the same budget. Per-engine time multipliers remain a
possible extension, not a shipped balancing feature.

`src/lc0/arenaBrowser.ts` reads the budget in each move wrapper. Timed Lc0
searches pass `movetimeMs` with no fixed visit limit; fixed searches pass the
seat's visits. CPU engines use UCI `go movetime N` or fixed depth. Switching
back must clear a previously merged timed option:

```ts
engine.setOptions({ depth, movetimeMs: undefined });
```

The suggested `ArenaMoveContext` interface was not required; wrappers still
receive history and an `AbortSignal` and read the selected budget.

Time expiry is a **soft stop**, not cancellation. Search checks deadlines
between work chunks and returns best-so-far. `AbortSignal` means the caller
cancelled the run; it must not be used to turn a timed result into an error.
Requested time, actual elapsed time, visits, eval calls and stop reason are
needed when interpreting timed results. One inference/chunk can overrun the
requested deadline; this is not a hard real-time clock.

Warmup hooks load/initialize participating engines before timed play. The
original rationale still matters: first-move model/WASM/WebGPU startup is not
steady-state search speed. Lc0 uses a tiny search; CPU warmups initialize and
run a shallow search. Keep startup/readiness separate from timed measurements.
Policy-only inference is not a time-searchable opponent just because a budget
control exists.

## Historical reference inventory and UI milestones

The 2026-06-10 comparison used
[EngineBattle](https://github.com/lepned/EngineBattle), a .NET desktop GUI
(local clone `repos/EngineBattle`). Its native-process feature set was a
reference, not a browser compatibility promise. The initial UI comparison also
used [Nibbler](https://github.com/rooklift/nibbler),
[Chessify](https://chessify.me/), and [OpeningTree](https://www.openingtree.com/).

The reference inventory covered:

- EngineBattle: RR/gauntlet/Swiss/Cup/Ladder, clocks, standings/Ordo, PV boards,
  N/Q and eval/NPS/time charts, review, books, Chess960, puzzle tests,
  tablebases and JSON engine definitions.
- Nibbler: Leela N/P/Q/S/U/V/WDL, winrate graph, PV hover, restricted moves,
  variation PGN/FEN loading, self-play, node limits and arrows.
- Chessify: multi-engine MultiPV, eval bars, variation trees, opening DB,
  game scan, tablebases and saved/imported games.
- OpeningTree: username/PGN import, opening frequency and W/D/L, filters and
  board navigation. Username import is now implemented here.

The early three-mode milestone reported browser verification of the shared
`gameTree.ts` and `chess/pgn.ts`, Analysis variation navigation/MultiPV/opening
explorer, Arena round-robin/gauntlet/standings/PGN export, and shared nav. It
reported 20 pure-logic tests. Those checks used the old HTML entries and are
historical evidence, not current run instructions. The workspaces now use
Svelte routes. Review, charts, compare and username import arrived later;
the old single-page inventory and build checklist are obsolete.

### Retained EngineBattle reference links

These document the original comparison targets, not newly verified upstream
feature claims:

- [Overview](https://github.com/lepned/EngineBattle/blob/main/README.md)
- [Swiss](https://github.com/lepned/EngineBattle/blob/main/SwissMode.md),
  [Cup](https://github.com/lepned/EngineBattle/blob/main/CupMode.md),
  [Ladder](https://github.com/lepned/EngineBattle/blob/main/LadderMode.md)
- [Tournament configuration](https://github.com/lepned/EngineBattle/blob/main/TournamentConfig.md)
- [Engine definitions](https://github.com/lepned/EngineBattle/blob/main/EngineDefConfig.md)
- [Book evaluation](https://github.com/lepned/EngineBattle/blob/main/BookEvaluation.md)
- [Game review](https://github.com/lepned/EngineBattle/blob/main/GameReview.md)
- [Analysis](https://github.com/lepned/EngineBattle/blob/main/AnalyzeConfig.md)
- [Puzzle tests](https://github.com/lepned/EngineBattle/blob/main/PuzzleConfig.md),
  [ERET](https://github.com/lepned/EngineBattle/blob/main/EretConfig.md)
- [Console tools](https://github.com/lepned/EngineBattle/blob/main/ConsoleTools.md)
  (including native runners, tuner, PGN summaries, Elo/speed stats and perft;
  no claim that this browser exposes that complete suite).
