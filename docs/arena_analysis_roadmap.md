# Arena & Analysis roadmap (EngineBattle gap analysis)

2026-06-10, branch `feature/arena-analysis-ux`. Source comparison:
[lepned/EngineBattle](https://github.com/lepned/EngineBattle) (local clone in
`repos/EngineBattle`), a .NET desktop tournament/testing GUI for native UCI
engines. Different design point — we ship zero-install browser engines — but
its feature set maps the gaps in our arena and analysis pages.

## Gap inventory

| Area | EngineBattle has | We have | Gap severity |
| --- | --- | --- | --- |
| Tournament modes | RR, gauntlet, Swiss, Cup, Ladder; standings; crosstable; Ordo tables; dynamic engine addition | A-vs-B pairs + offline `elo_arena_parallel` script | **High** — our most common question is "is the new build stronger than the pool?" |
| Time controls | Clock + increment, asymmetric per engine, forfeits | Per-move budgets only (fixed visits/depth, equal movetime) | Medium |
| Charts | Eval/time/NPS/NPM per game; live MCTS N/Q plots; PV boards per engine | Text diagnostics, eval bars | **High value, low cost** — engines already report the data; for MCTS plots we hold the real PUCT root (`result.root`), so we can chart true visit/Q distributions instead of parsing UCI lines |
| Game review | Accuracy %, move classes (Brilliant→Blunder), win-prob tracking, critical moves, annotated PGN, batch review | Nothing | High (analysis board) |
| Puzzles/EPD | ERET, Lichess puzzles incl. policy/value head tests, failure visualization | Dev-side test suites only | Medium |
| Adjudication | Tablebase adjudication, reference-PGN consistency, prevent-deviation | Rules-of-chess only | Low for us |
| Chess960 | Play + position generation | Not supported | Low (movegen work) |
| Book tooling | Books from PGN/EPD, out-of-book eval filtering, deviation finder | Opening suites + imported-game stats | Medium |
| Misc | Focus mode, dual-engine compare view, UCI script load, settings page, Winboard | `searchRoot` supports `rootMoves` (no UI); URL params + scattered localStorage | Low each |

## Adoption plan

### Stage 1 — Tournament infrastructure (building now)

- `src/lc0/tournament.ts`: pure scheduling/standings core, Node-tested.
  - Round-robin (all pairs) and gauntlet (seat 1 challenges the field)
    schedules, expanded over openings × games-per-pairing with color
    alternation, emitted as `(whiteId, blackId, opening)` games — a drop-in
    generalization of the existing `MatchGame` loop.
  - Standings: points, W-D-L, per-opponent records; Elo vs pool with ~95%
    error bars (logistic Elo on score fraction, normal approximation —
    Ordo-lite, labeled approximate).
- Arena: seat list generalized from fixed A/B to N seats with add/remove,
  mode select (Match / Round-robin / Gauntlet), standings table in the
  Result section (legacy one-line score remains for two engines).
- Non-goals at this stage: Swiss/Cup/Ladder (RR + gauntlet cover the actual
  benchmarking need), concurrency >1 game (single shared GPU/board), Ordo
  file export.

### Stage 2 — Charts

- Per-game time series under the board: eval, move time, NPS, nodes/move per
  engine, from data already in `lastInfo`/search stats. Plain SVG, no chart
  dependency.
- Live PUCT root chart for LC0-family engines: top-k root moves by visits
  with Q overlay, sourced from `result.root` — richer than UCI-derived
  N-plots. Reuses the search-tree data the arena already records.

### Stage 3 — Game review (analysis board)

- Move-by-move re-evaluation of a loaded game with the selected engine(s):
  win-prob track, accuracy %, move classification by centipawn/win-prob
  loss, critical-move list, annotated PGN export. Batch review later.

### Stage 4 — Smaller items, opportunistic

- Focus mode in analysis (UI for `rootMoves`).
- Lichess puzzle/policy-head test page (fits the eval culture; reuse
  `gameImport` fetch plumbing).
- Out-of-book opening filter for the arena suite (evaluate suite positions
  with one engine, flag |eval| outliers).
- Persistent settings page consolidating URL params + localStorage.

### Explicitly skipped

Winboard protocol, tablebase adjudication, Chess960 (until movegen support
exists), desktop-style multi-process concurrency. Real clock + increment time
controls are deferred until threaded engines land (clock fairness is
meaningless while budgets are the only strength dial that engines respect
uniformly in-browser).
