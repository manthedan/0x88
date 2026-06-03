# EngineBattle Feature Gap Notes

Reference project: <https://github.com/lepned/EngineBattle>

These notes compare EngineBattle's documented features with the current LC0 browser arena/analysis UI.

## Tournament features we lack

- **More tournament modes**: Cup/knockout, Swiss, Ladder, byes, tiebreaks, seeded brackets. We currently have round-robin + gauntlet only.
  - Sources: <https://github.com/lepned/EngineBattle/blob/main/README.md>, <https://github.com/lepned/EngineBattle/blob/main/SwissMode.md>, <https://github.com/lepned/EngineBattle/blob/main/CupMode.md>, <https://github.com/lepned/EngineBattle/blob/main/LadderMode.md>

- **Time controls**: EngineBattle supports time, increment, nodes, asymmetric per-engine configs, move overhead, and minimum move time. We mostly do fixed visits for LC0 and fixed depth for Stockfish lite.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/TournamentConfig.md>

- **External engine support**: UCI + Winboard/XBoard engine definition files, engine paths, logos, ratings, startup args/options. We currently have in-browser LC0 ONNX + Stockfish lite only.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/EngineDefConfig.md>

- **Pondering / engine lifecycle / startup timeouts / parallel console games**: EngineBattle handles more of the operational machinery needed for serious engine testing.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/TournamentConfig.md>

- **Adjudication**: draw/win adjudication by eval streaks and Syzygy tablebase adjudication. We only have chess-rule outcomes and draw rules.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/TournamentConfig.md>

- **Tournament persistence/resume**: Swiss/Cup/Ladder state JSON, resume dialogs, and bracket state. Our tournaments are ephemeral browser sessions.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/SwissMode.md>

## Opening/book features we lack

- **PGN/EPD opening books** with opening ply selection, randomized openings, stable seeds, and “play opening twice” color-swaps. We have built-in/custom FEN suites, but not PGN/EPD book import for arena yet.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/TournamentConfig.md>

- **Reference PGN / prevent move deviation**: EngineBattle can keep engines on a reference/opening line even through transpositions. We do not have this yet.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/README.md>

- **Out-of-book evaluation/filtering**: evaluate book exits with engines and filter positions that are too drawish or too one-sided. This would be highly relevant for generating better arena starts.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/BookEvaluation.md>

## Analysis/review features we lack

- **Game review**: Lichess-style accuracy, ACPL, move classifications, win-probability chart, critical move list, and annotated PGN export.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/GameReview.md>

- **Full dual-engine analysis UI** with charts and external engines. We have LC0/SF analysis lines and opening stats, but not the same depth of engine telemetry.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/AnalyzeConfig.md>

- **Focus mode / restricted candidate moves** and UCI script loading for engine setup.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/README.md>

## Testing / benchmarking tools we lack

- **Lichess puzzle testing** across engines/networks: policy, value, search, top-N accuracy, KLD, rating/theme grouping.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/PuzzleConfig.md>

- **ERET test suites**.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/EretConfig.md>

- **Console tooling**: tournament runner, puzzle runner, ERET, analyze, compare, benchmark, tuner, PGN summary, Elo, speed stats, and perft.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/ConsoleTools.md>

- **Bayesian parameter tuner** for engine settings.
  - Source: <https://github.com/lepned/EngineBattle/blob/main/ConsoleTools.md>

## Visualization/statistics we lack

- **Live MCTS charts**: N-plot/Q-plot for LC0/Ceres-style engines.
- **Eval, NPS, NPM, and time-usage charts** over games/tournaments.
- **PV boards per engine** with disagreement arrows.
- **Crosstables, streamer layouts, auto-cycling views, logos/themes/layout config**.
  - Sources: <https://github.com/lepned/EngineBattle/blob/main/README.md>, <https://github.com/lepned/EngineBattle/blob/main/TournamentConfig.md>

## Most relevant next features for us

Given our current direction, the best candidates are:

1. **PGN/UCI opening replay with true LC0 history planes**.
2. **EPD/PGN opening suite import for arena**.
3. **Opening-pair scheduling**: same opening twice with colors swapped.
4. **Randomized opening order + seed**.
5. **Simple crosstable**.
6. **Game review-lite**: blunder/inaccuracy classification from our analysis mode.
7. **Out-of-book eval filter** for generating better arena starts.
