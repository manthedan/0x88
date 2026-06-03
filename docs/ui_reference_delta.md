# LC0 browser UI: reference comparison and feature deltas

Goal: evolve the single `lc0-policy-only.html` debug page into **three modes** —
(1) single-engine play/test/debug, (2) multi-engine arena, (3) serious
multi-engine analysis board — by closing feature gaps against four references.

## Reference feature inventory

### EngineBattle (lepned) — arena reference
Tournaments: round-robin, gauntlet, cup (knockout), swiss, ladder. Live
standings table, per-game move lists. Time controls (incl. node-limited policy
tests, move delay). Per-engine PV boards with disagreement arrows. MCTS charts
(N-plot, Q-plot), eval chart, NPS/NPM/time charts. Dual-engine side-by-side
analysis. Lichess-style game review (accuracy, move classes). Opening books,
Chess960, PGN reference enforcement, Ordo ratings, puzzle testing, tablebase
adjudication. Engine registry via JSON defs.

### Nibbler (rooklift) — Lc0 GUI reference
Per-move Leela stats N/P/Q/S/U/V/WDL. Winrate graph. Hover a PV to preview it on
the board without changing the analysis position. `searchmoves` to restrict
search to candidate moves. PGN load (menu/clipboard/drag-drop) with
arbitrary-depth variations. FEN load, play from any position, self-play, 960.
Node-limit setting. Arrows for top choices.

### Chessify — cloud analysis reference
Analysis board with eval bar, multi-line (MultiPV) output with scores, move
tree/variations, choice of many engines, opening explorer over a large game DB,
game scan / blunder check, endgame tablebase, save games/analysis, PGN import.

### OpeningTree — game-import + opening study reference
Import games by Lichess/Chess.com username or PGN upload. Opening tree: per move,
frequency + win/draw/loss across the imported set, filterable by color, rating,
time control, ECO. Click through the tree on a board; move list navigation.

## Current state (single page)

`lc0-policy-only.html` + `policyOnlyBrowser.ts`: one chessground board; drag
moves; flip/reset; policy-move / fixed-visit search / analyze / stop / run-parity
buttons; FEN load; side + reply-mode + visits/batch/MultiPV inputs; clear cache;
WebGPU status; debug panel (backend, WDL, Q/MLH, parity, search summary/timing,
PV, flat move list); top legal priors + search children lists; best-move/MultiPV
board arrows; a board-driven "watch LC0 vs policy/Stockfish" battle. MultiPV
search exists. Engines available: LC0 (ONNX, policy + PUCT search) and
Stockfish-18-lite (UCI worker).

## Deltas (what's missing) and target mode

Shared foundations (needed by 1 & 3):
- **Variation tree** model (branching moves), not a flat list. [gameTree.ts]
- **PGN import/export** with variations. [pgn.ts]
- **Game navigation**: first/prev/next/last + keyboard, jump to any node.
- **Eval bar** (vertical) beside the board.
- **Engine registry**: name engines (LC0 policy, LC0 search@N, Stockfish@depth)
  as reusable, selectable participants rather than hardcoded pairs.

Mode 1 — Single engine (play / test / debug): mostly present. Add:
- Position editor (place pieces / set side, castling, ep) — or at least robust
  FEN editing (have). Play-vs-engine with the variation tree recording the game.
- `searchmoves` focus (restrict search/analysis to chosen candidate moves).
- Save the played/analyzed game as PGN.

Mode 2 — Multi-engine arena. Add:
- N-engine participant list from the registry.
- Tournament formats: round-robin + gauntlet (cup/swiss/ladder later).
- Live standings table (W/L/D, score, Elo-ish), per-game result rows, PGN export.
- Watch the current game on a board (recreational), with the option to run
  headless/fast for benching. Per-engine NPS/eval mini-stats.

Mode 3 — Serious analysis board (flagship delta). Add:
- **MultiPV analysis panel**: ranked lines with score + SAN PV, click to enter,
  hover to preview (Nibbler/Chessify staple).
- **Eval bar + eval graph** across the mainline.
- **Variation tree UI**: branch on alternative moves, promote/delete lines.
- **Multiple engines at once**: LC0 + Stockfish analyzing the same position
  side-by-side, with Leela N/P/Q/WDL surfaced.
- **PGN import + game stepping**; load a game and analyze any position.
- **Opening tree from imported games** (OpeningTree-style): paste/upload PGNs,
  build a move-stat tree with W/D/L per move, click to walk it on the board.

## Build order

0. This delta doc.
1. Shared model: `gameTree.ts` (variation tree + nav) and `pgn.ts`
   (parse/serialize with variations), Node-tested.
2. Analysis board page `lc0-analysis.html` (mode 3): eval bar, MultiPV panel,
   variation tree, multi-engine, PGN load. Reuses gameTree/pgn + existing
   evaluator/searcher/Stockfish adapters.
3. Arena page `lc0-arena.html` (mode 2): engine registry, round-robin/gauntlet,
   standings, watch board, PGN export. Reuses engineBattle core.
4. Single-engine page (mode 1): refine the current page; add searchmoves,
   PGN-of-game, position editor; make it the play/debug mode.
5. A small landing/nav linking the three modes.

Each step ships behind its own HTML entry (added to `vite.config.ts`), with pure
logic unit-tested and pages verified via typecheck + build.
