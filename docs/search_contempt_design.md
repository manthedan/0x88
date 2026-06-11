# Contempt in the LC0 browser PUCT search

Status, 2026-06-11: both mechanisms shipped (`src/search/puct.ts`), wired
through the search worker to every Lc0-family opponent on the Play page,
and validated against Maia as a human stand-in. Inspired by the lc0
search-contempt fork used by the Leela odds bots and by Monty's calibrated
contempt — patterns only, no code from either (both AGPL; so are we not,
hence the clean-room rule).

## Why

Engines tuned for self-play assume a perfect opponent: they steer toward
positions that are best *if the opponent always finds the refutation*, and
they treat draws as half a point regardless of who is playing. Against a
human, both assumptions waste winning chances. The Lichess Leela odds bots
(queen/knight/rook odds) beat strong humans from lost material positions
by dropping both assumptions.

## Mechanism 1: WDL draw contempt (`drawScore`)

`SearchOptions.drawScore` ∈ [-1, 1] is the value of a draw **for the root
side**; the opponent is assumed to value draws oppositely (zero-sum).

- Neural leaves: `q' = clamp(q + drawScore_side · d)` where `d` is the WDL
  head's draw mass and `drawScore_side` flips sign at opponent-to-move
  nodes (`valueFromEvaluation`).
- Terminal draws (stalemate, threefold, fifty-move, insufficient material,
  at any node including the root) score `contemptDrawValue` instead of 0.
  Checkmates are untouched.
- Side parity is **absolute** (`board.turn` vs `rootSideToMove`), so node
  values stay consistent under tree reuse between the engine's moves.

Negative values make the engine avoid draws and press for wins.

## Mechanism 2: search contempt (`searchContemptLimit`)

Models the opponent as a **budget-limited searcher** (the ScLimit idea
behind the odds bots). Opponent nodes explore normally for their first N
visits; past N the node's child-visit distribution is frozen
(`Node.scFrozenWeights`) and further descents **sample** from it instead
of taking the PUCT max. Deep refutations the modeled opponent "wouldn't
find" keep their minority share of the frozen mixture instead of taking
over the backed-up value — so trappy lines stay attractive.

Single hook in `selectBestEdge` covers the sequential and batched search
paths. `stats.scFrozenNodes` / `stats.scSampledSelections` expose the
activity. `tests/puct_search_contempt.test.mjs` proves the mechanism with
a synthetic trap (one refutation among 20 uniform replies: plain search
refutes the trap move, scLimit keeps it alive).

## Tree reuse safety

Backed-up node values depend on the contempt settings that produced them,
so a tree built under one setting must not seed a search under another.
`searchValueKey(context)` fingerprints `drawScore`/`searchContemptLimit`/
`rootSideToMove`; root reuse and transposition-table hits require a
matching key (`nodeSearchValueCompatible`), with pre-key trees compatible
only with the neutral default. Covered by
`tests/puct_contempt.test.mjs` ("changed contempt settings do not reuse
stale search tree").

Both mechanisms off (`0`) ⇒ bit-identical search (puct/lc0 parity suites).

## Empirical results (Maia as the human stand-in)

`scripts/contempt_vs_maia.mjs` plays Lc0 small (t1, fixed visits) against
a Maia net sampling from its human-move distribution — the same opponent
model the Play page ships. Arms are `--draw-scores` zipped with
`--sc-limits`. Queen odds (the Lc0 side starts without its queen) vs
sampled Maia 1900:

| Arm (12 games, v16) | Score | Record | Note |
| --- | ---: | --- | --- |
| baseline | 58% | +6 =2 −4 | |
| **scLimit 16** | **92%** | **+11 =0 −1** | all games decisive |
| ds −0.25 + scLimit 16 | 63% | +6 =3 −3 | drawScore drags it back |

Earlier drawScore-only runs (8–16 games, v8–16): ds −0.4 scored 50% vs
69% baseline; ds −0.25 scored 56% vs 59% — at small visit budgets with a
*standard* net, draw contempt mostly converts losses into draw "saves"
(stalemate tricks included) and adds noise, because t1's value head
correctly believes a queen-down position is lost and drawScore cannot
overrule it.

**Guidance**
- `searchContemptLimit` is the big lever for beating humans from worse
  positions; it works *with* a pessimistic value head, not against it.
- `drawScore` belongs where the engine is equal-or-better (normal play vs
  a human) or where the net's head is odds-calibrated (LeelaQueenOdds was
  trained to evaluate queen odds as equal). Keep magnitudes small at
  small visit counts.
- Settings are net- and scenario-dependent: re-run the harness when
  changing nets or ladders.

## Current Play-page settings (`src/lc0/playBrowser.ts`)

| Opponent | drawScore | scLimit | cpuct |
| --- | ---: | ---: | ---: |
| Lc0 small / t3 / BT4 | −0.25 | — | default |
| Leela Queen Odds | −0.5 | 24 | 1.5 |
| Maia (human model) | — | — | — |

LQO mirrors its README (DrawScore ±0.4–0.6, CPuct 1.5, ScLimit 32–40 at
12–15k nodes) scaled to browser visit budgets; unlike t1, its value head
is odds-calibrated, which is why drawScore stays on for it.

## Roadmap: Monty-style calibrated contempt

[Monty](https://github.com/official-monty/Monty) (AGPL, Rust MCTS,
CPU policy+value nets) parameterizes contempt as the **Elo difference vs
the opponent** (validated across ±1000) and has a `Contempt_Analysis`
mode that reports the *practical* best move against that opponent model
rather than the objective one. Planned uses, in order:

1. Port Monty to WASM as an opponent (recipe: viridithas-wasm template —
   single-thread spawn shim for `std::thread::scope`, patch out runtime
   zstd/mmap in the embed path, ship the ~500MB nets as separate cacheable
   assets via WASI preopened files, not embedded).
2. Expose `Contempt_Analysis` as an analysis-page lane: objective best
   (Stockfish) vs practical best vs a ~N-rated human (Monty).
3. Adopt its Elo→WDL-rescaling calibration in our PUCT so the raw
   `drawScore` constant becomes "opponent ≈ Maia 1500", and use its
   contempt analysis as ground truth to tune `searchContemptLimit`.

Note: contempt cannot be injected into the classical UCI engines —
modern Stockfish removed its Contempt option and the others never had
one; their only human-facing knob is Skill Level.
