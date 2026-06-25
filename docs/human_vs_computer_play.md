# Human vs computer play

Human-facing chess engines have two different jobs that are easy to conflate:

1. **Play like a human**: make the moves a rated human is likely to make.
2. **Beat a human**: keep a strong engine's search, but bias it toward practical
   chances against an imperfect opponent.

This project uses both patterns, but they should be labeled differently in the
UI and evaluated with different tests.

## Maia3: neural human modeling

Maia3 is the direct human-modeling lane. It is a neural policy/value model
trained on real human games and conditioned by rating. Given a position and a
self/opponent Elo pair, it predicts the move distribution and expected outcome
for players in that rating band.

Product meaning:

- **Opponent identity**: "a 1500-ish human," not "a weakened super-engine."
- **Move choice**: sample or argmax from the human-policy distribution.
- **Strength control**: rating conditioning, not depth or node throttling.
- **Analysis value**: shows what humans are likely to miss, what humans usually
  play, and how outcomes shift by rating.

This makes Maia3 the best fit for a human-like sparring partner and for the
Analysis page's "Human moves" panel. Its mistakes are part of the model: they
come from the human-game distribution rather than from intentionally crippling a
strong search.

## LQO / Monty: contempt-based practical play

The LQO/Monty lane keeps the computer-engine framing. The engine still searches
for strong moves, but the evaluation/search is adjusted for a fallible human
opponent instead of a perfect refuter.

In this repo that is represented by three related mechanisms:

- **Leela Queen Odds (LQO)**: an LC0-derived network fine-tuned for queen-odds
  play against humans. It evaluates the queen-down start as practically playable
  and pairs that net with aggressive search settings.
- **Search contempt (`searchContemptLimit`)**: opponent nodes are treated as
  budget-limited. After a visit limit, the search samples from the frozen child
  distribution instead of assuming the opponent always finds the deepest
  refutation.
- **Monty-style Elo contempt (`contemptElo`)**: a cheap analytic WDL rescale
  inspired by Monty's `apply_contempt`, validated against the native Monty
  binary and then rehosted in our own PUCT code.

Product meaning:

- **Opponent identity**: still a computer, but one trying to maximize practical
  chances against a human.
- **Move choice**: search move, biased toward traps, decisive outcomes, and
  non-sterile pressure.
- **Strength control**: search budget plus contempt knobs.
- **Analysis value**: "what is the best practical move against this class of
  opponent?" rather than "what would that opponent actually play?"

Monty itself remains a lab/reference engine rather than a product dependency:
its network pair is roughly a gigabyte resident, so the shippable version is the
Monty-shaped math and search behavior inside our existing PUCT lane.

## Comparison

| Dimension | Maia3 neural lane | LQO / Monty contempt lane |
| --- | --- | --- |
| Core question | What would a rated human play? | What move gives a strong engine practical chances against a human? |
| Training signal | Human games, rating-conditioned | Strong engine/search plus odds/contempt tuning |
| Typical UI label | Human-like, rated human, Maia3 | Practical engine, odds bot, contempt search |
| Move distribution | Human-policy probabilities | Search principal variation / best move |
| Errors | Authentic human-like errors | Engine avoids assuming perfect defense; does not become human |
| Best use | Sparring, human-move explorer, rating-conditioned outcome estimates | Odds play, anti-draw pressure, trap-aware engine play |
| Main risk | Can be less objectively strong by design | Can feel engine-like or overfit to a scenario if knobs are miscalibrated |

## Product guidance

- Use **Maia3** when the promise is human authenticity: "play like a 1500," "what
  do humans play here?", or "how does this line score in human games?"
- Use **LQO/search contempt** when the promise is practical engine play: "can the
  engine create chances from queen odds?", "avoid sterile draws," or "play the
  move that pressures this human."
- Do not market contempt search as human-like play. It models the opponent's
  limitations; it does not model the engine's own moves as human.
- Do not route Maia3 through LC0 PUCT by default. A future mixed-policy mode may
  be useful, but it should be labeled experimental because it is a hybrid, not
  pure Maia play.

## Current implementation notes

- Maia3 ships as a modular browser model with provenance and quantization notes
  in [`model_provenance/maia3.md`](model_provenance/maia3.md).
- Contempt search design, calibration, and Monty provenance live in
  [`search_contempt_design.md`](search_contempt_design.md).
- LQO is documented as a separate LC0-derived engine in
  [`engine_catalog.md`](engine_catalog.md) and the public docs page.
