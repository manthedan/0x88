# Contempt in the LC0 browser PUCT search

Status, 2026-06-11: all three mechanisms shipped (`src/search/puct.ts`),
wired through the search worker to every Lc0-family opponent on the Play
page, and validated against Maia as a human stand-in. **Every contempt
benefit runs without the ~1GB Monty dependency**: the search side is
native to our PUCT, and the human-modeling side (rating-conditioned
outcomes, rating inference, opponent modeling) runs on Maia3 (45.7MB).

Provenance: mechanisms 1–2 take *patterns only* from the lc0
search-contempt fork and Monty (both AGPL) — no code. Mechanism 3
(`applyEloContempt`) is an independent TypeScript re-expression of the
math in Monty's `apply_contempt` (the logistic-latent fit and the
`s²·elo·ln10/(400·16)` shift), written from reading that AGPL source —
noted here for provenance transparency.

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

## Mechanism 3: Elo-calibrated contempt (`contemptElo`)

Port of Monty's `apply_contempt` (src/chess.rs, the native-oracle lane
from the verdict below): the assumed **Elo advantage of the root side**,
in [-1000, 1000]. Each evaluated WDL triple is fitted to a logistic
latent model — win = σ((μ−1)/s), loss = σ((−μ−1)/s), giving mean μ and
sharpness s — then μ is shifted by `Δμ = s²·elo·ln10/(400·16)` (clamped
to ±0.8) and the triple re-derived.

Why this is the principled successor to a flat `drawScore`:

- the **s² factor fades contempt out in decided positions** (a flat
  drawScore keeps pushing even in lost ones — the exact failure we
  measured at queen odds);
- near-certain WDLs (and therefore terminal draws) are guarded and
  returned untouched;
- the knob has units: `contemptElo 600` means "treat the opponent as
  600 Elo weaker", directly comparable to Maia ladder ratings.

Sign flips at opponent nodes by the same absolute side parity as
`drawScore`; the setting is part of `searchValueKey`, and `0` is exact
identity (same array, no float ops). Implemented in `applyEloContempt`
(src/search/puct.ts), plumbed through searchWorker/Bt4SearchOptions.
`tests/puct_elo_contempt.test.mjs` checks the transform against 15
samples captured from the native Monty binary's `eval` command (matches
within print precision), plus sign/guard/parity/tree-reuse behavior.

### Calibration vs Maia3's learned contempt (2026-06-11)

`scripts/maia3_vs_elo_contempt.mjs` compares the analytic transform with
Maia3's rating-conditioned value head — the *learned* expected outcome
between players of the conditioned ratings — over 120 playout positions,
mean rating held fixed:

| base | gap | Maia3 shift | transform shift | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1500 | +200 / −200 | +0.046 / −0.121 | +0.023 / −0.023 | 2.0 / 5.2 |
| 1500 | +400 / −400 | +0.104 / −0.170 | +0.030 / −0.031 | 3.4 / 5.5 |
| 1500 | +800 / −800 | +0.181 / −0.218 | +0.038 / −0.038 | 4.8 / 5.7 |
| 2000 | +200 / −200 | +0.066 / −0.140 | +0.016 / −0.017 | 4.0 / 8.4 |
| 2000 | +400 / −400 | +0.129 / −0.224 | +0.020 / −0.020 | 6.5 / 11.1 |
| 2000 | +800 / −800 | +0.189 / −0.283 | +0.022 / −0.023 | 8.6 / 12.5 |

Findings:

1. **Monty's 16× damping (the `400·16` denominator) heavily undershoots
   real human outcome statistics** — by ~2–6× at base 1500 and ~4–12×
   at 2000. The constant was evidently tuned for engine search behavior
   (where the rescale compounds over the tree), not for matching human
   game outcomes per evaluation.
2. **Human rating gaps are asymmetric**: being outrated costs far more
   than outrating gains (the weaker side's blunders dominate), and the
   analytic transform is symmetric by construction.
3. Caveats: random-playout positions are not human-game positions
   (distribution shift for Maia3's value head), and per-eval shift size
   is not the same thing as best *search* behavior — the search A/B
   remains the arbiter for the contemptElo knob itself.

Practical reading: keep `contemptElo` as the cheap search-side rescale
(the A/B says it is safe), but where the actual rating-conditioned
outcome matters — the analysis lane, user-facing win estimates — query
Maia3 directly rather than scaling a strong-engine WDL.

### Rating inference status (for the Elo-contempt Play mode)

`scripts/maia_elo_probe.mjs` infers a player's rating from their moves.
Two scorers:

- `--scorer ladder` (5 discrete Maia nets): 3/3 correct buckets by ~20
  moves — but partly self-referential, since the simulated players ARE
  sampled Maia nets of the same model family.
- `--scorer maia3` (one session, continuous selfElo grid + parabolic
  MLE, oppoElo conditioned on the known opponent): cross-model test.
  Simulated Maia 1100 → ≈1073 (−27), Maia 1500 → ≈1379 (−121), both
  stable from ~20 moves. Simulated Maia 1900 → ≈1362 (−538) with a
  nearly flat likelihood profile above 1400.

The 1900 miss is confounded: Play-page tail-trimmed *sampling* from old
Maia-1900 may genuinely play below 1900 (an observation that also
applies to our "Maia 1900" opponent as shipped!), and/or Maia3's
discrimination weakens toward the top of the range. Resolving needs
real rated human games (e.g. lichess PGNs) as ground truth. For the
product mode: self-reported rating remains the primary input; the
Maia3 estimator is a refinement that is currently trustworthy in the
1000–1600 band where most human players live.

## Tree reuse safety

Backed-up node values depend on the contempt settings that produced them,
so a tree built under one setting must not seed a search under another.
`searchValueKey(context)` fingerprints `drawScore`/`searchContemptLimit`/
`contemptElo`/`rootSideToMove`; root reuse and transposition-table hits
require a matching key (`nodeSearchValueCompatible`), with pre-key trees
compatible only with the neutral default. Covered by
`tests/puct_contempt.test.mjs` ("changed contempt settings do not reuse
stale search tree") and `tests/puct_elo_contempt.test.mjs`.

All mechanisms off (`0`) ⇒ bit-identical search (puct/lc0 parity suites).

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

A second queen-odds run (12 games/arm, v16, vs Maia 1900) after
`contemptElo` landed:

| Arm | Score | Record | Note |
| --- | ---: | --- | --- |
| scLimit 16 | 71% | +7 =3 −2 | 92% last run ⇒ 12-game arms are noisy |
| scLimit 16 + contemptElo 600 | 71% | +8 =1 −3 | more decisive (11 mates, 1 draw) |
| contemptElo 600 alone | 63% | +6 =3 −3 | vs 58% baseline last run |

Reading both runs together: `searchContemptLimit` remains the lever;
`contemptElo` is neutral-to-mildly-positive and — unlike raw `drawScore`,
which dragged the scLimit arm from 92% to 63% — **safe to stack on top**
(its s² fade keeps it from fighting the pessimistic value head in lost
positions). Its real value is semantic: an interpretable, user-facing
knob denominated in Elo. Play-page defaults stay scLimit-based.

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
| Lc0 small / t3 / BT4 | −0.25 | 16 | default |
| Leela Queen Odds | −0.5 | 24 | 1.5 |
| Maia (human model) | — | — | — |

LQO mirrors its README (DrawScore ±0.4–0.6, CPuct 1.5, ScLimit 32–40 at
12–15k nodes) scaled to browser visit budgets; unlike t1, its value head
is odds-calibrated, which is why drawScore stays on for it.

Regular Lc0 opponents get the A/B-validated `scLimit 16` as well: the
limit models the *opponent's* search ability, not ours, so it stays fixed
across the visit ladder. Their drawScore −0.25 stays on because, unlike
the queen-odds arm where it dragged the score down, equal-material play
is the scenario the guidance above recommends it for (engine
equal-or-better). Validated by an equal-material harness run
(`--odds none`, see below).

## Monty: calibrated contempt in the browser (port DONE)

[Monty](https://github.com/official-monty/Monty) (AGPL, Rust MCTS,
CPU policy+value nets) parameterizes contempt as the **Elo difference vs
the opponent** (validated across ±1000) and has a `Contempt_Analysis`
mode that reports the *practical* best move against that opponent model
rather than the objective one.

**The wasm32-wasip1 port is built and gated** (2026-06-11):

- `patches/monty-wasip1.patch` (~130 lines on upstream `0950aff1`):
  inline-execution shims for the four `std::thread::scope` sites
  (search loop, go/stop stdin poll, tree init/clear), zstd+memmap2 moved
  to non-wasm target deps, a wasm `read_into_struct_unchecked` that reads
  nets into a leaked aligned box, and argv-seeded UCI commands for the
  one-shot worker mode. Build: `scripts/build_monty_wasi.mjs` →
  `public/monty/monty.wasm` (~560KB).
- Networks are **not embedded**: the build uses Monty's non-embed path,
  which opens the raw nets by canonical name from the WASI preopened cwd
  (`nn-09da29a4b6ed.network` value ~661MB, `nn-6e49a41bd7c0.network`
  policy ~286MB; extracted from the 0950aff1 release binary since
  tests.montychess.org was down). They are **not committed**; local lab
  runs may stage them under `public/models/monty/`, but product builds
  must not copy the ~950MB pair.
- Loader: `recklessWasiWorker.ts` gained generic `preopenFiles`
  (fetch + cache + progress); `MontyEngine` (montyEngine.ts) is a
  persistent-preferred `BrowserUciEngine` with a `contempt` option.
- Gates: `scripts/monty_wasi_smoke.mjs` (browser_wasi_shim + real nets);
  `lab/monty-smoke.html` click-tested in Chromium (persistent mode).
  Search parity vs the native release binary is **bit-identical** at
  fixed nodes (same node counts/scores/PVs, contempt WDL rescaling
  identical to the 0.01%), browser ~50-60k nps. Contempt visibly works:
  at +400 the startpos choice shifts d2d4 → e2e4 with cp 22 → 46.

### The 950MB verdict (decided 2026-06-11)

The net pair does not get smaller, so Monty is **not a product
opponent**:

- Both nets are **already int8-quantized**. The value net is one giant
  sparse layer (~80k threat features × 8192 hidden, i8) ≈ 660MB of
  1-byte weights; there is no f16/quantization halving left. The size is
  architectural — sized for ~5000-Elo TCEC play, not for beating humans.
- Max-level zstd (what the release binaries embed) only reaches ~505MB
  transfer, and does nothing about the ~1GB resident in wasm memory plus
  JS-side copies — tab-OOM territory on 8GB machines. Our shipping
  ceiling so far is LQO at 189MB, and even the ~330MB BT4 lane is heavy.
- Every contempt-capable commit (contempt landed 2025-11, #134/#135)
  uses this same 661MB value net. Only the policy net was ever smaller
  (16384-wide, ~115MB), which doesn't change the verdict — and old nets
  are only on the dead tests.montychess.org (upstream keeps one rolling
  prerelease; old binaries to carve nets from are deleted). Backporting
  contempt onto a 2024-era small-net Monty or training our own small
  value net with their in-repo trainer are real projects, not tweaks.

Decision — Monty's three roles split by where they run, with outcomes
(updated 2026-06-11, after Maia3 landed):

1. **Calibration oracle — COMPLETE, dependency retired.** Monty's
   Elo→WDL rescaling was ported as `applyEloContempt` and verified once
   against the native binary; those 15 oracle samples are frozen into
   `tests/puct_elo_contempt.test.mjs`, so the port stays honest without
   Monty present. The deeper calibration question ("does the transform
   match real human outcomes?") is now answered by **Maia3**, not Monty
   — see the learned-vs-analytic table above.
2. **Browser lane — lab only, never product** (unchanged).
   `lab/monty-smoke.html` / `MontyEngine` stay lab-scoped. The raw network
   files were removed from tracked public assets; local tests must stage
   ignored copies/symlinks under `public/models/monty/`. If ever surfaced:
   explicit ~950MB opt-in + AGPL corresponding-source archive +
   stream-into-wasm memory hardening.
3. **Contempt_Analysis — rehosted Monty-free, in a stronger form.** The
   practical-vs-objective lane is covered by (a) our own PUCT with
   `contemptElo`/`scLimit` (the Elo-shifted abstraction, same shape as
   Monty's mode) and (b) the planned mixed-policy search where opponent
   nodes take **Maia3 priors at the actual user's rating** — "best move
   against this human", which Monty's formula-based opponent model
   cannot express. User-facing outcome estimates query Maia3's value
   head directly (more faithful than the analytic transform by 2–12×).

**Net result: every contempt benefit ships without the ~1GB Monty
dependency.** The human-modeling side runs on Maia3 (45.7MB, cached,
already shipping); the search side is native to our PUCT (0MB). The
native Monty binary and extracted nets (`models/monty/`, ~947MB) are
archival — re-extractable from the GitHub release binary if a
re-validation is ever needed — and the wasm port remains a lab
curiosity.

Note: contempt cannot be injected into the classical UCI engines —
modern Stockfish removed its Contempt option and the others never had
one; their only human-facing knob is Skill Level.
