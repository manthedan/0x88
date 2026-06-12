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
  tests.montychess.org was down). They ship as separate cacheable assets
  (`public/models/monty/`, symlinks to `../models/monty/`).
- Loader: `recklessWasiWorker.ts` gained generic `preopenFiles`
  (fetch + cache + progress); `MontyEngine` (montyEngine.ts) is a
  persistent-preferred `BrowserUciEngine` with a `contempt` option.
- Gates: `scripts/monty_wasi_smoke.mjs` (browser_wasi_shim + real nets);
  `monty-smoke.html` click-tested in Chromium (persistent mode).
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

Decision — Monty's three roles split by where they run:

1. **Calibration oracle — native binary, the active lane.** Use Monty's
   Elo→WDL contempt rescaling as ground truth to calibrate our PUCT
   `drawScore`/`searchContemptLimit` (so settings become "opponent ≈
   Maia 1500" instead of raw constants). Zero download cost; this is
   what we actually wanted Monty for, and our own PUCT contempt is what
   ships to users.
2. **Browser lane — lab only, never product.** `monty-smoke.html` /
   `MontyEngine` stay lab-scoped (excluded from the product build). If
   ever surfaced, it is behind an explicit ~950MB opt-in and needs an
   AGPL corresponding-source archive first. Deferred hardening if that
   day comes: stream nets directly into wasm memory and drop the
   worker-side byte cache (halves peak memory).
3. **Contempt_Analysis as a product idea — rehost on our PUCT.** The
   practical-vs-objective-move lane is worth having, but our scLimit
   already models "best move against a budget-limited opponent" at zero
   extra MB; Monty serves as the native reference implementation.
   Revisit only if upstream publishes a small contempt-era net pair.

Note: contempt cannot be injected into the classical UCI engines —
modern Stockfish removed its Contempt option and the others never had
one; their only human-facing knob is Skill Level.
