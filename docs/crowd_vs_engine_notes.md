# Crowd vs engine ("Kasparov vs the World" mode)

2026-06-10 design note. Idea: a page where everyone present votes on the
next move — crowd vs engine, or crowd vs crowd (visitors A/B-routed to two
sides, each voting against the other). Cheating is explicitly fine: against
another crowd it cancels out; against an engine it's fair play.

## Two things we have that make this distinctive

### 1. Deterministic engines → the server barely has to exist

Our parity discipline proves the wasm engines are **bit-reproducible**:
fixed depth/nodes, single thread, same artifact → identical best move,
score, node count, PV on every machine (60/60, 40/40 exact across the
suite). So the engine's reply doesn't need a trusted server: **every
voter's browser computes the engine move locally** and submits it with the
position hash; the coordinator just takes the agreeing majority. The
"opponent" is a smart-contract-style deterministic function shipped as
wasm. Caveats:
- Use integer-deterministic variants: Reckless/Viridithas/Berserk simd128
  and their relaxed *integer-dot* variants are exact by proof; exclude
  PlentyChess's relaxed build (its f32 tail depends on fused-madd
  hardware).
- Budget must be nodes/depth, never movetime.
- Fallback/tiebreak: the coordinator can also run the same artifact in Node
  (the packaging repos' loaders already run in Node) as the authority.

### 2. Embedded analysis = sanctioned cheating

We already ship the full multi-engine analysis board. Put it *on the voting
page*. "Cheating" becomes the game: kibitz with Reckless/SF/LC0, argue via
votes. Crowd-vs-crowd then measures coordination, not chess skill — which
is the fun part of Kasparov vs the World anyway (the World played
near-GM-level *because* of open analysis).

## Mechanics sketch

- **Voting**: time-boxed windows (30–60s, adaptive to traffic); plurality
  over legal moves (client-validated by our movegen); deterministic
  tiebreak (engine-eval order). Light dedup per session; no heavier
  anti-cheat by design.
- **Crowd vs engine**: strength matters — a full-strength engine crushes
  any crowd (the World lost to one Kasparov). Use the calibrated mid-tier
  ladder, escalating: beat Reckless d4 → unlock d6 → … Gauntlet
  progression gives the page a metagame.
- **Crowd vs crowd (A/B)**: random side assignment on arrival ("you're
  playing the other half of the internet"); symmetric cheating; optional
  side-reveal only after the game.
- **Post-game**: run our game review automatically — "the crowd played at
  87% accuracy, two blunders" — plus annotated PGN. Free, and it's the
  shareable artifact.
- **No chat.** Votes are the only user content → near-zero moderation
  surface.

## Architecture

First server-dependent feature in the project — keep it tiny:
- One small coordinator (single WS/SSE service or a Durable-Object-style
  KV+broadcast): game FSM, vote tally, clock, move broadcast. Payloads are
  FENs and tallies; all chess/engine/analysis logic stays client-side.
- Static pages stay on the existing isolated host; the coordinator is the
  only new operational surface.
- Scale shape: read-mostly broadcast + an atomic counter per legal move.

## What it gets us

Community flywheel for the engine showcase; a public stress test of the
packaged engines' loaders (browser + Node); and the deterministic-consensus
trick is a genuinely novel construction worth a write-up on its own.

## Open questions

- Operator commitment: this is an always-on service, unlike everything else
  in the project.
- Vote-window pacing vs engine think budget (instant engine replies feel
  bad; add theatrical delay or show the live PUCT root chart while it
  "thinks" — we have that chart).
- Identity-free streaks/leaderboards: probably localStorage-only, keep it
  anonymous.
