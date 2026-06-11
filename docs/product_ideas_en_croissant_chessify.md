# Product ideas: En Croissant / Chessify gap notes

2026-06-10. Companion to `arena_analysis_roadmap.md` (EngineBattle gaps).
Based on product knowledge through early 2026 — pull the En Croissant repo
into `repos/` for an EngineBattle-style inventory before committing a
detailed roadmap.

## From En Croissant (local-first, open source — the closer cousin)

1. **Position-searchable game database** (their killer feature, our biggest
   gap). Import large PGN collections persistently; search by *position*
   (zobrist-keyed index), material, pattern. Browser shape: OPFS/IndexedDB +
   zobrist index, pure TS. Foundation for ideas 6 and the repertoire
   deviation detector; upgrades the existing opening explorer from "move
   stats" to "your actual games from this position".
2. **Repertoire builder + spaced repetition.** Repertoire lines as trees,
   SRS drilling, and deviation detection against imported games ("left book
   here"). Pure TS + OPFS.
3. **Unified engine manager page.** One screen for all engine families:
   variants, download sizes, cache status (Cache API), defaults, runtime
   status. We have all the data (typed catalog, artifact manifests, feature
   detection) — it's just scattered across per-family selectors. The
   per-engine packaging repos make this more natural.
4. **Multi-tab board workspaces.** Real value, but a structural rewrite of
   the single-board pages; ranked last.

## From Chessify (commercial, cloud-centric)

5. **Scan board/diagram → FEN.** Their best non-cloud idea. Unusually good
   fit: we already ship ONNX Runtime + WebGPU, so a small client-side board
   detection + piece classification model is a differentiating zero-install
   flex. Medium-hard (model needed); inference infra exists.
6. **Novelty detection + ECO opening naming.** Mark where a game leaves
   known territory (vs imported DB / master book), name openings. Cheap once
   idea 1 exists; pairs with game review ("novelty on move 12").
7. **Remote compute, adapted not copied.** Their core is renting server NPS;
   ours is zero-install. The mission-consistent version is a
   **UCI-over-WebSocket bridge**: point the analysis board at your own
   desktop engine or VPS. One adapter implementing `BrowserUciEngine` over
   WS; slots into the multi-engine UI and resource broker. "Bring your own
   compute."

## Ours, enabled by what we just built

8. **Blunder-to-puzzle training.** Game review already identifies blunders
   with best moves and win-prob swings; generate a personal puzzle deck from
   them (SRS-scheduled via idea 2's trainer). Nearly free; the feature
   people actually want game review for.

## Shortlist

(1) game DB → foundation; (8) blunder puzzles → weekend-sized win on game
review; (3) engine manager → makes the packaging/promotion work visible;
(7) WS bridge → best power-user lever; (5) scanner → the moonshot we're
well-placed for.
