# En Croissant feature ideas for lc0-webgpu

En Croissant is a GPL-3.0 desktop chess toolkit with strong analysis, engine-management, database, and repertoire workflows. Treat these as product/UX inspiration; do not copy implementation code without an explicit license decision.

Sources:

- <https://github.com/franciscoBSalgueiro/en-croissant>
- <https://encroissant.org/docs/>
- <https://encroissant.org/docs/guides/analyze-game>
- <https://encroissant.org/docs/guides/manage-repertoire>
- <https://encroissant.org/docs/guides/configure-engines>
- <https://encroissant.org/docs/reference/database-format>

## Candidate features

1. **Game report / auto-annotation**
   - Generate a full-game report with mistake/blunder labels and optional variations.
   - Use win-probability loss from centipawns, MultiPV comparison, novelty detection against a reference database, and tactical/sacrifice heuristics.
   - Natural fit for the analysis board once engine result storage is durable.

2. **Better multi-engine analysis UX**
   - Show side-by-side best lines for all selected engines instead of only a merged PV list.
   - Highlight first-move consensus and disagreements.
   - Show eval spread/disagreement in centipawns.
   - Add runtime/search badges such as depth, visits, nodes, and NPS where engines expose them.
   - Use this to compare LC0, Stockfish, Reckless, Viridithas, Berserk, and PlentyChess on one position.

3. **Engine configuration profiles**
   - Save reusable analysis profiles containing selected engines, variants, strengths, MultiPV, and LC0 runtime.
   - Later extend row limits beyond the current single `strength` value to support explicit UCI-style search limits: depth, movetime, nodes, infinite.
   - Later expose common UCI options such as MultiPV, Hash, Threads, Skill, and engine-specific options where browser adapters support them.

4. **Tablebase integration**
   - Query Lichess tablebase for <=7-piece endgames.
   - Show WDL/DTZ/DTM-style status in analysis and use tablebase best moves when available.

5. **Repertoire builder/trainer**
   - Store repertoire lines as PGN.
   - Add build/train modes, coverage/gap navigation, transposition awareness, and spaced repetition.

6. **Database/search/opening explorer**
   - En Croissant uses SQLite; browser equivalent could use IndexedDB or OPFS.
   - Useful workflows: import PGNs, absolute/partial position search, opening tree stats, and local reference databases.

7. **Asset and engine management**
   - Turn our engine artifact manifests into a UI: installed/missing assets, size, license, source link, runtime capability, and local build instructions.
   - Current first step: the analysis page preflights the large local BT4 ONNX asset and shows the `lc0_prepare_model_assets.mjs` recovery command instead of letting BT4 fail later during analysis.

8. **Lichess/chess.com import pipeline**
   - Expand current import flow into a first-class “fetch my games, analyze locally, generate report” workflow.

## Current development focus

Initial implementation starts with:

- multi-engine comparison summary/table in `lc0-analysis.html`, and
- local saved analysis profiles for built-in engine variants, engine list, MultiPV, and LC0 runtime. Custom URL variants are intentionally not persisted until profiles can record the exact artifact URLs.
