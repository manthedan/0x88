# Centipawn s4fix V EMA

The served model id is `s4fix-v-ema`. The 2026-11-13 ship bundle supplies
`public/models/s4fix_v_ema.onnx` (6,224,461 bytes) and
`public/models/s4fix_v_ema.meta.json` (4,180 bytes). Both are unchanged from the
bundle. The ONNX is float32, with dynamic batch and int64 indices. Neither the
fp16 nor fixed-batch-16 variants is shipped.

SHA-256:

- ONNX: `58090183f9e3d7a50c68009a04a46da22222a7455127252904c9d16d03533027`
- Metadata: `7264cd1dcc86a530361d69b39ce87ba0581f12c63b4ef9acc8999ce1d52f6179`

## Provenance

The bundle identifies the source as tiny_leela_v2, branch `rebuild/lean-data`.
The checkpoint is `campaign_20261028T175023Z/stage4_fix/runs/s4fix_v/checkpoint_latest.pt`
under that training project's run-data directory. Export uses
`python -m tl2.export --checkpoint CKPT --weights ema --output s4fix_v_ema.onnx`.
The strength measurements used this exact float32 export. The bundle does not
state a model licence or a new training-corpus inventory; do not infer either
from the preceding model's provenance. Model licensing needs maintainer confirmation.

This is an eight-layer SquareFormer with ThreatGraph TG1, width 128 and about
1.16 million parameters. The preceding served `bt4_soap_rem_c19000_final` has
six layers. Its files and compiled runtime remain for historical releases,
not as a runtime for this new graph.

## Contract

Inputs are `tokens`, int64 `[batch, 64, 24]`, and `attack_summary`, float32
`[batch, 64, 28]`. Attack features are raw counts; the graph divides by eight.
The token input is compact embeddings, with seven earlier piece planes and
side-to-move-as-White rank-flip normalization (`stm_white_rankflip_v1`).
Squares are a1=0 through h8=63 in that normalized frame.

Outputs are `policy` logits `[batch, 20480]`, `wdl` logits `[batch, 3]` in
win/draw/loss order for the side to move, `q` `[batch]`, `hidden`
`[batch, 64, 128]`, and optional `wdl_sf18` logits `[batch, 3]`. The evaluator
requests outputs by name and does not read hidden. Search uses policy and
ordinary WDL, not q or the auxiliary head.

Ordinary policy indices are `from * 64 + to`. Promotion indices are
`4096 + (from * 64 + to) * 4 + promotion`, with knight=0, bishop=1, rook=2,
queen=3. Black moves are rank-flipped into the model frame.

The evaluator reads the same metadata contract as the preceding model:
`kind`, `input_dim` 112, `policy_size` 20480, `history_plies` 7,
`token_features` 24, `attack_summary_schema` `threatgraph_square_summary_v1`,
`attack_summary_feature_count` 28, `board_normalization`, `input_mode`,
`input_format`, `input_index_dtype` int64 and `onnx_dynamic_batch` true.
There is no action-value head; its legal-width metadata fields are absent.
The metadata contains no ONNX digest or policy-temperature override.

`src/nn/runtimeRegistry.ts` supplies policy temperature 2 for this model;
other models default to 1. `SquareFormerEvaluator` takes softmax(logits / 2)
over legal moves only. WDL and auxiliary WDL retain plain softmax.
`src/search/centipawn.ts` supplies the native monty_lc0both constants to all
three product search surfaces without changing generic PUCT or other engines.

## Encoder findings

The initial site/native comparison had 872 token differences, and no attack
summary differences. Six FEN-only midgame positions (the four promotion and
two castling examples) account for 616 differences: history planes 1–7 were
empty in the site, but repeat current pieces in native inputs. For example,
white promotion position b2 planes 1–7 were 0 instead of black pawn id 7;
g2 was 0 instead of king id 6, e7 was 0 instead of pawn id 1, and g7 was 0
instead of king id 12. Castling examples differ on every occupied square.
The fixture's ordinary start position retains empty unavailable planes.

The other 256 differences are repetition flags, broadcast over all 64
squares: `game 27w ply 133` plane 20, `game 30w ply 137` plane 19,
`game 48b ply 54` plane 20, and `game 33b ply 86` plane 17 were 0 instead
of 1. Their matching older history indices are 7, 20, 7 and 8 respectively
(zero-based, newest first). Piece planes read seven previous boards, but
repetition flags must look farther back. The former site implementation
searched only those seven boards.

The new model's `nativeHistory` setting fixes both cases without changing
historical encoders. Product surfaces keep the whole line and do not truncate
this model's evaluator cache context to the legacy 16-position window.
Synthetic repeated padding never creates repetition flags. The fixture covers
FEN-only midgames with fullmove greater than one; behavior for other missing
history shapes is not independently specified by this bundle.

## Strength facts from the bundle

At 256 nodes, against the previous ship candidate `pb4_1m_sqrt0_ema`, the
model scored 466 wins, 691 draws and 379 losses in 1,536 games: +20 Elo,
95% interval +7 to +33. That preceding candidate measured about 2980–2990
on the CCRL Blitz scale using Stockfish 18 UCI_Elo anchors and rated-engine
ladders, with about 70 Elo variation by opponent. This model has not itself
been measured against those anchors, nor compared with the previously served
`bt4_soap_rem_c19000_final`.

Self-play measured +131 Elo per node doubling from 256 to 2,048 nodes and
+83 from 2,048 to 4,096. There were no wins against full-strength Stockfish
18 in 300 games. Moves losing 30 points of winning chance occurred about
1.5 times per 100 moves, similar to the previous candidate. The eight-layer
family measured 1.28 times the WASM time per position of the six-layer family.
Policy temperature 2 was worth +60 to +86 Elo at 256 nodes on this model
family. Search batching above one was not part of those strength measurements.

The bundle attributes these results to tiny_leela_v2's
`docs/rebuild/gpu_runbook.md`, sections "Stage 4 fix screen" and "Match play
against other engines". These are supplied measurement claims, not a new site
Elo calibration. Existing Play strength labels are not recalibrated here.

## Local checks

The copied fixtures are `tests/fixtures/centipawn_s4fix_parity_fixture.json`
and `tests/fixtures/centipawn_s4fix_search_fixture_n256.json`.
Run the parity test with Node's type stripping. Run the full-line search check
with `node --experimental-strip-types scripts/check_centipawn_s4fix_search.mjs`;
an optional output filename writes the JSON report. It gates at 70% agreement
and requires all positions with absolute native value at least 0.8 to agree.
The search check exceeds 60 seconds locally, so is not a full-suite test.
The bundle records 256 total native visits but no per-move native visit counts.

The new model is ONNX Runtime Web only. No compiled runtime is registered for
it. Explicit or persisted TVMJS/custom selectors, even strict selectors with
fallback disabled or an old manifest URL, are intercepted before compiled
loading and resolve to ORT with the existing runtime-audit fallback warning.
Historical runtime code and files remain unchanged. No browser/GPU validation
or remote publication is part of these local checks.
