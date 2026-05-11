# Release Gates, Protocol Cards, and Distributed Arena Jobs

## One-command release gate

Run the standard non-architecture gate for any exported ONNX model:

```bash
scripts/run_model_release_gate.sh \
  --name MODEL_NAME \
  --model path/to/model.onnx \
  --meta path/to/model.meta.json
```

Useful knobs:

```bash
--visits-list 1,32,64,128,192,256,384,512
--quick-pairs 3
--full-pairs 20
--cpuct 1.5
--bucket-rows 5000
--stockfish-lite-full-searches shallow:nodes=32,deep:depth=6
--out-dir artifacts/release_gates/MODEL_NAME
```

The gate runs client build/tests, PUCT consistency, queen-risk diagnostics, bucket eval, and quick anchor games. By default, each quick/full anchor packet also includes two full-strength Stockfish Lite anchors (`UCI_LimitStrength=false`): `stockfish_lite_full_shallow_nodes32` to test whether we can contend with shallow full-strength search, and `stockfish_lite_full_deep_depth6` to expose how we lose to deeper Stockfish Lite. The budgets are bounded and can be overridden.

## Protocol cards

Benchmark writers should emit both:

```text
result.json
result.json.protocol.json
```

The protocol card records model paths, anchor configuration, openings, visit budget, cpuct, max plies, creation time, and shard metadata where applicable. Current support includes:

```text
eval/uci_anchor_arena.mjs
eval/onnx_round_robin_arena.mjs
eval/onnx_bucket_eval_jsonl.mjs
eval/puct_consistency_check.mjs
eval/merge_uci_anchor_arena.mjs
```

## Visit curve protocol

Cheap curve:

```text
visits: 1, 32, 64, 128, 192, 256, 384, 512
optional: 768, 1024
anchors: Stockfish1320, Stockfish1600, Maia1100, full-strength Stockfish Lite shallow nodes32, full-strength Stockfish Lite deep depth6
pairs: 3-5
max plies: 100
openings: eval/opening_suite_uho_lite_v1.fen, reversed pairs
root noise: off
temperature: 0
```

Promotion protocol:

```text
If a visit setting wins the cheap curve, rerun it with pairs=10-20 and full anchors:
Stockfish1320, Stockfish1600, Maia1100, Maia1500, Maia1900, and full-strength Stockfish Lite shallow/deep anchors (or documented stronger Lite budgets).
```

Claims should stay protocol-relative and include WDL, illegal counts, anchors, visit budget, openings, backend, and error bars.

## OpenBench promotion lane

Use [AndyGrant/OpenBench](https://github.com/AndyGrant/OpenBench) as the second-layer distributed testing system for serious promotion candidates, not as a replacement for local model diagnostics.

OpenBench is appropriate once a candidate has already passed the local release gate, queen/material diagnostics, and Stockfish/Maia/Lite anchor packets. Its role is to answer the high-sample question:

```text
candidate UCI engine vs incumbent UCI engine, fixed openings/search budget, SPRT/fixed-game confidence
```

Keep local tooling responsible for model-aware work:

```text
ONNX metadata, browser latency, policy drift, AV/rank/regret/risk/uncertainty weights,
queen diagnostics, shallow Stockfish blunder analysis, and Stockfish/Maia anchor packets.
```

Prerequisite: expose each model/search configuration as a stable UCI engine wrapper, e.g.:

```text
scripts/uci_tiny_leela.mjs
```

Minimum UCI contract:

```text
uci
isready
ucinewgame
position fen ...
position startpos moves ...
go nodes N      # map to PUCT visits=N or configured visit budget
go depth N      # optional compatibility mapping
go movetime N   # optional time-managed search
bestmove ...
```

Useful options:

```text
setoption name Model value path.onnx
setoption name Meta value path.meta.json
setoption name Mode value policy/puct/aux
setoption name Visits value 64
setoption name Cpuct value 1.5
setoption name AvWeight value 0.0025
setoption name RankWeight value 0
setoption name RegretWeight value 0
setoption name RiskWeight value 0
setoption name UncertaintyWeight value 0
```

Adoption checklist:

```text
[ ] Implement scripts/uci_tiny_leela.mjs.
[ ] Add UCI smoke tests: uci/isready/position/go nodes/bestmove/legal move.
[ ] Validate wrapper locally with cutechess-cli against Stockfish and incumbent tiny engine.
[ ] Add protocol cards for UCI wrapper config and OpenBench test IDs.
[ ] Stand up a private OpenBench instance or connect to an existing trusted instance.
[ ] Add promotion policy: local gate -> OpenBench SPRT/fixed-game -> promote/release.
```

Recommended promotion flow:

```text
1. Fast internal arena and eval-bucket checks.
2. Stockfish/Lite/shallow/Maia anchor packet.
3. Queen/material/blunder diagnostics and Stockfish cp-drop override review.
4. OpenBench high-sample test vs current champion/incumbent.
5. Promote only if strength, illegal rate, diagnostics, bytes, and latency all pass.
```

## Local distributed arena

Run a locally parallel, cloud-shaped visit curve:

```bash
NAME=chessformer_v1_100m_e3 \
MODEL=public/models/chessformer_v1_100m_e3_single.onnx \
META=public/models/chessformer_v1_100m_e3_single.meta.json \
OUT_DIR=artifacts/distributed_arena/run1 \
VISITS_LIST=1,32,64,128,192,256,384,512 \
PAIRS=3 \
OPENING_SHARD_SIZE=3 \
JOBS=3 \
scripts/run_distributed_visit_curve_local.sh
```

Outputs:

```text
OUT_DIR/manifest.json
OUT_DIR/jobs/*.json
OUT_DIR/jobs/*.json.protocol.json
OUT_DIR/jobs/*.json.done
OUT_DIR/logs/*.log
OUT_DIR/merged/vVISITS.json
```

Merge arbitrary shards:

```bash
node eval/merge_uci_anchor_arena.mjs \
  --inputs 'artifacts/distributed_arena/run1/jobs/v512_*.json' \
  --out artifacts/distributed_arena/run1/merged/v512.json \
  --allow-mixed=true
```

## Accepted/candidate model discipline

For any future self-play, reanalysis, or distributed promotion loop, keep a hard separation:

```text
accepted models: allowed to generate future self-play / actor data
candidate models: evaluation-only until they pass the gate
rejected models: archived with protocol card and reason
```

This avoids poisoning future training data with every transient checkpoint.  Minimal manifests:

```text
accepted_model.json
candidate_model.json
promotion_result.json
```

Each should record model path, meta path, git commit, feature schema, move-map version, search config, protocol card paths, and promotion/rejection reason.  This pattern is expanded in `docs/distributed_selfplay_training_system_design.md`.

## Permanent parity tests

Current permanent checks cover:

```text
policy top-k equals root PUCT priors: eval/puct_root_prior_parity.mjs
promotion/castling action map roundtrips: tests/move_codec_roundtrip.test.mjs
mirrored FEN consistency: tests/mirrored_fen_consistency.test.mjs
browser-style bytes vs Node path evaluator parity: tests/evaluator_parity.test.mjs
CNN vs SquareFormer legal action-id adapter contract: tests/evaluator_parity.test.mjs
```

The browser parity test compares ONNX Runtime Web evaluator construction from filesystem path vs raw model bytes, matching the browser loading path. Full DOM/browser parity can later be upgraded to Playwright if needed.
