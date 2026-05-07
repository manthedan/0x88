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
--out-dir artifacts/release_gates/MODEL_NAME
```

The gate runs client build/tests, PUCT consistency, queen-risk diagnostics, bucket eval, and quick anchor games.

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
anchors: Stockfish1320, Stockfish1600, Maia1100
pairs: 3-5
max plies: 100
openings: eval/opening_suite_uho_lite_v1.fen, reversed pairs
root noise: off
temperature: 0
```

Promotion protocol:

```text
If a visit setting wins the cheap curve, rerun it with pairs=10-20 and full anchors:
Stockfish1320, Stockfish1600, Maia1100, Maia1500, Maia1900.
```

Claims should stay protocol-relative and include WDL, illegal counts, anchors, visit budget, openings, backend, and error bars.

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
