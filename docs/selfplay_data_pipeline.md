# Self-play data pipeline

Status: pilot/contract stage. Raw self-play rows are immutable; annotation and diagnostics are sidecars.

## Flow

```text
self-play generation workers
  -> raw selfplay_chunk_v1 JSONL/JSONL.ZST + manifest
  -> annotation sidecars
      - Stockfish root/selected-move eval
      - agent diagnostics and failure-packet candidates
      - optional future board-canonicalization metadata
  -> pipeline manifest joining chunks + sidecars
  -> cache generation for residual/CNN, SquareFormer/BT4, and AV/rank/regret heads
  -> training
```

## Current contracts and scripts

- Raw rows: `contracts/schemas/selfplay_chunk_v1.schema.json`
- Annotation rows: `contracts/schemas/selfplay_annotation_v1.schema.json`
- Failure packets: `contracts/schemas/failure_packet_v1.schema.json`
- Raw validation: `scripts/selfplay_chunk_validate.py`
- Stockfish sidecar: `scripts/selfplay_stockfish_annotate.py`
- Annotation validation: `scripts/selfplay_annotation_validate.py`
- Agent diagnostics: `scripts/selfplay_agent_diagnostics.py`
- Joined manifest: `scripts/selfplay_pipeline_manifest.py`

Example dry pipeline without a Stockfish binary:

```bash
.venv-onnx/bin/python scripts/selfplay_stockfish_annotate.py \
  --input data/selfplay/pilot/chunk.jsonl.zst \
  --out data/selfplay/pilot/stockfish_annotations.jsonl.zst \
  --mock-stockfish

.venv-onnx/bin/python scripts/selfplay_agent_diagnostics.py \
  --input data/selfplay/pilot/chunk.jsonl.zst \
  --annotation data/selfplay/pilot/stockfish_annotations.jsonl.zst \
  --out data/selfplay/pilot/agent_diagnostics.jsonl.zst \
  --failure-dir artifacts/selfplay_failure_packets/pilot

.venv-onnx/bin/python scripts/selfplay_pipeline_manifest.py \
  --chunk data/selfplay/pilot/chunk.jsonl.zst \
  --annotation data/selfplay/pilot/stockfish_annotations.jsonl.zst \
  --annotation data/selfplay/pilot/agent_diagnostics.jsonl.zst \
  --out artifacts/selfplay_manifests/pilot.json \
  --strict-annotations
```

## Why lc0-style side flipping exists

Many chess networks canonicalize positions so the network always sees the side to move as "us". For black-to-move rows, the board can be rotated 180 degrees and colors swapped; policy moves and value targets are transformed with it. This is not just cosmetic:

- it makes white and black patterns share parameters;
- pawns, king safety, castling attacks, and promotion races have a consistent direction;
- data efficiency improves because the network does not need to learn two color-specific copies of the same motif;
- relative-position attention/bias tables in transformer-like models can focus on chess geometry from the mover's perspective.

For Tiny Leela, side flipping should be introduced as an explicit deterministic cache transform, not as an annotation that mutates raw chunks. Before using it for BT4/SquareFormer training, add parity tests proving that FEN, history, legal moves, selected move, policy distribution, castling rights, en-passant square, and side-to-move WDL targets are transformed together.
