# Gumbel-Zero Phase 0 Parameter Card

Status: ready for small clean test, then initial CPU self-play chunk.  This lane is **rules-only / no supervised model**.

## Lane invariants

- Evaluator: `uniform` only for Phase 0 bootstrap.
- No `--model` / `--meta` arguments.
- No teacher labels, Stockfish, Maia, lc0, or supervised checkpoint provenance.
- Output roots stay under `data/selfplay/gumbel_zero/`.
- Supervised-bootstrap data stays separate under `data/selfplay/supervised_sp/`.

## Selected params

| Field | Small test | Initial run |
|---|---:|---:|
| evaluator | `uniform` | `uniform` |
| games | `8` | `2000` |
| max plies | `48` | `160` |
| visits | `8` | `16` |
| candidate count | `8` | `16` |
| min candidate visits | `1` | `1` |
| estimated Q | `pessimistic` | `pessimistic` |
| move selection | `argmax` | `argmax` |
| target temperature | `1.0` | `1.0` |
| cpuct | `1.5` | `1.5` |
| fpu | `0` | `0` |
| gumbel scale | `1.0` | `1.0` |
| visit penalty | `0.05` | `0.05` |
| batch size | `1` | `1` |

Rationale: this preserves the already-reviewed Phase 1 defaults, keeps the first full chunk deterministic enough for replay/debug, and avoids supervised contamination. `sample-target` is reserved as a later diversity ablation after the first chunk validates.

## Small test command

```bash
RUN=data/selfplay/gumbel_zero/phase0_smoke_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"
npm run selfplay:gumbel-zero -- \
  --evaluator uniform \
  --games 8 \
  --max-plies 48 \
  --visits 8 \
  --candidate-count 8 \
  --min-candidate-visits 1 \
  --estimated-q pessimistic \
  --move-selection argmax \
  --target-temperature 1.0 \
  --cpuct 1.5 \
  --fpu 0 \
  --gumbel-scale 1.0 \
  --visit-penalty 0.05 \
  --batch-size 1 \
  --seed 730001 \
  --progress-every 8 \
  --out "$RUN/chunk.jsonl" 2>&1 | tee "$RUN/run.log"
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$RUN/chunk.jsonl" | tee "$RUN/validate.log"
.venv-onnx/bin/python scripts/gumbel_zero_chunk_report.py "$RUN/chunk.jsonl" | tee "$RUN/report.log"
npm run selfplay:gumbel-to-training -- \
  --input "$RUN/chunk.jsonl" \
  --output "$RUN/training_expanded.jsonl" \
  --manifest-out "$RUN/adapter_manifest.json" \
  --lane zero \
  --mode expanded \
  --value-target result \
  --history-plies 8 | tee "$RUN/adapter.log"
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$RUN/training_expanded.jsonl" \
  --min-policy-mass 0.99 --max-policy-mass 1.01 | tee "$RUN/adapted_validate.log"
```

## Initial run command

```bash
RUN=data/selfplay/gumbel_zero/phase0_initial_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"
nice -n 10 npm run selfplay:gumbel-zero -- \
  --evaluator uniform \
  --games 2000 \
  --max-plies 160 \
  --visits 16 \
  --candidate-count 16 \
  --min-candidate-visits 1 \
  --estimated-q pessimistic \
  --move-selection argmax \
  --target-temperature 1.0 \
  --cpuct 1.5 \
  --fpu 0 \
  --gumbel-scale 1.0 \
  --visit-penalty 0.05 \
  --batch-size 1 \
  --seed 730100 \
  --progress-every 25 \
  --out "$RUN/chunk.jsonl" 2>&1 | tee "$RUN/run.log"
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$RUN/chunk.jsonl" | tee "$RUN/validate.log"
.venv-onnx/bin/python scripts/gumbel_zero_chunk_report.py "$RUN/chunk.jsonl" | tee "$RUN/report.log"
npm run selfplay:gumbel-to-training -- \
  --input "$RUN/chunk.jsonl" \
  --output "$RUN/training_expanded.jsonl" \
  --manifest-out "$RUN/adapter_manifest.json" \
  --lane zero \
  --mode expanded \
  --value-target result \
  --history-plies 8 | tee "$RUN/adapter.log"
.venv-onnx/bin/python scripts/selfplay_chunk_validate.py "$RUN/training_expanded.jsonl" \
  --min-policy-mass 0.99 --max-policy-mass 1.01 | tee "$RUN/adapted_validate.log"
```

## Go / no-go after small test

Proceed to initial run only if:

- raw chunk validation passes,
- adapted chunk validation passes,
- `terminal_reasons` are sane (`max_plies` likely high for random play, but no parser/no-move blowup),
- candidate count and visit histograms match expected low-node behavior,
- adapter manifest has `lane=zero` and blank `source_model`.
