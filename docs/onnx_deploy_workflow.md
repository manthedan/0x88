# ONNX deployment optimization workflow

Use `onnxsim` as the default first deployment optimization pass for release/browser ONNX artifacts.

## Why

Early tests showed that file size may not move much, but graph noise drops substantially:

```text
Cast / Shape / Unsqueeze / Concat / Reshape / Gather clutter ↓
```

That can reduce ONNX Runtime session/init and inference overhead, especially for browser/WASM/WebGPU and multi-head aux exports.

## Standard flow

Keep training artifacts untouched and write a deploy sibling:

```bash
.venv-onnx/bin/python scripts/simplify_onnx_for_deploy.py \
  --model artifacts/path/model.onnx \
  --meta artifacts/path/model.meta.json \
  --out artifacts/path/model.sim.onnx \
  --meta-out artifacts/path/model.sim.meta.json
```

Then run semantic parity:

```bash
node --experimental-strip-types eval/onnx_parity_check.mjs \
  --model-a artifacts/path/model.onnx \
  --model-b artifacts/path/model.sim.onnx \
  --meta artifacts/path/model.meta.json \
  --positions 8 \
  --tolerance 1e-5
```

Then run latency/search smoke:

```bash
ORT_INTRA_OP_NUM_THREADS=4 node --experimental-strip-types eval/puct_batch_benchmark.mjs \
  --model artifacts/path/model.sim.onnx \
  --meta artifacts/path/model.sim.meta.json \
  --visits 32,64,128 \
  --batches 16 \
  --repeats 5 \
  --positions 4
```

## Release gate

A simplified model is deployable only if:

```text
onnx.checker passes
policy/WDL parity <= 1e-5
aux head parity <= 1e-5 when exported
PUCT smoke has no illegal moves/runtime failures
latency is no worse than original, or the size/init benefit justifies it
```

## Notes

- Use `--no-large-tensor` by default to avoid accidental graph bloat from folding `Tile` or `ConstantOfShape`.
- For dynamic candidate models, use `--check-n 0` and rely on project parity tests rather than random-input `onnxsim` checks.
- `onnx/optimizer` remains optional/manual; try it only after `onnxsim` if a model still has obvious graph overhead.
