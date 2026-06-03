# lc0web custom WebGPU inference checkpoint

This checkpoint records the current custom-kernel path for the batch-8 f16 `lc0web` pack. Native LC0/BLAS and f32 ONNX remain the source-of-truth correctness baselines; these browser probes are staged subgraph checks for a future custom WebGPU runtime.

## Implemented probes and benchmarks

- `?kernelBench=1`: fixed-shape WGSL `MatMul + Add` benchmark with scalar/tiled/transposed variants.
- `?ortOpBench=1`: tiny generated ORT `MatMul + Add` comparison.
- `?qkvProbe=1` / `?qkvBench=1`: encoder0 Q/K/V projections.
- `?attentionScoreBench=1`: encoder0 QK score matmul including smolgen score bias.
- `?attentionScoreOrtBench=1`: tiny ORT attention-score comparison including smolgen bias input.
- `?softmaxBench=1`: row softmax over the 8×64 attention-score rows.
- `?attentionValueBench=1`: attention-probability × value matmul.
- `?attentionValueOrtBench=1`: tiny ORT batched attention-value MatMul comparison.
- `?attentionBlockBench=1`: QKV → QK(+smolgen) → softmax → value.
- `?attentionOutputBench=1`: attention block plus output projection, residual, and ln1.
- `?attentionOutputOrtBench=1`: tiny ORT comparison for attention-value output through output projection, alpha residual, and ln1.
- `?encoder0FfnBench=1`: encoder0 FFN from ln1 through dense1/bias/sqrrelu/dense2/bias/alpha residual/ln2.
- `?encoder0FfnOrtBench=1`: tiny ORT comparison for encoder0 FFN dense1/sqrrelu/dense2/alpha residual/ln2.
- `?encoder0BlockBench=1`: full encoder0 attention+FFN block through ln2.
- `?encoder0BlockOrtBench=1`: tiny ORT comparison for attention-value output through attention output projection/ln1 plus FFN/ln2.

The browser page now emits a `benchmarkReport` object with browser metadata, GPU adapter info where available, pack verification mode, and timing summaries. `scripts/lc0_browser_wgsl_smokes.mjs` automates the main browser smokes and parses `maxAbsError`.

## Current evidence

Recent local Chromium/WebGPU/WASM smokes on the batch-8 f16 lc0web pack passed. These are browser smoke results, not CI guarantees:

- `npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-ffn,encoder0-block --timeout 25000`
  - `FFN_BENCH_DONE`, max absolute error about `3.34e-6`.
  - `ENCODER0_BLOCK_BENCH_DONE`, max absolute error about `3.58e-6`.
- `npm run lc0:browser-wgsl-smokes -- --no-server --only attention-value-ort-wasm --timeout 25000`
  - `ATTENTION_VALUE_ORT_BENCH_DONE`, max absolute error about `9.54e-7`.
- `npm run lc0:browser-wgsl-smokes -- --only attention-output-ort-wasm --timeout 25000`
  - `ATTENTION_OUTPUT_ORT_BENCH_DONE`, max absolute error about `1.67e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-ffn-ort-wasm --timeout 25000`
  - `FFN_ORT_BENCH_DONE`, max absolute error about `1.91e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-block-ort-wasm --timeout 25000`
  - `ENCODER0_BLOCK_ORT_BENCH_DONE`, max absolute error about `1.97e-6`.

Validation commands used during this checkpoint:

```sh
npm run typecheck
TINY_LEELA_ORT_EP=wasm node --experimental-strip-types --test tests/lc0_wgsl_kernel_probe.test.mjs
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-ffn,encoder0-block --timeout 25000
npm run lc0:browser-wgsl-smokes -- --no-server --only attention-value-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only attention-output-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder0-ffn-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder0-block-ort-wasm --timeout 25000
```

## Current interpretation

The custom path now validates a complete encoder0 block in staged WGSL form, including smolgen score bias and FFN. This is a stronger milestone than the earlier attention-core-only checkpoint, but it is still not an end-to-end LC0 evaluator:

- Only encoder0 is covered; remaining transformer layers still need repeated-layer validation.
- Timing uses command submission/readback synchronization rather than GPU timestamp queries.
- The full encoder0 benchmark uses an explicit queue boundary between attention-output and FFN stages so the reported synchronized time covers the whole block.
- ORT tiny comparisons are same-value subgraph checks, not full deployment performance proof.

## Decision on full custom inference

Do **not** promote full custom LC0 inference yet.

Next gates before an end-to-end custom runtime:

1. Repeat encoder-block validation across additional encoder layers or generalized tensor names.
2. Add ORT comparisons for attention output and/or full attention block if practical.
3. Measure multiple alternating browser runs against ORT WebGPU with fresh sessions.
4. Prove layer-to-layer activation layout reuse without extra main-thread copies.
5. Preserve f32 ONNX/native parity as the correctness ladder while using f16/WebGPU as deployment target.
