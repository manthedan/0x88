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
- `?encoderPrefix=/encoderN`: experimental tensor-prefix override for attention-output/FFN/full-block routes so the same plumbing can target later encoder layers.

The browser page now emits a `benchmarkReport` object with browser metadata, GPU adapter info where available, pack verification mode, and timing summaries. Full encoder0 WGSL block results also include per-stage diagnostic timings for QKV projection, attention scores, softmax, attention value, output projection + ln1, FFN dense1, FFN dense2 + residual, and ln2. `scripts/lc0_browser_wgsl_smokes.mjs` automates the main browser smokes, parses `maxAbsError`, and surfaces encoder-block stage timings when present. `scripts/lc0_browser_wgsl_vs_ort_webgpu.mjs` runs fresh-session, alternating encoder0-block WGSL vs ORT WebGPU measurements and marks results as non-promotional diagnostics.

## Current evidence

Recent local Chromium/WebGPU/WASM smokes on the batch-8 f16 lc0web pack passed. These are browser smoke results, not CI guarantees:

- `npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-ffn,encoder0-block --timeout 25000`
  - `FFN_BENCH_DONE`, max absolute error about `3.34e-6`.
  - `ENCODER0_BLOCK_BENCH_DONE`, max absolute error about `3.58e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-block --timeout 25000`
  - Stage-timing smoke passed with max absolute error about `3.58e-6`.
  - One local single-iteration diagnostic sample reported stage avg timings: QKV projection `0.6 ms`, attention scores `0.4 ms`, softmax `0.3 ms`, attention value `0.2 ms`, output projection + ln1 `0.5 ms`, FFN dense1 `0.6 ms`, FFN dense2 + residual `1.0 ms`, ln2 `0.7 ms`; these include per-stage queue completion overhead and are bottleneck hints, not pure GPU timestamps.
- `npm run lc0:browser-wgsl-smokes -- --no-server --only attention-value-ort-wasm --timeout 25000`
  - `ATTENTION_VALUE_ORT_BENCH_DONE`, max absolute error about `9.54e-7`.
- `npm run lc0:browser-wgsl-smokes -- --only attention-output-ort-wasm --timeout 25000`
  - `ATTENTION_OUTPUT_ORT_BENCH_DONE`, max absolute error about `1.67e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-ffn-ort-wasm --timeout 25000`
  - `FFN_ORT_BENCH_DONE`, max absolute error about `1.91e-6`.
- `npm run lc0:browser-wgsl-smokes -- --only encoder0-block-ort-wasm --timeout 25000`
  - `ENCODER0_BLOCK_ORT_BENCH_DONE`, max absolute error about `1.97e-6`.
- `npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 2 --timeout 25000 --wgsl-iters 1 --ort-iters 2`
  - Alternated fresh browser sessions in order `wgsl, ort, ort, wgsl`.
  - ORT reported `webgpu->webgpu` with WebGPU provider accepted in both ORT samples.
  - Sample medians from that run: WGSL synchronized readback/block about `3.25 ms`, ORT WebGPU average/run about `2.55 ms`; measurement-only, not promotion evidence.

Validation commands used during this checkpoint:

```sh
npm run typecheck
TINY_LEELA_ORT_EP=wasm node --experimental-strip-types --test tests/lc0_wgsl_kernel_probe.test.mjs
npm run lc0:browser-wgsl-smokes -- --no-server --only encoder0-ffn,encoder0-block --timeout 25000
npm run lc0:browser-wgsl-smokes -- --no-server --only attention-value-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only attention-output-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder0-ffn-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-smokes -- --only encoder0-block-ort-wasm --timeout 25000
npm run lc0:browser-wgsl-vs-ort-webgpu -- --dry-run --samples 2
npm run lc0:browser-wgsl-vs-ort-webgpu -- --samples 2 --timeout 25000 --wgsl-iters 1 --ort-iters 2
```

## Current interpretation

The custom path now validates a complete encoder0 block in staged WGSL form, including smolgen score bias and FFN. This is a stronger milestone than the earlier attention-core-only checkpoint, but it is still not an end-to-end LC0 evaluator:

- Only encoder0 is covered; remaining transformer layers still need repeated-layer validation.
- Timing uses command submission/readback synchronization rather than GPU timestamp queries.
- The full encoder0 benchmark uses an explicit queue boundary between attention-output and FFN stages so the reported synchronized time covers the whole block.
- The per-stage encoder0 timing breakdown currently points first at the naive FFN dense2 + residual and ln2/QKV/dense1 kernels, but stage timings include queue-completion overhead until GPU timestamp timing is added.
- ORT tiny comparisons are same-value subgraph checks, not full deployment performance proof.

## Decision on full custom inference

Do **not** promote full custom LC0 inference yet.

Next gates before an end-to-end custom runtime:

1. Repeat encoder-block validation across additional encoder layers using `?encoderPrefix=/encoderN` once later-layer activation handoff is implemented.
2. Add ORT comparisons for the remaining full attention block if practical.
3. Repeat and broaden alternating browser runs against ORT WebGPU after validating more than encoder0.
4. Prove layer-to-layer activation layout reuse without extra main-thread copies.
5. Preserve f32 ONNX/native parity as the correctness ladder while using f16/WebGPU as deployment target.
