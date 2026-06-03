# lc0web custom WebGPU inference checkpoint

This checkpoint records the current state after moving from a single packed-model tensor probe to the first repeated LC0 sub-block.

## Implemented probes

- `?kernelBench=1`: queues many fixed-shape WGSL `MatMul + Add` dispatches and reads back once for correctness.
- `?ortOpBench=1`: compares the same extracted `MatMul + Add` values through a tiny generated ORT ONNX graph.
- `?kernelVariant=tiled16`: runs a 16x16 workgroup-reduction WGSL variant.
- `?kernelVariant=scalar-transposed`: runs the scalar kernel against a transposed temporary weight layout.
- `?qkvProbe=1`: runs encoder0 Q/K/V projection as one repeated LC0 sub-block using packed lc0web tensors.

## Current evidence

Manual Chromium/WebGPU smokes on the batch-8 f16 lc0web pack passed. These are browser smoke results, not CI guarantees:

- `?kernelBench=1&kernelVariant=tiled16&kernelBenchWarmup=2&kernelBenchIters=50&packVerify=0`
  - status: `KERNEL_BENCH_DONE`
  - max absolute error: `3.874302e-7`
- `?kernelBench=1&kernelVariant=scalar-transposed&kernelBenchWarmup=2&kernelBenchIters=50&packVerify=0`
  - status: `KERNEL_BENCH_DONE`
  - max absolute error: `1.66893e-6`
- `?ortOpBench=1&ep=webgpu&ortBenchWarmup=1&ortBenchIters=5&packVerify=0`
  - status: `ORT_BENCH_DONE`
  - average run time: about `0.9 ms`
  - max absolute error: `1.192093e-6`
- `?qkvProbe=1&qkvWarmup=1&qkvIters=3&packVerify=0`
  - status: `QKV_DONE`
  - Q max absolute error: `1.66893e-6`
  - K max absolute error: `1.66893e-6`
  - V max absolute error: `9.536743e-7`

Validation commands passed:

```sh
npm run typecheck
TINY_LEELA_ORT_EP=wasm node --experimental-strip-types --test tests/lc0_wgsl_kernel_probe.test.mjs
npm run build:client
```

## Decision on full custom inference

Do **not** start full custom LC0 inference yet.

The packed-tensor custom WebGPU path is now correctness-validated on the tested batch-8 f16 lc0web pack for a single projection and the encoder0 Q/K/V projection sub-block. It does not validate a full attention block, repeated layers, or end-to-end inference. The next performance-sensitive operations are not implemented or benchmarked yet:

1. QK attention-score matmul.
2. Scaling and mask handling.
3. Softmax.
4. Attention-value matmul.
5. Output projection.
6. Residual and normalization operations.
7. Layer-to-layer activation layout and cache reuse.

The next build-out should extend from `Q/K/V projections` to the attention core before attempting an end-to-end custom runtime. Full inference should only be promoted when a browser benchmark shows repeated-block performance competitive with ORT and parity remains within tolerance across representative fixtures.
