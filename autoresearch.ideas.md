# Autoresearch Ideas: LC0 generated/fused f16 encoder lane

- Current highest-ROI lane: generated/tiled/packed-f16 smolgen kernels. The first keep is `mixed-tvm-ffn-smolgen-project`; next target is smolgen dense1 (`2048 -> 256`) and then a combined smolgen-project+dense1 variant if dense1 wins independently.
- Keep every generated/fused/tiled kernel as an explicit encoder-kernel opt-in until full fixed-suite speed, drift guard, and `autoresearch.checks.sh` pass.
- Prefer generator-quality tiled matmul or fused kernels over ad-hoc scalar WGSL rewrites. Local paired-output smolgen project and shared-input dense1 tiling attempts were slower, while 64-column tiled smolgen project won in full fixed-suite despite noisy/slower substage timestamps.
- Treat simple WebGPU int8 FFN as discarded for now: drift passed, but scalar dequant-in-inner-loop regressed fixed-suite throughput. CPU/WASM int8 can be a separate future fallback lane.
- Revisit dense2+residual+ln2 fusion only if it preserves a good tiled matmul shape; reducing dispatches is not enough if matmul gets slower.
- Revisit GPU legal/top-k only as a fused mask/softmax/top-k/readback-shrink path, not as another standalone legal-prior backend toggle.
- Rebaseline every comparison under the same pre-run browser-harness cleanup policy; do not compare cleaned runs to degraded-window controls as runtime speedups.
- Keep `batchPipelineDepth>1` and pipe2 readback strategies as speculative scheduler/readback-overlap diagnostics only; do not use them as fixed-search promotion evidence.
- Deprioritize readback byte/micro-toggles unless new attribution changes the bottleneck picture: slice/subarray/copy/unmap placement, map range, 256-byte padding, tiny dispatch fusions, and compact per-legal-move copies.
