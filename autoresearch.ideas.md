# Autoresearch Ideas: LC0 generated/fused f16 encoder lane

- Current best opt-in lane: `mixed-tvm-ffn-smolgen-project` with the 64-column smolgen-project tile. It reconfirmed at `5.416 ms/eval` on the 32-position suite with 9-fixture drift passing. Simple 32-column and 128-column project tile alternatives did not beat it.
- Keep every generated/fused/tiled kernel as an explicit encoder-kernel opt-in until full fixed-suite speed, drift guard, and `autoresearch.checks.sh` pass.
- Prefer generator-quality tiled matmul or fused kernels over ad-hoc scalar WGSL rewrites. Local paired-output smolgen project and simple shared-input dense1 tiling attempts were slower, while 64-column tiled smolgen project won in full fixed-suite despite noisy/slower substage timestamps.
- If future smolgen dense1 work resumes, try a materially different generated/packed schedule that reuses input/weight tiles across multiple columns, not the reverted one-output-per-lane shared-input tile, one-workgroup dense1+swish+LN1 fusion, or row-split partial-reduction shape. The fusion reduced dispatches but lost matmul parallelism; the row-split shape added too many reduction workgroups/dispatches.
- If future smolgen-project/attention-score fusion resumes, start with isolated score+bias parity. The first direct fusion failed WGSL-head drift with zero/uniform mapped policy and was reverted.
- Do not revisit adjacent current-best toggles without a new implementation shape: TVM outproj+project, TVM QKV+project, split-FFN TVM dense1-only/dense2-only, GPU/WASM legal priors, and JS/WGSL input backends all lost fast screens against the WASM-input/JS-legal project winner.
- Treat simple WebGPU int8 FFN as discarded for now: drift passed, but scalar dequant-in-inner-loop regressed fixed-suite throughput. CPU/WASM int8 can be a separate future fallback lane.
- Revisit dense2+residual+ln2 fusion only if it preserves a good tiled matmul shape; reducing dispatches is not enough if matmul gets slower.
- Revisit GPU legal/top-k only as a fused mask/softmax/top-k/readback-shrink path, not as another standalone legal-prior backend toggle.
- Rebaseline every comparison under the same pre-run browser-harness cleanup policy; do not compare cleaned runs to degraded-window controls as runtime speedups.
- Keep `batchPipelineDepth>1` and pipe2 readback strategies as speculative scheduler/readback-overlap diagnostics only; do not use them as fixed-search promotion evidence.
- Deprioritize readback byte/micro-toggles unless new attribution changes the bottleneck picture: slice/subarray/copy/unmap placement, map range, 256-byte padding, tiny dispatch fusions, and compact per-legal-move copies.
