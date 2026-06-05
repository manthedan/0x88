# Autoresearch Ideas: LC0 quantized/int8 FFN/encoder lane

- First candidate: explicit opt-in FFN int8 kernel variant with per-output-channel symmetric weight scales and WGSL dequant inside matmul. Keep baseline path unchanged.
- Try dense1-only and dense2-only int8 variants if full FFN int8 either drifts or loses throughput to dequant overhead.
- Add quantized drift probes before full-search runs if a candidate risks large policy/value changes; fail fast on top-prior or WDL drift.
- Rebaseline every comparison under the same pre-run browser-harness cleanup policy; do not compare cleaned runs to degraded-window controls as runtime speedups.
- Keep GPU legal priors as an opt-in scaffold/correctness path only; byte reduction (`7444` -> `3084`) did not beat JS legal priors under adjacent recovered-state controls.
- Keep `batchPipelineDepth>1` and pipe2 readback strategies as speculative scheduler/readback-overlap diagnostics only; do not use them as fixed-search promotion evidence.
- Deprioritize readback byte/micro-toggles unless new attribution changes the bottleneck picture: slice/subarray/copy/unmap placement, map range, 256-byte padding, tiny dispatch fusions, and compact per-legal-move copies.
