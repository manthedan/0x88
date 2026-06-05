# Autoresearch Ideas: LC0 WGSL readback b4

- Productize/document the current opt-in candidate: WASM input + `mixed-tvm-ffn` + JS legal priors + b4/depth1, with stable defaults unchanged and benchmark hygiene called out explicitly.
- Rebaseline any future comparisons under the same pre-run browser-harness cleanup policy; do not compare cleaned runs to degraded-window controls as runtime speedups.
- Start a separate quantized/int8 FFN or encoder lane against the recovered-state `mixed-tvm-ffn` JS-legal baseline. Require full fixed-suite ms/eval, parity/top-k/value drift checks, explicit opt-in wiring, and no default changes.
- Keep GPU legal priors as an opt-in scaffold/correctness path only; byte reduction (`7444` -> `3084`) did not beat JS legal priors under adjacent recovered-state controls.
- Keep `batchPipelineDepth>1` and pipe2 readback strategies as speculative scheduler/readback-overlap diagnostics only; do not use them as fixed-search promotion evidence.
- Deprioritize readback byte/micro-toggles unless new attribution changes the bottleneck picture: slice/subarray/copy/unmap placement, map range, 256-byte padding, tiny dispatch fusions, and compact per-legal-move copies.
- Deprioritize batch-size sweeps (`b2/b8/b16`) for this lane unless a stronger alternating matrix shows a new regime.
- If returning to readback diagnostics, prefer matrix/attribution runs that explain E2E full-search behavior rather than isolated microbench wins.
