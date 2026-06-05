# Autoresearch Ideas: LC0 WGSL readback b4

- Compact WGSL-head readback: avoid full mapped-policy readback when only legal priors/top candidates are needed.
- Revisit opt-in GPU legal priors with stronger lifecycle checks and compare `readbackBytes`, map count, and E2E eval/s against JS legal priors.
- Split WDL and policy readback timing/shape to determine whether WDL-only or policy-only reads can avoid some fence cost.
- Add a focused microbench for policy readback bytes vs map/fence latency, but require E2E fixed-suite confirmation before keeping complex changes.
- Improve telemetry for GPU completion vs CPU copy: timestamp queries or staged command submission only if it informs E2E search changes.
- Consider WASM legal-prior candidate/probability prep only if telemetry shows JS legal-prior postprocess becomes material after readback reductions.
- Revisit generated/TVM/f16 kernels only after readback/fence telemetry stops dominating total eval time.
