# t3 onboarding, ORT QDQ int8, and the WebNN probe (2026-06-10)

Three lanes run together; all artifacts in `artifacts/tvm/` and
`public/models/lc0/` per existing conventions.

## 1. t3-512x15x16h-distill onboarded (the ladder's mid-rung)

Full pipeline (same as BT4-it332): net + sha + describenet (512×15×16h,
attention body, smolgen swish — same family), leela2onnx b1/4/8/16 f16,
native BLAS fixture baseline
(`fixtures/lc0/native_fen_only_blas.t3-512x15x16h-distill-swa-2767500.jsonl`),
detached-params TVM probe (4 batches, one shared 162 MB sidecar), staged at
`public/runtimes/lc0-tvmjs-webgpu/t3-512x15x16h-distill-swa-2767500/f16/v1`.

Cleanest gate of any net so far (`t3_tvmjs_smoke_b8_v16_ortf16.json`):
**native 8/8 STRICT** (max prior diff 0.0044 — no tie tolerance needed),
ORT f16 8/8, search 8/8; TVMJS 119.4 ms vs ORT 129.8 ms at v16/b8. The
progressive ladder now reads: tiny → t1 (~40 ms) → t3 (~120, ~60-90 expected
with tree reuse) → BT4 (~250/137). Follow-ups run same day:

- **Batch sweep: b8 wins for t3** (`t3_tvmjs_b16_v16.json`: b16 140.0 ms vs
  b8 119.4 ms at v16, native 8/8). t1's b16 advantage does not transfer —
  512-wide already saturates the GPU near b8, consistent with BT4 saturating
  at b4. (That artifact's `ok:false` is a harness artifact: the run omitted
  `--ort-model`, so its search comparison ran against the default t1 ORT
  model; the timing and native-parity numbers are valid.)
- **Tree-reuse game A/B** (`t3_tvmjs_game_reuse_ab_v16_p12.json`): fresh
  114.8 → reused **73.8 ms/move (−36 %)**, 12 plies v16/b8, agreement 8/12.
  In-game ladder: ~t1 40 / t3 74 / BT4 137 ms per move.
- Still open: footprint sidecar for the t3 staging.

## 2. int8 QDQ ONNX for the ORT lane (stable-runtime download win)

`scripts/lc0_quantize_onnx_weights_qdq.py`: MatMul B initializers >4096
elements → per-output-channel int8 + f16 scales + in-graph DequantizeLinear
(opset bumped 17→21 for f16 DQL). Compute stays f16; the ORT-lane twin of the
TVMJS detached-int8 sidecar.

- t1 b8: 40.4 → 20.6 MB (0.506). Browser gate (TVMJS-f16 vs ORT-QDQ on real
  fixtures): eval 8/8, search 8/8, drift 0.021. ~15-20 % slower inference
  (in-graph dequant runs per eval).
- BT4-it332 b8: 370.5 → **186.7 MB (0.503)**. Gate: eval 6/8 + 2 tolerated
  ties (max drift 0.0106 — tighter than the TVMJS int8 sidecar's 0.0172),
  search within tolerance. Timing contaminated by a concurrent compile;
  quiet rerun pending.
- Caution: random-input node-CPU comparisons show large drift (rel RMS ~0.12)
  — out-of-distribution inputs amplify int8 error; judge QDQ only on the
  real-fixture harness.

This de-risks TVM dependence: the stable ORT lane now has its own ~2×
download reduction.

## 3. ORT WebNN EP probe (`lc0-ort-webnn-probe.html`) — big, with a catch

WebNN requires Chrome launched with
`--enable-features=WebMachineLearningNeuralNetwork` (agent-browser:
`--args`). With it, `navigator.ml` contexts create for cpu/gpu/npu on Apple
Silicon (CoreML; npu = ANE). Probe = same model + same input through
webnn EP and webgpu EP, 10-20 iters.

| Model | webnn(gpu) ms/eval | probe webgpu EP | numerics vs webgpu |
| --- | ---: | ---: | --- |
| t1-256x10 f16 b8 | **2.9** (npu 3.5) | 30.6 | clean (policy 0.047) |
| t3-512x15 f16 b8 | 12.5 | 445 | **BROKEN** (wdl ~0.93) |
| BT4-1024 f16 b8 | 27.4 | 528 | **BROKEN** (wdl ~0.99) |
| t3-512x15 **f32** b8 | **30.0** | 414 | **near-exact** (2e-5) |

Findings:
- **CoreML breaks f16 LC0 graphs beyond t1 size** (silent garbage — looks
  like f16 accumulation overflow in the attention/smolgen logits; t1 squeaks
  under). f32 is correct.
- Session create = CoreML compile: 7-52 s per model (cache behavior unknown).
- The probe's webgpu-EP numbers are far worse than the production
  `Lc0OnnxEvaluator` path (no preferredOutputLocation etc.) — do NOT read
  the ratio as 10-19×. Honest comparison points: WebNN-f32 t3 ≈ 30 ms/b8
  ≈ TVMJS-f16 t3 per-batch; WebNN t1 ~3 ms vs TVMJS t1 ~14 ms (**4-5×** on
  small nets — ANE-class hardware really is faster there).
- Synthesis worth pursuing: **QDQ int8 storage + DQL to f32 + WebNN EP** —
  half-of-f16 download AND overflow-safe f32 compute AND CoreML speed.
- Not shippable until WebNN rides without a flag; track Chrome's rollout.
  This is also effectively our first non-WebGPU coverage datapoint.
