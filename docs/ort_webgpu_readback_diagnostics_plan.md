# ORT WebGPU Readback Diagnostics Plan

## Goal

Understand why ONNX Runtime WebGPU often looks better than the custom LC0 WGSL path at end-to-end search, especially around command submission, output download, `mapAsync`, and readback/fence behavior.

This is a diagnostic lane, not a runtime replacement. The output should tell us whether to invest next in scheduler/readback changes, generated/TVM kernels, graph-level fusion, or compact legal-prior/top-k readback.

## Why this matters

The custom WGSL path currently exposes detailed telemetry and LC0-specific levers, but fast-loop experiments show many apparent wins are dominated by noisy WebGPU fence/readback time. ORT WebGPU is a useful comparison point because it runs the full ONNX graph through a mature WebGPU backend and may differ in:

- command-buffer batching;
- dispatch count and fusion;
- buffer reuse;
- output download timing;
- whether `session.run()` waits for CPU-visible outputs;
- GPU kernel shape choices.

## What to inspect in ORT

Installed source paths worth reading/breakpointing:

- `node_modules/onnxruntime-web/lib/wasm/jsep/backend-webgpu.ts`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/gpu-data-manager.ts`
- `node_modules/onnxruntime-web/lib/wasm/wasm-core-impl.ts`

The standard ORT WebGPU download path appears to be:

```text
create MAP_READ | COPY_DST read buffer
copyBufferToBuffer(outputGpuBuffer -> readBuffer)
flush/submit command encoder
await readBuffer.mapAsync(GPUMapMode.READ)
getMappedRange()
copy/clone bytes into CPU typed array
destroy read buffer
```

That is close enough to our WGSL readback path that we can compare mechanics directly.

## Implemented diagnostic mode

The opt-in query mode is:

```text
ortReadbackProfile=1
```

It enables, unless explicitly overridden:

```text
ortWebGpuProfile=1
ortMonkeyPatchWebGpu=1
ortPreferredOutputLocation=gpu-buffer
```

The convenience runner emits explicit `=0` overrides for `--no-kernel-profile`, `--no-monkey-patch`, and `--no-gpu-outputs`, so those flags really disable the default `ortReadbackProfile=1` subfeatures.

A convenience runner is available:

```bash
npm run lc0:browser-ort-readback-profile -- --iters 5 --warmup 1
```

The mode creates an ORT WebGPU session with:

```ts
preferredOutputLocation: 'gpu-buffer'
```

and times these phases separately in LC0 ONNX evaluator results:

```text
sessionRunMs / ortRunMs   // ORT graph execution returning GPU-backed outputs
ortPolicyGetDataMs        // explicit policy tensor download
ortWdlGetDataMs           // explicit WDL tensor download
ortMlhGetDataMs           // explicit MLH tensor download
ortAllGetDataMs           // Promise.all of required downloads
postprocessMs             // legal-prior and search output prep
totalEvalMs               // encode + ORT run + downloads + postprocess
```

This separates GPU graph execution from CPU-visible output download.

## ORT WebGPU profiling hook

The diagnostic mode enables ORT's WebGPU timestamp profiling:

```ts
ort.env.webgpu.profiling.mode = 'default';
ort.env.webgpu.profiling.ondata = (data) => {
  // collect kernelName, programName, input/output metadata, startTime, endTime
};
```

Aggregated numeric fields currently emitted into evaluator timing include:

```text
ortKernelCount
ortKernelGpuMs
```

Note: ORT only emits timestamp records when the browser/device exposes the required WebGPU timestamp-query features. On devices without those features, these fields can remain zero even while the WebGPU API monkey-patch counters are working.

## WebGPU API monkey-patch

Before ORT initializes, diagnostic mode wraps WebGPU methods for black-box counts:

```text
GPUQueue.submit
GPUBuffer.mapAsync
GPUCommandEncoder.copyBufferToBuffer
GPUDevice.createBuffer
GPUDevice.createComputePipeline
GPUDevice.createComputePipelineAsync
```

Emitted numeric timing fields include:

```text
webgpuSubmitCount
webgpuSubmittedCommandBufferCount
webgpuMapAsyncCount
webgpuMapAsyncMs
webgpuCopyBufferToBufferCount
webgpuCopyBufferToBufferBytes
webgpuMapReadBufferCount
webgpuMapReadBufferBytes
webgpuCreateBufferCount
webgpuCreateBufferBytes
webgpuComputePipelineCreateCount
webgpuComputePipelineCreateAsyncCount
```

Keep this behind the explicit diagnostic flag because monkey-patching can perturb timings.

## Comparison questions

1. Does `session.run()` still wait for completion when outputs are GPU-backed?
2. How much latency is added by `policy.getData()` vs WDL/MLH download?
3. Does ORT use fewer queue submits/maps than the custom WGSL path?
4. How many GPU kernels does ORT run for LC0, and which programs dominate GPU time?
5. Is ORT's advantage mainly graph/kernel execution, or output download/fence behavior?
6. Does ORT benefit from graph capture or persistent GPU outputs in a way we can emulate?

## Implementation locations

Primary files:

- `src/nn/ortRuntime.ts` — opt-in session options, ORT WebGPU profiling hook, WebGPU API instrumentation snapshots.
- `src/lc0/onnxEvaluator.ts` — timing hooks around `session.run()` and output `getData()`.
- `src/lc0/searchWorker.ts` and `src/lc0/policyOnlyBrowser.ts` — pass diagnostic options into worker-owned ORT sessions.
- `scripts/lc0_browser_ort_readback_profile.mjs` — convenience browser runner.
- `docs/lc0web_custom_inference_checkpoint.md` — summarize findings after larger repeated runs.

Avoid changing stable defaults. ORT ONNX/WebGPU should remain the arena baseline unless repeated E2E evidence says otherwise.

## Promotion implications

Interpret results as follows:

- If ORT graph execution is fast but `getData()` dominates, prioritize scheduler/readback overlap and compact legal-prior/top-k readback.
- If ORT kernels are much faster or fewer, prioritize targeted TVM/generated WGSL or operator fusion lanes.
- If ORT uses fewer submits/maps, prioritize command-buffer consolidation and batch/evaluate sequence changes.
- If ORT's advantage disappears when outputs are GPU-backed and downloads are isolated, readback is the main shared bottleneck.

## First local smoke observations

These are sanity checks, not promotion evidence. They used short 3-iteration browser runs on the current development host.

With explicit GPU-backed outputs:

```bash
node scripts/lc0_browser_ort_readback_profile.mjs --iters 3 --warmup 1
```

observed roughly:

```text
avgMs                         30.96
mean ortRunMs                  7.45
mean ortAllGetDataMs          22.99
webgpuSubmitCount             28 per eval
webgpuMapAsyncCount            3 per eval
webgpuCopyBufferToBufferCount  4 per eval
webgpuMapReadBufferBytes    7472 per eval
```

With default CPU-visible outputs, but still instrumented:

```bash
node scripts/lc0_browser_ort_readback_profile.mjs --iters 3 --warmup 1 --no-gpu-outputs
```

observed roughly:

```text
avgMs                         19.36
mean ortRunMs                 18.79
mean ortAllGetDataMs           0.02
webgpuSubmitCount             27 per eval
webgpuMapAsyncCount            3 per eval
webgpuCopyBufferToBufferCount  4 per eval
webgpuMapReadBufferBytes    7472 per eval
```

Interpretation:

- `preferredOutputLocation=gpu-buffer` appears to move download/fence cost from `session.run()` into explicit `tensor.getData()`, as intended.
- In this short noisy run, forcing GPU-backed outputs was slower end-to-end because downstream JS still immediately downloads all three outputs.
- ORT still maps three output tensors and copies about 36 KB through `copyBufferToBuffer` per eval, even though the CPU-visible model outputs are only about 7.3 KB.
- ORT timestamp kernel profiling remained unavailable on this browser/device (`ortKernelCount=0`), so WebGPU API counters are the reliable attribution signal here.

## Fixed-FEN strategy matrix

A reusable fixed-FEN comparison runner now exists:

```bash
npm run lc0:browser-readback-strategy-matrix -- \
  --out /tmp/lc0_readback_strategy_matrix.json \
  --max-positions 4 --repeats 2
```

It compares:

- `ort-cpu` — ORT WebGPU with default CPU-visible outputs, still instrumented.
- `ort-gpu` — ORT WebGPU with `preferredOutputLocation=gpu-buffer` and explicit `tensor.getData()` timing.
- `wgsl-pipe1` — custom WGSL encoder + WGSL heads with physical batching and normal search scheduling.
- `wgsl-gpu-legal` — same path with GPU legal-prior filtering, reducing per-position readback bytes from full mapped policy to legal-move triples + WDL.
- `wgsl-pipe2` — same custom WGSL path with `batchPipelineDepth=2`. The matrix also exposes `--pipe2-batch` as an override/cap for overlap experiments; the default now matches `--batch 4` after deferred resources are preallocated before submitting pipelined work.
- `wgsl-gpu-legal-pipe2` — combined interaction test: GPU legal-prior compact readback plus `batchPipelineDepth=2`, using the same `--pipe2-batch` cap.

A short local two-FEN smoke was run:

```bash
npm run lc0:browser-readback-strategy-matrix -- \
  --out /tmp/lc0_readback_strategy_matrix_2fen.json \
  --max-positions 2 --repeats 1 \
  --strategies ort-cpu,ort-gpu,wgsl-pipe1 \
  --ort-iters 3 --ort-warmup 1 \
  --wgsl-eval-iters 2 --wgsl-search-iters 2 --wgsl-search-warmup 1 \
  --visits 32 --batch 4
```

Median smoke summary:

```text
ort-cpu avgMs                 13.01
ort-cpu ortRunMs              12.28
ort-cpu ortAllGetDataMs        0.02
ort-cpu MAP_READ bytes      7472

ort-gpu avgMs                 14.41
ort-gpu ortRunMs               8.13
ort-gpu ortAllGetDataMs        5.44
ort-gpu MAP_READ bytes      7472

wgsl-pipe1 evalMeanMs         14.25
wgsl-pipe1 searchMeanMs      306.47  // 32 visits, batch 4
wgsl-pipe1 visits/s          104.41
wgsl-pipe1 eval readbackMs    10.40
wgsl-pipe1 search readbackMs  24.48
```

A follow-up two-FEN smoke compared normal WGSL readback with existing GPU legal-prior readback:

```text
wgsl-pipe1 eval bytes          7444
wgsl-pipe1 search bytes       27295
wgsl-pipe1 search readbackMs  24.08
wgsl-pipe1 visits/s          106.67

wgsl-gpu-legal eval bytes      3084
wgsl-gpu-legal search bytes   11308
wgsl-gpu-legal search readbackMs 24.61
wgsl-gpu-legal visits/s       104.73
```

So GPU legal-prior readback reduced bytes by roughly 59% but did not reduce readback wall time in this short run. That points at fixed `mapAsync`/queue-drain/fence cost rather than raw transfer size for the current small outputs.

The original unbounded `wgsl-pipe2` smoke with batch=4/depth=2 did not complete usefully. After changing the harness to recognize `HYBRID_SEARCH_BENCH_FAILED` immediately, it surfaced deterministic-looking WGSL-head validation failures such as:

```text
WGSL hybrid batch heads produced zero or uniform mapped policy for slot 3,
mode deferred-double-buffered, batch 1,
fen rnbqkbnr/p4ppp/1pp1p3/3pP3/3P3P/2N2N2/PPP2PP1/R1BQKB1R b KQkq h3 0 5
```

The first attempted fix split deferred compute/upload resource banks per readback ring slot, but local batch=4/depth=2 search still hit the slot-3 validation failure. The stabilizing fix was to preallocate both deferred ring slots' compute, upload, and readback resources to the maximum batch length before any pipelined command buffer is submitted. That avoids lazy creation/growth of the second ring slot while the first deferred batch is already in flight.

A two-FEN smoke including ORT, GPU legal-prior readback, and restored batch=4/depth=2 pipe2 was run:

```bash
npm run lc0:browser-readback-strategy-matrix -- \
  --out /tmp/lc0_readback_matrix_all_pipe2_fixed.json \
  --max-positions 2 --repeats 1 \
  --strategies ort-cpu,ort-gpu,wgsl-pipe1,wgsl-gpu-legal,wgsl-pipe2 \
  --ort-iters 2 --ort-warmup 1 \
  --wgsl-eval-iters 1 --wgsl-search-iters 1 --wgsl-search-warmup 0 \
  --visits 16 --batch 4 --pipe2-batch 4
```

Median smoke summary:

```text
ort-cpu avgMs                 16.06
ort-gpu avgMs                 12.26  // diagnostic-only; short/noisy run

wgsl-pipe1 batch/depth         4 / 1
wgsl-pipe1 searchMeanMs      257.76
wgsl-pipe1 visits/s           62.13
wgsl-pipe1 search readbackMs  35.16
wgsl-pipe1 search bytes       25310

wgsl-gpu-legal batch/depth     4 / 1
wgsl-gpu-legal searchMeanMs  244.45
wgsl-gpu-legal visits/s       65.47
wgsl-gpu-legal search readbackMs 35.88
wgsl-gpu-legal search bytes   10486

wgsl-pipe2 batch/depth         4 / 2
wgsl-pipe2 searchMeanMs      299.73
wgsl-pipe2 visits/s           54.04
wgsl-pipe2 search readbackMs  35.92
wgsl-pipe2 search bytes       23200
```

Immediate implication: batch=4/depth=2 is now a valid matrix cell again, but this short local smoke did not show an end-to-end win. The earlier batch=2/depth=2 signal still suggests readback/fence overlap can matter, but the scheduler/resource path needs repeated fixed-suite runs before any promotion. GPU legal-prior byte reduction can now be tested directly with overlap via `wgsl-gpu-legal-pipe2` before any promotion decision.

Combined GPU-legal + pipe2 smoke results:

```bash
npm run lc0:browser-readback-strategy-matrix -- \
  --out /tmp/lc0_local_combined_readback_pipe2b2_smoke.json \
  --max-positions 2 --repeats 1 \
  --strategies wgsl-pipe1,wgsl-gpu-legal,wgsl-pipe2,wgsl-gpu-legal-pipe2 \
  --wgsl-eval-iters 1 --wgsl-search-iters 1 --wgsl-search-warmup 0 \
  --visits 16 --batch 4 --pipe2-batch 2

npm run lc0:browser-readback-strategy-matrix -- \
  --out /tmp/lc0_local_combined_readback_pipe2b4_smoke_rerun.json \
  --max-positions 2 --repeats 1 \
  --strategies wgsl-pipe1,wgsl-gpu-legal,wgsl-pipe2,wgsl-gpu-legal-pipe2 \
  --wgsl-eval-iters 1 --wgsl-search-iters 1 --wgsl-search-warmup 0 \
  --visits 16 --batch 4 --pipe2-batch 4
```

Median smoke summary:

```text
pipe2-batch 2:
  wgsl-pipe1              visits/s 60.64  searchMs 267.01  readbackMs 34.79  bytes 25310
  wgsl-gpu-legal          visits/s 45.88  searchMs 370.82  readbackMs 41.16  bytes 10486
  wgsl-pipe2              visits/s 47.92  searchMs 389.40  readbackMs 28.83  bytes 14061
  wgsl-gpu-legal-pipe2    visits/s 44.09  searchMs 374.32  readbackMs 24.76  bytes  5825

pipe2-batch 4:
  wgsl-pipe1              visits/s 41.96  searchMs 384.65  readbackMs 40.91  bytes 25310
  wgsl-gpu-legal          visits/s 27.38  searchMs 585.09  readbackMs 78.86  bytes 10486
  wgsl-pipe2              visits/s 22.53  searchMs 720.51  readbackMs 86.53  bytes 23200
  wgsl-gpu-legal-pipe2    visits/s 48.71  searchMs 331.53  readbackMs 38.00  bytes  9612
```

The combined strategy is therefore worth keeping in the matrix: with `pipe2-batch=2` it reduced bytes/readback wait but did not beat pipe1 E2E; with `pipe2-batch=4` it was the best cell in this noisy two-FEN smoke. Treat this only as a hypothesis for repeated fixed-suite/autoresearch validation, not as promotion evidence.

## Guardrails

- Do not use browser diagnostic numbers from one noisy run as promotion evidence.
- Do not promote `preferredOutputLocation: 'gpu-buffer'` into normal ONNX evaluator behavior until all downstream consumers explicitly handle GPU tensors.
- Keep monkey-patching disabled by default.
- Preserve LC0 native/BLAS and f32 ONNX parity ladders.
