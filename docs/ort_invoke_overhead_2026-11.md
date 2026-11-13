# ORT WebGPU per-invoke overhead for Lc0 Small (November 2026)

How much of an ORT-web 1.27 WebGPU invoke of Lc0 Small at batch 1 is CPU work
and sync, not GPU work, and how much of it can the app remove without
changing outputs?

**Verdict: ORT graph capture with GPU-resident IO is the lever.** All outputs
stay bit-identical, and every search compared gave the same result. The
numbers below are paired ratios (lower is faster).

| machine, model | invoke wall, capture / shipped | search time, capture / shipped |
|---|---|---|
| RTX 3090, plain qdq8 | **0.554** (MAD 0.038, p10-p90 0.477-0.603) | **0.560** (0.013, 0.503-0.578) |
| Apple M4, split-K (the Apple default) | 0.770 (0.061, 0.455-0.837) | **0.541** (0.026, 0.433-0.573) |
| Apple M4, plain qdq8 | 0.858 (0.028, 0.807-0.926) | **0.649** (0.079, 0.568-0.860) |

Search time is a fixed 256-visit search, so 0.56 means ~1.8x the search
throughput. Invoke rows: n=40 paired blocks; search rows: n=30 (3090) and
n=40 (M4) paired searches.

The machines were not loaded the same way. The 3090 runs went through the shared
gpu-queue and are `timingValid` (not contended, verified). The M4 was shared
with other agents: load average 55-100 during the invoke runs and 5-35 during
the search runs. On a lightly loaded M4 the GPU is busy for most of an invoke
(§1), so the idle-machine M4 gain is probably smaller than these numbers; it
was not measured. On the M4 the search-level gain is larger than the
invoke-level gain. The two ran at different load, in different contexts (page
main thread vs the search worker), and the cause was not isolated.

The branch `perf/ort-invoke-overhead` makes this the search-worker default for
the device-selected Lc0 Small model (§5). Nothing is merged or released.

Setup: headless Chrome 157 on the Apple M4 (10-core GPU, `apple / metal-3`),
Chrome for Testing 151 on Yukon (RTX 3090, Vulkan, runner default flags
including `vulkan_enable_f16_on_nvidia`). The production `dist-client` plus the
lab overlay, served by `scripts/serve_isolated_static.mjs` (M4) or the remote
runner (Yukon). M4 timings ran inside `notes/kernel-hillclimb/benchlock.sh`.
ORT-web 1.27, JSPI build, native C++ WebGPU EP.

## 1. Where an invoke's time goes

`lab/kernel-hillclimb/ort-overhead.html` wraps the WebGPU device the ORT session
uses and timestamps every call ORT makes during one `session.run` (trace
medians of 30 invokes, batch 1):

| | M4 qdq8 | M4 split-K | 3090 qdq8 |
|---|---|---|---|
| wall per invoke (ms) | 9.40 | 11.54 | 8.65 |
| dispatches / submits | 356 / 26 | 595 / 41 | 356 / 26 |
| createBindGroup + uniform writeBuffer | 356 + 356 | 595 + 595 | 356 + 356 |
| last compute submit at (ms) | 5.56 | 9.42 | 6.61 |
| readback chain: 3 copies, 3 serial mapAsync (ms) | 3.66 | 1.90 | 1.70 |
| after the last map resolves (ms) | 0.18 | 0.17 | 0.20 |
| GPU busy per invoke, timestamp queries (ms) | 8.6 | 12.1 | 3.2 |

(M4 columns: the lighter-loaded run `full2`; the 3090 column: the Yukon run.)

- **Encoding is CPU-bound and serial.** ORT encodes each kernel as its own
  dispatch with a fresh bind group and a fresh uniform buffer write: about
  15 µs of CPU per dispatch on the M4 and about 18 µs on the 3090. The last
  compute submit lands 5.6-9.4 ms into the invoke.
- **On the 3090 the GPU waits on the CPU.** It is busy 3.2 ms of an 8.7 ms
  invoke. On the M4 the GPU is busy for most of the invoke (8.6 of 9.4 ms),
  and the encode mostly overlaps GPU work. That is why the M4 gain is smaller
  and depends on load: when other processes slow the CPU, encode becomes the
  critical path again.
- **Readback is three serial round trips.** One copy and one `mapAsync` per
  output (policy, wdl, mlh).
- **The app's own JS is small.** Around the run the evaluator spends ~0.3 ms:
  input build 0.04, getData 0.12, legal priors 0.035, postprocess 0.10.
  Preallocating the input tensor or the output arrays has nothing to win.

## 2. Levers (paired ABAB, invoke wall vs the shipped path)

Each lever was measured in paired ABAB blocks (40 pairs, 5 invokes per block,
20 warm-up invokes) against the shipped CPU-IO path in the same page. Every
variant reproduced the shipped outputs bit-for-bit on 10 fixtures x 2 passes.

| lever | what it removes | M4 qdq8 | M4 split-K | 3090 qdq8 |
|---|---|---|---|---|
| `gpuout`: `preferredOutputLocation: 'gpu-buffer'` + `getData()` | nothing per invoke (same 3 maps) | 0.926 (0.027) | 0.976 (0.086) | 0.875 (0.043) |
| `iobind`: persistent input/output GPU buffers, one staging copy, one map | 4 buffer creations, 2 of 3 maps | 0.932 (0.039) | 0.926 (0.089) | 0.793 (0.048) |
| `capture`: `iobind` + ORT graph capture | all 356/595 bind groups and uniform writes; encode | **0.858** (0.028) | **0.770** (0.061) | **0.554** (0.038) |

Values are median ratio (MAD). With capture the last compute submit moves to
1.0 ms (M4 qdq8), 1.3 ms (M4 split-K) and 0.7 ms (3090), and per invoke ORT
makes 0 `createBindGroup` calls and 1 `writeBuffer` (the input).

On the M4, an earlier run (`full2`, lower load) gave capture/shipped 0.892 for
qdq8. That run used the string-form provider (§4), which turned out not to
capture anything, so the gain there came from IO binding alone.

## 3. Search level

`lab/kernel-hillclimb/ort-search.html` starts one built product search worker
(`searchWorker-*.js`) per mode and runs the same 256-visit, batch-1 search on 10
fixture positions. Before every search it clears the tree and the eval cache.
Each (round, position) row runs all modes in rotating order; the ratio is per
row against the CPU-IO worker in that row. Search results (best move, visits,
PV, value) must match across modes, and did in every row.

Median search time (ms) per mode, and paired ratios against CPU IO, as
median (MAD; p10-p90):

| machine, model | n | CPU IO (shipped) | GPU IO | GPU IO + capture | GPU IO / CPU IO | capture / CPU IO |
|---|---|---|---|---|---|---|
| 3090, qdq8 | 30 | 2252 | 1736 | 1269 | 0.767 (0.020; 0.683-0.796) | **0.560** (0.013; 0.503-0.578) |
| M4, split-K | 40 | 3600 | 3294 | 1942 | 0.902 (0.049; 0.795-1.010) | **0.541** (0.026; 0.433-0.573) |
| M4, qdq8 | 40 | 4521 | 4058 | 2897 | 0.879 (0.091; 0.711-1.019) | **0.649** (0.079; 0.568-0.860) |

An earlier M4 run of the same page (n=20, 2 rounds, load 7-55) gave capture /
CPU IO 0.504 (0.043) for split-K and 0.683 (0.048) for qdq8, consistent with
the table. The M4 CPU-IO medians have large MADs (360-500 ms) because of load.

All three workers in a row serve the same model (the page checks the ready
message). Each search makes 257 evaluations at batch 1, so on the 3090 a
search-level evaluation goes from ~8.8 ms to ~4.9 ms.

## 4. ORT-web 1.27 behaviour this depends on

- **IO buffers must be pinned.** `webgpuRegisterBuffer` imports the caller's
  `GPUBuffer` and stores `[handle, refcount]` under a private symbol on it.
  After each run ORT unregisters the IO tensors; at refcount 0 it releases the
  Dawn handle (the JS buffer stays alive). The next run imports a new handle.
  Graph capture records the first run's handles, so replays then bind released
  buffers. Without pinning, capture aborts with `RuntimeError: unreachable`.
  `pinOrtGpuBufferImport` (`src/nn/ortGpuIo.ts`) intercepts that record and
  marks it like ORT's own output buffers (`NaN` count, never released).
  `OrtGpuIoBinding` refuses to run if ORT did not register the buffers that way,
  so an ORT upgrade that changes this fails at session creation, where the
  CPU-IO fallback still applies.
- **Graph capture needs the object-form provider.** ORT-web forwards
  `enableGraphCapture` to the native EP only when `executionProviders` holds an
  object entry (`[{ name: 'webgpu' }]`). With the string form `['webgpu']` the
  JS side switches to capture rules (IO bound once) but the EP never captures:
  still 356 bind groups per invoke. `ortGraphCaptureSessionOptions()` returns
  the object form.
- **f16 input bits.** In browsers that have `Float16Array`,
  `ort.Tensor('float16', Uint16Array)` re-wraps the data as a `Float16Array`.
  The GPU-IO path writes the f16 bits straight to the input buffer and never
  builds an `ort.Tensor` for the input.
- ORT's log level defaults to fatal. The lab pages take `?ortLogSeverity=N` to
  see EP warnings.

## 5. Product change (branch only)

- `src/nn/ortGpuIo.ts`: `OrtGpuIoBinding` (persistent input buffer, 16-byte
  aligned output buffers, one `MAP_READ` staging buffer, one copy encoder and
  one `mapAsync` per invoke, runs serialized), `pinOrtGpuBufferImport`,
  `ortGraphCaptureSessionOptions`.
- `src/lc0/onnxEvaluator.ts`: `ioMode: 'cpu' | 'gpu' | 'gpu-capture'`. A GPU mode
  needs a WebGPU placement and a model with fixed shapes. The session is created
  WebGPU-only with GPU outputs and a zero-input warm-up run. Any failure there
  logs a warning and falls back to the ordinary CPU-IO session (which keeps its
  WASM fallback).
- `src/lc0/searchWorkerCore.ts`: an init that names no `ortIoMode` gets
  `defaultLc0OrtIoMode(deviceModelSelection)`: `'gpu-capture'` for the
  device-selected Lc0 Small model (Analysis, Play, Arena and Critical moments
  without `?model=`), `'cpu'` for everything else (explicit model URLs, big
  nets, lab tools). The ready message reports the mode in use.
- Main-thread Lc0 Small loaders are unchanged (CPU IO).
- Checked on the built M4 preview (`ort-search.html?modes=cpu,default,gpu-capture&reinit=1`):
  an init with no `ortIoMode` and device selection reports `gpu-capture` and
  serves split-K. A re-init that builds a second captured session while the
  first is still alive works. All searches matched the CPU-IO worker (4
  positions, a functional check, not a timing run).
- Tests: `tests/ort_gpu_io.test.mjs` (fake ORT register/unregister: layout,
  per-invoke call order, pin keeps the handle alive across runs, capture needs
  the pin, refusal when ORT does not register, input validation and recovery,
  model eligibility, the object-form provider, the worker default).

Not verified: graph capture on D3D12 (Windows), Android, Intel/AMD GPUs and
Safari; and
device loss after creation (no fallback mid-search, same as the CPU-IO path).

## 6. What did not work

- **GPU outputs alone** (`gpuout`): the same three map round trips, a 2-12%
  gain at best, and noisy on split-K (p90 1.18).
- **IO binding without capture**: one map instead of three, but encode is
  unchanged. Worth 7% on the M4 and 21% on the 3090; capture beats it everywhere.
- **The string-form capture flag**: no capture at all (§4).
- **Capture without pinning**: aborts on the first replay.
- **GPU busy time from the pass-mode profiler** varies between unpaired runs
  on the loaded M4 (8.6-15 ms for the same model), so it is used here only for
  the per-invoke busy-vs-wall picture, not to compare levers.

## 7. Reproduce

```bash
BUILD_SCOPE=product npm run build:client && node scripts/kernel_hillclimb/build_lab.mjs
node scripts/serve_isolated_static.mjs dist-client 5264 &
notes/kernel-hillclimb/benchlock.sh ortov -- node scripts/kernel_hillclimb/bench.mjs \
  --page ort-overhead --query 'variants=cpu,gpuout,iobind,capture&rounds=40&block=5&warmup=20&traceInvokes=30' \
  --base-url http://127.0.0.1:5264 --browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features
# split-K: add &model=/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.splitk.onnx
# search: --page ort-search --query 'worker=/_app/immutable/workers/searchWorker-<hash>.js&modes=cpu,gpu,gpu-capture&rounds=4'
#   (add &deviceModelSelection=0 for plain qdq8 on Apple)
```

Yukon: the same pages through `scripts/kernel_hillclimb/remote/run.mjs --host
yukon --dist dist-client --with-models` with an `--include` list of `lab/**`,
`assets/**`, `_app/**`, `fixtures/lc0/**`, `lc0/**`, the
`ort/ort-wasm-simd-threaded*` files and the model. Result JSON:
`artifacts/kernel-hillclimb/results/` (`ort-overhead-*-full3.json`,
`ort-search-*-full4.json`, `ortov-yukon-*.json`; local, not committed).
