# ORT 1.27 native WebGPU: honest kernel-profiler availability

Branch: `fix/ort-profiler-native-ep`. Follow-up to [the rejected ORT 1.30 A/B](ort_1_30_ab_2026-10.md), not an ORT upgrade or performance optimization.

## Installed API/source evidence

Both installed packages, `onnxruntime-web` and `onnxruntime-common`, report **1.27.0**. These are package-relative citations to the inspected installed files (not assumptions from newer upstream documentation):

- `onnxruntime-common/lib/env.ts:143–180`: defines timestamp-event metadata and `env.webgpu.profiling.{mode,ondata}`. The 1.27 type comment does **not** explicitly say JSEP-only; implementation establishes the limitation.
- `onnxruntime-web/lib/wasm/wasm-core-impl.ts:150–165`: JSEP calls `initJsep`; native WebGPU instead calls `webgpuInit` with a callback that sets `env.webgpu.device`, with no profiling callback wiring.
- `onnxruntime-web/lib/wasm/jsep/backend-webgpu.ts:390–435`: maps timestamp results and invokes `env.webgpu.profiling.ondata`, converting the nanosecond timestamps to JavaScript numbers. This is the only `ondata` invocation found in the installed WebGPU implementation.
- `onnxruntime-common/lib/inference-session.ts:565–572`: `startProfiling()` and `endProfiling()` return `void`, not a kernel profile.
- `onnxruntime-web/lib/wasm/session-handler-inference.ts:147–153`: `startProfiling()` is a TODO; `endProfiling()` delegates to the WASM core.
- `onnxruntime-web/lib/wasm/wasm-core-impl.ts:1106–1119`: `_OrtEndProfiling` obtains a profile filename and immediately frees it, explicitly noting that the filename is not used yet. It does not expose a browser-readable kernel-timing result.

No exposed native kernel-timing replacement was found in these 1.27 source/types. No replacement, runtime switching, timestamp-query monkeypatch, or new profiling API is introduced.

## Consumer audit and semantics

Audit commands:

```sh
git grep -n -i -E 'ondata|profiling|kernel.?profile' -- src/ lab/ scripts/
git grep -n -E 'ortKernel|kernelGpuMsTotal|topPrograms|webgpuDiagnostics|webgpuDelta|OrtWebGpuProfiling' -- src/ lab/ scripts/
```

Actual ORT consumers:

- `src/nn/ortRuntime.ts`: callback collector, cumulative runtime diagnostics, and per-evaluation snapshot subtraction. Summaries now have `status` (`disabled`, `unavailable`, `available`) and an explanatory `reason` when no events exist. Zero-event totals and program lists are **null**, not `0`/`[]`. A requested zero-event interval says **"unavailable on native WebGPU EP"**, even if the cumulative snapshot includes older JSEP events. `eventCount=0` is explicitly the number of callbacks observed, not the number of GPU kernels executed.
- `src/lc0/onnxEvaluator.ts`: evaluation timing carries `ortKernelProfileStatus`, `ortKernelProfileReason`, and nullable `ortKernelGpuMs` alongside the unchanged callback/API counters.
- `src/lc0/policyOnlyBrowser.ts`: benchmark JSON previously stripped all nonnumeric timing values. `src/lc0/timingReport.ts` preserves the kernel status/reason/null duration in `lastBackendTiming`, while leaving finite-number-only timing sampling unchanged. No unavailable duration contributes a zero sample to `phaseTimingStats`.
- `scripts/lc0_browser_ort_readback_profile.mjs`: passes through that benchmark JSON; help now explains that kernel timestamps are JSEP-only/unavailable on native WebGPU.
- `scripts/lc0_browser_readback_strategy_matrix.mjs`: retains status/reason/null duration in each ORT cell and carries distinct statuses/reasons into the printed aggregate summary.

The `lab/lc0-tvmjs-webgpu-smoke.html` profiler and TVM sweep scripts use their own WebGPU timestamp-query collector, not ORT's callback. They are unchanged. Other search matches are unrelated GPU probes, profiling harnesses, server `onData`, and TVM `ToNonDataflow`.

Real events remain authoritative: timestamp conversion, total rounding, program sorting/truncation, callback chaining, diagnostic waits, and API instrumentation are unchanged. A measured zero-duration event remains `available` with numeric `0`; an empty interval is not a measurement. The existing callback stays installed after disabling; historical/events-still-arriving data remain reportable, with `enabled` separately recording whether profiling was requested. Neither absence of events nor this status indicates WASM fallback.

## Production-browser diagnostic proof

Built with `BUILD_SCOPE=product npm run build:client`; served by `node scripts/serve_isolated_static.mjs dist-client 5196`. An existing, unowned IPv4 listener occupied `127.0.0.1:5196`; it was not killed or modified. The owned server was checked at **`http://[::1]:5196`**, on the assigned port only.

Native `agent_browser`, unique session `ort-profiler-native-5196`, Google Chrome executable `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`, `--enable-unsafe-webgpu`. Browser UA: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36`. Ten logical CPUs, cross-origin isolated.

All browser inference/profiling was inside the machine-wide lock holder:

```sh
/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/701135f9-2bed-49a5-8cf9-e17058c5574a/scratchpad/benchlock node .local-dev-artifacts/ort-profiler/hold-lock.mjs
```

The holder signalled acquisition before opening the inference page and was released after closing the browser. The owned server was then stopped by its recorded PID. Initial IPv4 HTTP-404 navigation attempts executed no inference.

A/B here means **requested kernel profiling / disabled kernel profiling**, both on the fixed branch, not before/after code or ORT versions. Order **A1, B1, B2, A2, A3, B3**; each full navigation/reload creates a new production `searchWorker`/ORT session. Strict `ep=webgpu`, one thread, native Asyncify pair, physical **batch 1**, startpos, existing DQL fold enabled. Both arms retain WebGPU API instrumentation and `preferredOutputLocation=gpu-buffer`; one warmup and three timed evaluations per session. Model:

```text
t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx
SHA-256 9396c11226de7f29dbad8c60147b1082b8093adc7a0b77819cd0fb33103a095f
```

| Diagnostic arm | Profile status (3/3 sessions) | Kernel ms | Last-evaluation GPU submits/maps | Median session median wall ms [min, max] |
|---|---|---|---|---|
| A: requested | unavailable, explicit native-EP reason | null | 27 / 3 | 11.105 [9.785, 11.280] |
| B: disabled | disabled, explicit reason | null | 27 / 3 | 10.035 [8.940, 11.670] |

All six reports have `ortKernelCount=0`, no `phaseTimingStats.ortKernelGpuMs`, and `backend=webgpu->webgpu`. The positive API counters establish real GPU work independently of the absent kernel callback. Instrumented short-run wall times are reported only for protocol completeness: they are perturbed diagnostics, **not a speedup/regression or GPU-duration claim**.

Reported output parity across all six sessions: `bestMove=d2d4`, `q=0.04321289`, `mlh=184.375`. This is single-position reported-output parity, not a full GPU tensor/search-corpus parity assertion. Unit tests additionally compare complete synthetic evaluator outputs with and without emitted JSEP events, excluding timing metadata.

Artifact identities:

| Artifact | SHA-256 |
|---|---|
| `ort-wasm-simd-threaded.asyncify.mjs` | `7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655` |
| `ort-wasm-simd-threaded.asyncify.wasm` | `7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a` |
| `searchWorker-DZ4wh5UX.js` | `0af4ed3c8919200fecdf4e459496b7c11ec3f81031f8102e75d7b374866334da` |

Selected browser result captures and summary remain local/ignored at `.local-dev-artifacts/ort-profiler/` (not a portable committed harness):

| Capture | SHA-256 |
|---|---|
| `A1.json` | `ded8e020843300ed219d7259f347590dab2d504142724f5d4f1b0d13aa871e05` |
| `B1.json` | `8e96e862d07f91521a5f66fee0a80420e5f17441fc5cfc78c5441f9c75c5130b` |
| `B2.json` | `75862d0a6528bc9690029ad7a451606ab06a3db3ccaf2e68f4e5b1c4c1a889c4` |
| `A2.json` | `92dbcf10ce9d9c03050a294597b2792e41d0f5b6cac6515af68824393b4b8943` |
| `A3.json` | `9eb75e9c6b34294886544d93bd18082f0afafdd6d720876fd5bb1ae7868f6bc3` |
| `B3.json` | `c39d3e0c6ec1aad24cd7ac028328ada69c4f9ea0c7f1696d6799d6231c85c8aa` |

## Gates and limits

- `npm run typecheck`: zero errors/warnings.
- Focused ORT profiler/report/matrix/fallback/build-config tests: **36/36 passed**.
- Full `npm test`: **812 passed, zero failures, three optional skips** (815 total).
- Product build and its SvelteKit runtime-id check passed; built WASM-dedup check passed. The additional runtime-asset allowlist check on the raw, unpruned Vite output rejects the copied full ORT dist directory, as expected before release asset staging; this branch does not change that staging or claim that the raw build is a staged deployment.
- `git diff --check`: passed. Structured branch review is recorded at closeout.

No models, package versions, execution-provider selection, graph rewrites, or deployment configuration changed. No push/merge/deploy. The open limitation remains upstream: ORT 1.27 exposes no inspected native callback replacement. JSEP event-path compatibility is tested by injecting its documented event shape through the installed callback; an actual JSEP runtime/GPU run is not claimed.
