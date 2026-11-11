# Kernel hill-climb harness

Measures where the shipped GPU paths stand against the device peak, and judges
later kernel candidates (hand-written WGSL, runtime configs, fusions) on one
fixed scale. Baseline numbers and the ranked candidate list are in
`docs/kernel_hillclimb_roofline_2026-11.md`.

## Pieces

| file | role |
|---|---|
| `lab/kernel-hillclimb/peak.html` | device peaks: f32/f16 FMA, read/copy bandwidth, dispatch/submit/readback overhead, tiled GEMM at real model shapes, `chromium-experimental-subgroup-matrix` GEMM when exposed |
| `lab/kernel-hillclimb/ort-profile.html` | ORT WebGPU (shipped Lc0 / Maia3 path) timing + per-dispatch GPU profile + ORT WASM parity |
| `lab/kernel-hillclimb/tvmjs-profile.html` | TVMJS WebGPU (shipped Centipawn path, Lc0 research lanes) timing + profile + ORT parity |
| `lab/kernel-hillclimb/gpu-profiler.mjs` | runtime-agnostic WebGPU profiler (see below) |
| `scripts/kernel_hillclimb/build_lab.mjs` | production-compiles the pages and overlays them onto `dist-client` |
| `scripts/kernel_hillclimb/candidates.mjs` | named candidates: page + query + correctness gate |
| `scripts/kernel_hillclimb/bench.mjs` | local runner (agent-browser): opens a page, waits for the result, writes one JSON |
| `scripts/kernel_hillclimb/onnx_cost.py` | per-node FLOP / byte cost of an ONNX model at a fixed batch |
| `scripts/kernel_hillclimb/roofline.mjs` | joins profiles + costs + peaks into the roofline table |

## Build and serve

```sh
BUILD_SCOPE=product npm run build:client          # once; needs models/runtimes symlinked as for any build
node scripts/kernel_hillclimb/build_lab.mjs       # overlays lab pages, fixtures, lc0 TVMJS runtimes symlink
node scripts/serve_isolated_static.mjs dist-client 5241   # COOP/COEP static server
```

`build_lab.mjs` symlinks `dist-client/runtimes/lc0-tvmjs-webgpu` to the main
checkout's local (gitignored) copy when missing; without it only the Lc0
TVMJS targets 404. Pages need cross-origin isolation (ORT threads), so serve
with COOP/COEP; never `vite dev` (it masks built-worker ORT failures). The
static server caches for 1h: force-reload after a rebuild.

## Result contract (remote devices)

Every bench page is self-contained: static dist + COOP/COEP + URL query params.
A driver (agent-browser, Playwright, a phone farm, a human with devtools) only:

1. opens `http(s)://HOST/lab/kernel-hillclimb/<page>.html?<query>`;
2. polls `window.__kernelBenchResult` (object) and `window.__kernelBenchError` (string);
3. stores the result object verbatim.

Exactly one of the two gets set. `__kernelBenchError` carries the message and
stack; a page never leaves both unset unless it is still running (progress is
in `#log`). Result object (`schema: "kernel-hillclimb.bench-result.v1"`):

| field | meaning |
|---|---|
| `schema`, `generatedAt`, `bench` | `peak` / `ort-profile` / `tvmjs-profile` |
| `ok` | correctness gate passed (and no profiler/runtime errors). Timing from a result with `ok: false` must not be used |
| `device` | adapter vendor/architecture/description, subgroup min/max, subgroup-matrix configs, adapter + device features, limits, WGSL language features, UA + UA-CH high-entropy values, platform, cores, `crossOriginIsolated`, page URL + query |
| `candidate` | runtime, preset/target, model or manifest, physical batch, dequant-fold summary, effective backend |
| `params` | warmup / iters / profileInvokes / fixture FENs |
| `timing.runMs` | noise-aware stats over unprofiled invokes: n, median, MAD, MAD/median, mean, min, p10, p90, max, `noisy` (MAD/median > 10%) |
| `profile` | per-dispatch GPU profile (null if `timestamp-query` missing): `kernelMsPerInvoke`, `perInvoke[]` (busy / envelope / gap ms, dispatches), `kernels[]` (by name), `sequence` (invoke 1: `[name, µs, grid, boundBytes]`), `submitsPerInvoke`, `timestampsQuantized` |
| `parity` | gate details (per-output max-abs, best-move matches, tie tolerances) |
| `peaks` (peak page) | `fmaF32Tflops`, `fmaF16Tflops`, `readGBps`, `copyGBps`, best GEMM per shape + kernel |

Query params per page:

- `peak`: `tests=fma,bandwidth,overhead,gemm,sgmatrix` `trials=7` `gemmShapes=MxNxK,...`
- `ort-profile`: `preset=lc0-small|lc0-small-b8|t3|bt4|lqo|maia3|maia3-grid` (or `model=URL&batch=N`), `warmup=5` `iters=30` `profileInvokes=5` `parity=wasm` `ortDequantFold=0`
- `tvmjs-profile`: `target=centipawn|lc0-t1|lc0-t3|lc0-bt4` `manifest=URL` `batch=N` `warmup` `iters` `profileInvokes` `parity=ort` `tieEpsilon=0.01`

Timestamps: Chrome quantizes `timestamp-query` to 65.536 µs unless launched
with `--enable-webgpu-developer-features`; `profile.timestampsQuantized` says
which you got. Per-dispatch rows are only meaningful unquantized; `timing.runMs`
(wall) is valid either way. `chromium-experimental-subgroup-matrix` needs
`--enable-unsafe-webgpu` (Chrome 157 on M4); it is not available to users.

## Local runner

```sh
export AGENT_BROWSER_SESSION=kernel-hillclimb
LOCK=/Users/macthedan/projects/lc0_browser/notes/kernel-hillclimb/benchlock.sh   # shared Mac: always wrap timing
$LOCK kernel-ort-small -- node scripts/kernel_hillclimb/bench.mjs --candidate ort-lc0-small --gate --repeat 3 \
  --browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features
```

`--gate` appends the candidate's in-page gate query. `--command-gate` also runs
the candidate's repo smoke (starts its own vite dev server; local checkout only).
`--browser-args` must use the `=` form. Output:
`artifacts/kernel-hillclimb/results/<name>-<stamp>.json` (`schema:
kernel-hillclimb.bench-run.v1`: url, browser args, `host.uptime`, load before /
after each repeat, gate summary, `runMedianMsAcrossRepeats`, all page results).
Record `uptime` with every number and repeat anything promotion-relevant;
compare candidate vs baseline with `scripts/lib/perf_canary_stats.mjs`
`compareMetric` (threshold max(10%, 3×MAD/median)).

## Correctness gates per path

| path | gate | where |
|---|---|---|
| ORT WebGPU (Lc0 small / t3 / BT4 / LQO, Maia3) | `parity=wasm`: per-output max-abs vs ORT WASM on identical encoded fixture inputs, within the preset's threshold. `npm test` separately pins ORT WASM to native lc0 fixtures, so this chains the shipped GPU path to native | in-page |
| TVMJS Centipawn | `parity=ort`: legal-policy and WDL max-abs ≤ 0.001 vs the ORT Centipawn evaluator (16 positions). Promotion also needs the search-parity runs in `docs/centipawn_tvmjs_subgroups_2026-10.md` | in-page |
| TVMJS Lc0 (t1 / BT4) | `parity=ort&tieEpsilon=0.01`: best legal move per position vs ORT f16 b1, f16 near-tie flips within 0.01 prior tolerated (the BT4 onboarding gate), plus max prior / WDL diff | in-page; `--command-gate` adds `npm run lc0:tvmjs-webgpu-smoke` search parity |
| hybrid custom-WGSL heads (not in v0) | `npm run lc0:browser-wgsl-smokes` | command (`WGSL_HEADS_GATE`) |
| a new hand-written kernel | its own unit gate vs a CPU/ORT reference at the real shape, then the path gate above with the kernel swapped in | both |

## Profiler

`installGpuProfiler()` patches `GPUAdapter.prototype.requestDevice` before the
runtime boots (adds `timestamp-query`), wraps pipeline / bind-group creation to
learn kernel names (WGSL entry point, else label, else shader hash) and bound
bytes, and while armed splits every compute pass into one timestamped pass per
dispatch (pipeline + bind groups replayed). It works for ORT's native WebGPU EP
(emdawnwebgpu) and TVMJS alike. ORT names pipelines by op type (`Gemm`,
`MatMul`, `Softmax`...), so `roofline.mjs` maps matmul dispatches to ONNX nodes
by bound bytes; TVMJS kernels map to probe-JSON MACs by function name.
Splitting passes costs ~1 µs per dispatch and serializes the queue, so profiled
busy time is an upper bound; the unprofiled `timing.runMs` is the number that
counts.

## Roofline

```sh
PY=/Users/macthedan/projects/lc0_browser/leelaweb/.venv-onnx/bin/python
$PY -I scripts/kernel_hillclimb/onnx_cost.py public/models/lc0/<model>.onnx --batch 8 --out artifacts/kernel-hillclimb/cost/<cost>.json
node scripts/kernel_hillclimb/roofline.mjs --md artifacts/kernel-hillclimb/roofline.md
```

Per kernel group (name, grid, bound bytes): FLOPs, bytes, median µs, TFLOP/s,
GB/s, % of FMA peak and of read bandwidth, arithmetic intensity, and a bound
label: `launch` (median dispatch < 8 µs, ~10× the empty-dispatch cost),
`compute` / `bandwidth` (≥ 50% of the attainable roof, by AI vs the ridge), or
`underused` (neither roof reached: occupancy or kernel quality, the WGSL
target). Per invoke: wall, GPU busy, non-kernel wall (= wall − busy: submit,
readback, JS, idle), dispatches, matmul TFLOP/s.
