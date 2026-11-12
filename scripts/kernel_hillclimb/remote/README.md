# Remote kernel-bench runner

Runs a static bench page in headless Chrome on a remote benchmark host (or locally) and
returns one JSON object. Zero npm dependencies: the agent uses node's built-in `WebSocket`
to drive Chrome over the DevTools protocol.

| host       | machine                                   | Chrome                                   | scheduling            |
|------------|-------------------------------------------|------------------------------------------|-----------------------|
| `yukon`    | Ryzen 9 3900XT, RTX 3090 (Vulkan), Linux  | Chrome for Testing 151 (puppeteer cache) | shared `gpu-queue`    |
| `mac-mini` | Apple M4 10-core GPU, 16 GB, macOS 27     | Google Chrome.app 157                    | lock + wait-for-idle  |
| `local`    | Apple M4 (this Mac)                       | Google Chrome.app                        | `benchlock.sh`        |

## Page contract (shared with the bench harness)

- The page is served from a **static directory** (`--dist`) at `http://127.0.0.1:<free port>/`,
  bound to loopback only, with these headers on every response:
  `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`,
  `Cross-Origin-Resource-Policy: same-origin`, `Cache-Control: no-store`.
  MIME types: `.wasm` → `application/wasm`; `.js`/`.mjs` → `text/javascript`;
  `.json`, `.html`, `.css`, `.wgsl` (text/plain). `/foo` falls back to `/foo.html`, and
  directories fall back to `index.html`. So `crossOriginIsolated === true`, and
  SharedArrayBuffer / wasm threads work. Only regular files whose resolved path (symlinks
  followed) stays inside the dist root are served. Anything else gets a 403 or 404.
- `--page` is a path relative to the dist root, with an optional query string
  (`kernels/bench.html?kernel=conv3x3&batch=8`). The page reads its own parameters
  from `location.search`.
- On success the page sets **`window.__kernelBenchResult`**, a JSON-serialisable object.
  On failure it sets **`window.__kernelBenchError`**, a string. The agent polls every 250 ms,
  and whichever appears first ends the run. Neither appearing within `--timeout` seconds
  (default 120) is a timeout.
- **A failed or partial measurement is an error, not a result.** If any benchmark step throws,
  an error scope or `uncapturederror` reports a WebGPU error, or the device is lost, the page sets
  `__kernelBenchError` and must not set `__kernelBenchResult`.
  `lab/kernel-hillclimb/probe-contract.mjs` (`settleProbe`, `publish`) implements this, and
  bench pages should reuse it.
- Console output and uncaught exceptions are captured (last 60 lines go into `meta.console`).

## Usage

```bash
# from the repo root (worktree)
node scripts/kernel_hillclimb/remote/run.mjs --host yukon    --dist <dir> --page <path?query> [--timeout 120]
node scripts/kernel_hillclimb/remote/run.mjs --host mac-mini --dist <dir> --page <path?query>
node scripts/kernel_hillclimb/remote/run.mjs --host local    --dist <dir> --page <path?query>

# proof page (self-contained adapter/features/limits + ~2 s FMA and bandwidth microbench)
node scripts/kernel_hillclimb/remote/run.mjs --host yukon --dist lab/kernel-hillclimb --page device-probe.html --gpu-info
#   device-probe.html?bench=0   adapter/feature probe only
#   device-probe.html?ms=2000&mib=64&power=high-performance
```

Options (see the header of `run.mjs` for all of them):

- `--include a,b/**` syncs only these paths or globs from `--dist`. Use it to keep payloads small
  (for example `--dist build --include kernel-bench,_app/immutable/workers`).
- Model blobs are excluded by default: `*.onnx`, `*.pt`, `*.safetensors`, `*.nnue`, `*.lc0web`,
  `models/`, `tensor-cache/`, `params_shard_*.bin` and `*.map`. Pass `--with-models` when the
  page needs them. The default excludes take precedence over `--include`, so an `--include` that
  names an excluded directory (such as `models/`) or only excluded files (such as `net.onnx` or
  `nets/*.onnx`) without `--with-models` is refused before any ssh, sync or queue step,
  instead of silently syncing nothing. `--host local` serves `--dist` in place and is not checked. Each run hard-links unchanged files against the previous run's dist on the
  host (`rsync --link-dest`), so a repeat upload costs only the changed files.
- `--gpu-info` adds CDP `SystemInfo.getInfo` (the `chrome://gpu` data: GPU devices, driver,
  feature status, ANGLE/Vulkan aux attributes) to `meta.gpuInfo`.
- `--estimate-min N` (yukon) sets the honest gpu-queue estimate. The default is 5.
- `--no-wait` (mac-mini) skips the idle gate. Use it only for adapter and feature probes. The
  result gets `runner.timingValid: false`.
- `--max-wait S` sets how long the runner waits locally (default 6 h). Remote jobs are detached,
  so if the runner gives up or is killed, the job keeps going. Collect the result later with
  `run.mjs --host <h> --fetch <run-id>`. Every ssh call is bounded: ServerAlive catches a dead
  link, and a command timeout (at most 120 s, never past the remaining budget) catches a hung
  remote command. If a submit or start call times out or hits an ssh transport failure (exit
  255), the runner can't tell whether it took effect. It then exits 76 (pending) with the run id
  instead of failing or resubmitting.
- `--chrome`, `--flags-json '[...]'` and `--add-flags=--a,--b` override the host's browser or flags.
- `--out FILE` also writes the JSON to `FILE`. The JSON always goes to stdout, and progress goes
  to stderr.

Exit codes:
- `0` result.
- `2` page error or timeout.
- `3` runner or agent failure.
- `4` refused: a 0x88 job is already waiting in gpu-queue.
- `75` **confirmed** never started: the host or GPU stayed busy, or the host lock was not
  acquired, within max-wait.
- `76` still pending remotely when the runner's own `--max-wait` ran out. The job may still run,
  so collect it with `--fetch <run-id>`.

Run ids are `<UTC second>-<page slug>-<12 hex random>`. The run directory is created with a
plain `mkdir`, so a collision fails instead of sharing a directory.

## Output

```jsonc
{
  "ok": true,
  "result": { /* window.__kernelBenchResult */ },
  "error": null,                       // __kernelBenchError, "timeout after Ns", or "busy: …"
  "meta": {
    "host", "platform", "cpus", "cpuModel",
    "browser": { "binary", "product", "userAgent" }, "flags", "url", "wallMs",
    "loadStart", "loadEnd",            // loadavg, 500 ms CPU busy %, GPU (nvidia-smi or ioreg)
    "duringRun": { "samples", "cpuFailures", "gpuFailures", "foreignCpuCoresMax",
                   "foreignCpuCoresMean", "load1Max", "foreignGpuApps",
                   "foreignGpuPct", "foreignGpuPctMax", "foreignGpuPeakAt", "foreignGpuTop" },
    "idleWait",                        // mac-mini: seconds waited, thresholds
    "contended": false, "contendedReasons": [],   // someone else measurably used the machine
    "verified": true, "unverifiedReasons": [],    // false: a reading failed, so we can't say
    "console": [...], "gpuInfo": {...}
  },
  "runner": { "host", "runId", "remoteDir", "scheduler", "queueJobId", "timingValid", "run": {...} }
}
```

`loadEnd` is sampled about 3 s after Chrome exits, so its GPU utilisation is other people's
load, not ours. `duringRun.foreignCpuCores` sums `ps %cpu` over every process outside the
agent and its Chrome process group. During the run, foreign GPU use is attributed per process:
- Linux: `nvidia-smi --query-compute-apps` pids outside our process group.
- macOS: `ioreg -r -c AGXDeviceUserClient -l` per-client `accumulatedGPUTime` (ns, no root
  needed) for clients outside our process group. `foreignGpuPctMax`, the busiest ~1 s sampling
  interval, is what gets compared with `--gpu-max`, so a short burst can't hide in the
  whole-run mean (`foreignGpuPct`, which is kept as metadata).

Samples are taken right after navigation, then every second, then once at the end. Each records
whether its CPU and GPU queries succeeded. Any failed sample, fewer than 2 samples, or an unknown
start or end GPU reading sets `verified: false`. **`runner.timingValid` is true only when
`verified === true` and `contended === false`.** On mac-mini, the agent must also report that
the idle gate ran: `meta.noWait === false` plus an `idleWait` record. The controller's `run.json`
can only invalidate a result, because fetching it can fail. Results from agents older than these
fields are never timing-valid.

## Scheduling: the hosts are shared

### Yukon: `gpu-queue` (`~/projects/GPU-POLICY.md`)

Every job that touches the 3090 goes through the shared one-slot FIFO, including adapter probes.
The runner:

1. Refuses (exit 4) if a `[0x88:…]` job is already **queued**. Policy allows at most one 0x88 job
   waiting. The check and the `gpu-queue submit` run as one critical section on Yukon, under
   `flock ~/kernel-hillclimb/.submit.flock`, so two runners cannot both pass the check. Other
   tools that submit 0x88 jobs without that flock are not covered.
2. Writes `job.sh` into `~/kernel-hillclimb/<run-id>/`. This one script holds the whole
   lifecycle: static server, headless Chrome, page, teardown. It runs `nvidia-smi` before and
   after, and lists any leftover `khc-chrome-*` processes, which should be none.
3. Submits it as `gpu-queue submit -p 0x88 -e <est> -n <name> -o <run>/job.log -- <run>/job.sh`.
   `<est>` must be a positive number and is shell-quoted. The runner
   and records the job id in `<run>/queue-job-id` and `runner.queueJobId`.
4. Polls `gpu-queue list` and `<run>/result.json` every 20 s. While it waits, it prints which jobs are ahead.
   - If the job finishes without a result and exit `75`, it never started because the GPU was busy for 30 min.
     **Investigate. Do not resubmit in a loop.**
   - The runner never kills, reorders or removes queue entries.

The queue starts a job only when no process holds more than 1 GiB of GPU memory, so results from
inside the queue count as **clean**. `meta.contended` is still set if any compute app shows up
in `nvidia-smi` at start or end, or if a compute app outside our Chrome process group shows up
during the run (sampled every second).
Light CPU only: rsync, ssh and one Chrome. Nothing here needs `cpu-run`.

### Mac mini: queue behind (lock + wait-for-idle)

Mac mini has no shared queue, and other projects run CPU arenas and Metal jobs there. The job:

1. Runs the agent under `lockf -k -t <max-wait> ~/kernel-hillclimb/.host.lockf` (BSD lockf(1)),
   so our own jobs serialize. The lock is a kernel advisory lock that the kernel releases when its
   holder dies. There is no stale-lock reclamation step to race. Timing out on the lock gives
   exit 75 and a `busy` result.json. The old mkdir lock (`~/kernel-hillclimb/.lock`) is retired;
   only jobs started before 2026-11-11 23:30Z still use it.
2. Waits until the 1-min load average is `< --load-max` (default 3) **and** `ioreg -c IOAccelerator`
   `"Device Utilization %"` is `< --gpu-max` (default 10). Both must hold for `--hold`
   seconds (default 60). A missing or unparseable GPU reading counts as **not idle** and resets
   the hold. It polls every `--poll` seconds (default 30) and logs each reason for waiting to
   `<run>/agent.err`, which the runner relays.
3. Gives up when `--max-wait` runs out with exit 75 and a `busy: …` error. That budget is
   measured from job start and covers both the lock wait and the idle wait.
4. Flags `meta.contended: true` (with reasons) in these cases:
   - the thresholds were crossed at start or end;
   - foreign CPU went over `--load-max` cores during the run;
   - foreign GPU (`foreignGpuPctMax`, attributed as above) went over `--gpu-max` in any sampling
     interval during the run.

   Unknown or failed readings set `verified: false` instead (see the result shape).
   Device-wide load1 and ioreg utilisation taken mid-run include our own Chrome, so they are not
   used during the run. `runner.timingValid` is false when the run is contended, unverified, or
   was started with `--no-wait`.

   The deadline is checked before a completed idle hold is accepted, and every poll sleep is
   capped at the remaining budget. Chrome never launches after max-wait has passed.

The job is started with `nohup` and runs detached, so a dropped ssh session does not kill it.
It never stops or renices the other jobs.

### Local: `benchlock.sh`

`--host local` runs the agent under
`<workspace>/notes/kernel-hillclimb/benchlock.sh` (`<workspace>` is the parent of the checkout; set
`KERNEL_BENCHLOCK` if it lives elsewhere), so local timing
runs serialize with the other agents on this Mac.

## Chrome flags (what works)

Common to all hosts: `--headless=new --enable-unsafe-webgpu --enable-webgpu-developer-features`.

- `--enable-unsafe-webgpu` is required for `chromium-experimental-subgroup-matrix`, and with it
  `adapter.info.subgroupMatrixConfigs`. Without it the feature is absent even on M4 and Chrome
  153/157. Real users of 0x88.app will not have it unless Chrome ships it.
- `--enable-webgpu-developer-features` adds `adapter.info.backend`, `type`, `driver`,
  `memoryHeaps` and `powerPreference`, and leaves timestamp-query unquantized.
- Yukon (Linux, no DISPLAY) also needs the flags in the next section.

### Yukon bring-up (gpu-queue job 585, 2026-11-11)

Matrix: 3 builds x 6 flag sets (`bringup_matrix.mjs`, `device-probe.html?bench=0`).
The builds were puppeteer Chrome for Testing 151.0.7922.71, playwright chromium-1228
(CfT 149.0.7827.55) and chromium_headless_shell-1228.

- **`--no-sandbox` is mandatory.** Without it, Chrome never writes `DevToolsActivePort` on any of
  the three builds, because the setuid/userns sandbox is unusable there and we have no sudo.
  Every config without it failed.
- With `--no-sandbox`, all three builds expose the hardware adapter:
  `nvidia / ampere / NVIDIA GeForce RTX 3090`, `backend: vulkan`, `type: discrete GPU`,
  driver `NVIDIA: 580.178.04`. ANGLE also runs on Vulkan (`chrome://gpu`: vulkan enabled_on,
  webgpu enabled). It is not SwiftShader or lavapipe.
- **`shader-f16` is not exposed** on NVIDIA Vulkan by default, so subgroup-matrix offers only
  `u8/i8 -> u32/i32` configs (16x16x32 and 16x8x32). Dawn has the toggle
  `vulkan_enable_f16_on_nvidia` ("Enables F16 on Nvidia GPUs with Vulkan"), which the runner
  passes through `--enable-dawn-features=`. That is confirmed in gpu-queue job 588: with it,
  `shader-f16` is true and subgroup-matrix adds `f16 -> f16|f32` configs at 16x16x16, 16x8x16
  and 16x8x8. Note that real users on NVIDIA/Linux Chrome do not get f16 this way.
- The default runner flags for yukon:
  `--headless=new --enable-unsafe-webgpu --enable-webgpu-developer-features --no-sandbox
  --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --use-angle=vulkan
  --ignore-gpu-blocklist --disable-vulkan-surface --enable-gpu
  --enable-dawn-features=vulkan_enable_f16_on_nvidia`

The agent always adds `--user-data-dir=<fresh tmp profile> --remote-debugging-port=0
--no-first-run --no-default-browser-check --disable-extensions` and the three
`--disable-*-throttling/backgrounding` flags. It deletes the profile afterwards.

## Process hygiene

- Each remote run lives in `~/kernel-hillclimb/<run-id>/`: `dist/`, `agent.mjs`, `job.sh`,
  `run.json`, `job.log`, `agent.out`, `agent.err`, `result.json`, and `queue-job-id` or `job.pid`.
- Chrome is spawned as its own process group. Teardown runs CDP `Browser.close`, then SIGTERM
  and SIGKILL to **that process group only**, then closes the server. Nothing is killed by
  name or pattern.
- The agent has a hard deadline of `--timeout + 90` s once it starts. With `--watch-stdin`, it
  also cleans up when its controller's ssh goes away. The queued and detached modes do not use
  `--watch-stdin`.
- Old run directories are kept for inspection, and they are small because dist is hard-linked.
  Delete `~/kernel-hillclimb/<run-id>` by hand when done.

## Files

- `run.mjs` is the local controller: rsync, schedule, poll, fetch, augment.
- `agent.mjs` runs on the host: static COOP/COEP server, Chrome launch, CDP poll, load sampling
  and idle gate. Exactly one terminal outcome: cleanup is a single shared promise, and stdout is
  drained before exit.
- `lib.mjs` holds the pure helpers shared by both sides: run ids, estimate validation, quoting,
  static-path containment, the lock command, queue parsing and the idle verdict. They are tested
  in `tests/kernel_remote.test.mjs`.
- `bringup_matrix.mjs` is the bring-up helper. It tries every Chrome binary and flag set with
  `device-probe.html?bench=0` and writes JSONL. On Yukon it must run inside one gpu-queue job.
- `../../../lab/kernel-hillclimb/device-probe.{html,mjs}` is the proof page: adapter info,
  features, limits, f16/subgroup/subgroup-matrix shader compile checks, WASM SIMD/relaxed-SIMD
  and threads detection, an f32 (and f16) FMA microbench, and a vec4 copy bandwidth
  microbench, timed with timestamp-query when available. The FMA kernel is the one from
  `lab/kernel-hillclimb/wgsl-kernels.mjs` on `perf/kernel-hillclimb` (commit 6924d9a9). It uses
  per-lane constants and seeds and stores from every invocation, so no vector lane can be folded.
