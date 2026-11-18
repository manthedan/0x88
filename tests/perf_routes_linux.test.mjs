import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { browserIdentity, chromeFootprint, compareRuns, footprintSum, parseArgs, plan, platformProblem } from '../scripts/perf_routes/gate.mjs';
import {
  LINUX_CHROME_ARGS,
  linuxFootprintReader,
  parseNvidiaProcesses,
  parseSmapsRollup,
  warmWebGpu,
  webgpuBackendProblem,
} from '../scripts/perf_routes/linux.mjs';
import { selectSessions, sessionId } from '../scripts/perf_routes/resume.mjs';

// Captured on Yukon (Linux 7.0, Chromium 149, RTX 3090): headless GPU process after a 256 MiB WebGPU allocation.
const GPU_SMAPS = `00010000-7ffcb8118000 ---p 00000000 00:00 0                              [rollup]
Rss:              277876 kB
Pss:              215139 kB
Pss_Dirty:         68450 kB
Pss_Anon:          56440 kB
Pss_File:         158691 kB
Pss_Shmem:             8 kB
Shared_Clean:      62744 kB
Shared_Dirty:      25824 kB
Private_Clean:    133768 kB
Private_Dirty:     55540 kB
Referenced:       265688 kB
Anonymous:         63388 kB
KSM:                   0 kB
LazyFree:              0 kB
AnonHugePages:         0 kB
ShmemPmdMapped:        0 kB
FilePmdMapped:         0 kB
Shared_Hugetlb:        0 kB
Private_Hugetlb:       0 kB
Swap:                  0 kB
SwapPss:               0 kB
Locked:                0 kB
`;
const RENDERER_SMAPS = GPU_SMAPS.replace(/Pss_Dirty:\s+\d+/, 'Pss_Dirty:          9578').replace(/SwapPss:\s+\d+/, 'SwapPss:            1024');

test('smaps_rollup footprint is dirty PSS plus swapped PSS, never plain PSS/RSS', () => {
  assert.equal(parseSmapsRollup(GPU_SMAPS), 68450 / 1024);
  assert.equal(parseSmapsRollup(RENDERER_SMAPS), (9578 + 1024) / 1024);
  // Pre-6.0 kernels have no Pss_Dirty: refuse rather than substitute another counter.
  assert.throws(() => parseSmapsRollup(GPU_SMAPS.replace(/^Pss_Dirty:.*\n/m, '')), /lacks Pss_Dirty/);
  assert.throws(() => parseSmapsRollup(''), /lacks/);
});

// `nvidia-smi -q -d PIDS` captured on Yukon (driver 580) during a session; the Chromium GPU process is C+G.
const pidsReport = (processes) => `
==============NVSMI LOG==============

Timestamp                                              : Wed Nov 18 01:06:44 2026
Driver Version                                         : 580.178.04
CUDA Version                                           : 13.0

Attached GPUs                                          : 1
GPU 00000000:2A:00.0
${processes}
`;
const proc = (pid, type, used) => `        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : ${pid}
            Type                          : ${type}
            Name                          : /home/ddbb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome --type=gpu-process --x=a:b
            Used GPU Memory               : ${used}`;
const vramReport = (...procs) =>
  pidsReport(procs.length ? `    Processes\n${procs.join('\n')}` : '    Processes                                          : None');

test('nvidia-smi PIDS report parses every context type to MiB per PID; unavailable values are errors, not zero', () => {
  assert.deepEqual(parseNvidiaProcesses(vramReport(proc(313973, 'C+G', '90 MiB'))), new Map([[313973, 90]]));
  // Graphics-only contexts are absent from --query-compute-apps but count here.
  assert.deepEqual(
    parseNvidiaProcesses(vramReport(proc(5, 'G', '40 MiB'), proc(6, 'C', '2 MiB'))),
    new Map([
      [5, 40],
      [6, 2],
    ]),
  );
  assert.deepEqual(parseNvidiaProcesses(vramReport()), new Map());
  assert.deepEqual(
    parseNvidiaProcesses(vramReport(proc(7, 'C+G', '10 MiB'), proc(8, 'G', '1 MiB'), proc(7, 'C', '5 MiB'))),
    new Map([
      [7, 15],
      [8, 1],
    ]),
  );
  assert.throws(() => parseNvidiaProcesses(vramReport(proc(7, 'G', 'Not available'))), /Used GPU Memory: Not available/);
  assert.throws(() => parseNvidiaProcesses(vramReport(proc(7, 'G', '10 MiB').replace(/.*Used GPU Memory.*/, ''))), /without Used GPU Memory/);
  assert.throws(() => parseNvidiaProcesses(''), /Unrecognized/);
  assert.throws(() => parseNvidiaProcesses('No devices were found\n'), /Unrecognized/);
});

test('Linux reader adds VRAM to the GPU process, omits exited PIDs and tracks sampled peaks with reset', async () => {
  const smaps = new Map([
    [10, RENDERER_SMAPS],
    [20, GPU_SMAPS],
  ]);
  let vram = vramReport(proc(20, 'C+G', '300 MiB'));
  const read = linuxFootprintReader({
    readText: async (path) => {
      const text = smaps.get(Number(/\/proc\/(\d+)\//.exec(path)[1]));
      if (!text) throw Error('ENOENT');
      return text;
    },
    queryVram: async () => vram,
  });
  const gpuHost = 68450 / 1024,
    renderer = (9578 + 1024) / 1024;
  let rows = await read([10, 20, 99], { reset: true });
  assert.equal(rows.has(99), false);
  assert.deepEqual(rows.get(20), { cur: gpuHost + 300, life: gpuHost + 300, interval: gpuHost + 300, vram: 300 });
  assert.deepEqual(rows.get(10), { cur: renderer, life: renderer, interval: renderer, vram: null }, 'unlisted: unavailable, not zero');
  vram = vramReport(proc(20, 'C+G', '100 MiB'));
  rows = await read([20]);
  assert.deepEqual(rows.get(20), { cur: gpuHost + 100, life: gpuHost + 300, interval: gpuHost + 300, vram: 100 });
  rows = await read([20], { reset: true });
  assert.equal(rows.get(20).interval, gpuHost + 300, 'the resetting read still reports the interval it closes');
  rows = await read([20]);
  assert.equal(rows.get(20).interval, gpuHost + 100);
  assert.equal(rows.get(20).life, gpuHost + 300);
  assert.deepEqual(await read([]), new Map());
  vram = vramReport(proc(20, 'C+G', 'Not available'));
  await assert.rejects(read([20]), /Unparseable/);
});

test('Chrome sampling accepts the async Linux reader and keeps VRAM separately visible', async () => {
  const pids = { rendererPids: [1, 2], gpuPids: [3] };
  const rows = new Map([
    [1, { cur: 10, life: 12, interval: 11, vram: 0 }],
    [2, { cur: 5, life: 6, interval: 5, vram: 0 }],
    [3, { cur: 400, life: 500, interval: 450, vram: 300 }],
  ]);
  const sample = await chromeFootprint(
    async () => pids,
    async () => rows,
  );
  assert.deepEqual(sample.render, { cur: 15, life: 18, interval: 16, vram: 0 });
  assert.deepEqual(sample.gpu, { cur: 400, life: 500, interval: 450, vram: 300 });
  // A GPU process nvidia-smi does not list is an unavailable reading, never zero VRAM.
  rows.set(3, { cur: 100, life: 100, interval: 100, vram: null });
  await assert.rejects(
    chromeFootprint(
      async () => pids,
      async () => rows,
    ),
    /does not list Chrome GPU PID 3: VRAM unavailable/,
  );
  // Unlisted renderers (no GPU context) keep the renderer sum readable; its VRAM is unavailable.
  rows.set(3, { cur: 400, life: 500, interval: 450, vram: 300 });
  rows.set(2, { cur: 5, life: 6, interval: 5, vram: null });
  const unlisted = await chromeFootprint(
    async () => pids,
    async () => rows,
  );
  assert.deepEqual(unlisted.render, { cur: 15, life: 18, interval: 16, vram: null });
  // macOS rows stay exactly as before: no vram key appears.
  assert.deepEqual(footprintSum([1], new Map([[1, { cur: 1, life: 2, interval: 3 }]]), 'renderer'), { cur: 1, life: 2, interval: 3 });
});

test('Linux runs only Chrome; macOS keeps every browser', () => {
  const dry = (...extra) => parseArgs(['--baseline', 'main', '--candidate', 'HEAD', '--dry-run', ...extra]);
  assert.equal(platformProblem(dry(), 'darwin'), null);
  assert.equal(platformProblem(dry('--browser', 'safari'), 'darwin'), null);
  assert.equal(platformProblem(dry('--browser', 'chrome'), 'linux'), null);
  assert.match(platformProblem(dry(), 'linux'), /--browser chrome/);
  assert.match(platformProblem({ browser: 'safari' }, 'linux'), /--browser chrome/);
  assert.match(platformProblem(dry('--browser', 'chrome'), 'win32'), /macOS or Linux only/);
});

test('the gate refuses to compare macOS and Linux runs and requires VRAM on Linux', () => {
  const options = parseArgs(['--baseline', 'a', '--candidate', 'b', '--dry-run', '--browser', 'chrome', '--journey', 'navigation']);
  const rows = (platformOf) =>
    plan(options).map((cell, i) => ({ ...cell, platform: platformOf(i), host: { platform: platformOf(i) }, status: 'ok', metrics: { jsBytes: 10 } }));
  assert.throws(
    () =>
      compareRuns(
        rows((i) => (i ? 'linux' : 'darwin')),
        options,
      ),
    /different platforms: darwin, linux/,
  );
  // Older macOS rows carry only host.platform.
  const legacy = rows(() => 'darwin').map(({ platform, ...row }, i) => (i ? row : { ...row, platform: 'linux' }));
  assert.throws(() => compareRuns(legacy, options), /different platforms/);
  const [linux] = compareRuns(
    rows(() => 'linux'),
    options,
  );
  assert.equal(linux.platform, 'linux');
  assert.equal(linux.metrics.find((m) => m.name === 'memory.gpuVramPeakMb').status, 'unavailable');
  assert.equal(linux.status, 'incomplete');
  const [mac] = compareRuns(
    rows(() => 'darwin'),
    options,
  );
  assert.equal(mac.platform, 'darwin');
  assert.equal(
    mac.metrics.some((m) => m.name === 'memory.gpuVramPeakMb'),
    false,
  );
});

test('Linux Chrome flags and Chromium channel apply only on Linux; macOS launches installed Chrome unchanged', () => {
  const source = readFileSync(new URL('../scripts/perf_routes/session.mjs', import.meta.url), 'utf8');
  assert.match(source, /linux \? \{ channel: 'chromium', executablePath: process\.env\.PERF_ROUTES_CHROMIUM \|\| undefined \} : \{ channel: 'chrome' \}/);
  assert.match(source, /args: \['--enable-blink-features=ForceEagerMeasureMemory', \.\.\.\(linux \? LINUX_CHROME_ARGS : \[\]\)\]/);
  assert.ok(LINUX_CHROME_ARGS.includes('--enable-features=Vulkan'));
  assert.ok(!LINUX_CHROME_ARGS.includes('--enable-unsafe-webgpu'), 'the WebGPU blocklist stays as users have it');
  assert.ok(LINUX_CHROME_ARGS.includes('--enable-dawn-features=vulkan_enable_f16_on_nvidia'), 'NVIDIA drivers < 615.71 hide shader-f16 otherwise');
  const warm = source.indexOf('await warmWebGpu(');
  assert.ok(warm > source.indexOf('newBrowserCDPSession()') && warm < source.indexOf('browser.newContext('), 'warm-up precedes the measured context');
  assert.ok(warm < source.indexOf('sampling = setInterval(sample, 100)'), 'warm-up precedes sampling');
});

test('WebGPU warm-up retries the early null adapter, rejects software and f16-less adapters, requires the renderer exit and always closes its context', async () => {
  const fakeBrowser = (answers) => {
    const state = { closed: 0, routed: 0, evaluations: 0 };
    return {
      state,
      newContext: async () => ({
        route: async () => state.routed++,
        newPage: async () => ({ goto: async () => {}, evaluate: async () => (state.evaluations++, answers.shift() ?? null) }),
        close: async () => state.closed++,
      }),
    };
  };
  const nvidia = { vendor: 'nvidia', architecture: 'ampere', device: '', description: '', fallback: false, shaderF16: true };
  let gone = 0;
  const ok = fakeBrowser([null, nvidia]);
  const info = await warmWebGpu(ok, async () => ++gone > 2, { delayMs: 0 });
  assert.equal(info.attempts, 2);
  assert.equal(info.vendor, 'nvidia');
  assert.deepEqual(ok.state, { closed: 1, routed: 1, evaluations: 2 });
  assert.equal(gone, 3, 'waits for the throwaway renderer to exit');
  for (const answers of [[], [{ ...nvidia, fallback: true }], [{ vendor: 'google', description: 'SwiftShader Device', fallback: false }]]) {
    const browser = fakeBrowser(answers);
    await assert.rejects(
      warmWebGpu(browser, async () => true, { attempts: 3, delayMs: 0 }),
      /No hardware WebGPU adapter/,
    );
    assert.equal(browser.state.closed, 1);
  }
  const lingering = fakeBrowser([nvidia]);
  let polls = 0;
  await assert.rejects(
    warmWebGpu(lingering, async () => (polls++, false), { delayMs: 0, exitPolls: 3, exitDelayMs: 0 }),
    /warm-up renderer still running/,
  );
  assert.equal(polls, 4, 'checks once more after the last wait');
  assert.equal(lingering.state.closed, 1);
  const noF16 = fakeBrowser([{ ...nvidia, shaderF16: false }]);
  await assert.rejects(
    warmWebGpu(noF16, async () => true, { attempts: 3, delayMs: 0 }),
    /lacks shader-f16/,
  );
  assert.equal(noF16.state.closed, 1);
});

test('WebGPU warm-up gives up at its wall-clock deadline or on shutdown and closes what it opened', async () => {
  const stalled = () => {
    const state = { closed: 0, lateClosed: 0 };
    const context = {
      route: async () => {},
      newPage: async () => ({ goto: async () => {}, evaluate: () => new Promise(() => {}) }),
      close: async () => state.closed++,
    };
    return { state, newContext: async () => context };
  };
  const timedOut = stalled();
  await assert.rejects(
    warmWebGpu(timedOut, async () => true, { timeoutMs: 20 }),
    /WebGPU warm-up exceeded 20 ms/,
  );
  assert.equal(timedOut.state.closed, 1, 'a stalled requestAdapter() does not keep the context open');
  const shutdown = new AbortController();
  const aborted = stalled();
  const warming = warmWebGpu(aborted, async () => true, { signal: shutdown.signal });
  setTimeout(() => shutdown.abort(Error('Interrupted by SIGTERM')), 10);
  await assert.rejects(warming, /Interrupted by SIGTERM/);
  assert.equal(aborted.state.closed, 1);
  // A context that arrives after the deadline is closed when it lands.
  let arrive;
  const late = { close: async () => late.closed++, closed: 0 };
  await assert.rejects(
    warmWebGpu({ newContext: () => new Promise((resolve) => (arrive = resolve)) }, async () => true, { timeoutMs: 10 }),
    /exceeded 10 ms/,
  );
  arrive(late);
  await new Promise((r) => setImmediate(r));
  assert.equal(late.closed, 1);
});

test('Linux sessions refuse journeys whose ORT sessions ran on WASM or reported no backend', () => {
  // Shapes captured on Yukon: CPU engine workers report no backend; Play's Maia3 lists its fallback provider.
  const cpuEngine = { backend: undefined };
  assert.equal(webgpuBackendProblem('analysis-default', [{ backend: 'auto->webgpu', ortIoMode: 'gpu-capture' }], []), null);
  assert.equal(webgpuBackendProblem('analysis-four', [{ backend: 'auto->webgpu' }, cpuEngine, cpuEngine], []), null);
  assert.equal(webgpuBackendProblem('play-maia3', [{ backend: 'auto->webgpu,wasm' }], []), null);
  assert.equal(webgpuBackendProblem('arena-lc0', [], ['webgpu']), null, "Arena's main-thread Lc0 reports through the console");
  assert.equal(webgpuBackendProblem('arena-cpu', [cpuEngine, cpuEngine], []), null, 'no neural engine to check');
  assert.match(webgpuBackendProblem('analysis-default', [{ backend: 'auto->wasm', ortIoMode: 'cpu' }], []), /ORT ran on WASM, not WebGPU \(auto->wasm\)/);
  assert.match(webgpuBackendProblem('arena-lc0', [], ['wasm']), /ORT ran on WASM/);
  assert.match(webgpuBackendProblem('navigation', [{ backend: 'auto->webgpu' }, { backend: 'auto->wasm' }], []), /auto->wasm/);
  assert.match(webgpuBackendProblem('arena-lc0', [cpuEngine], []), /No ORT backend observed for arena-lc0/);
  const source = readFileSync(new URL('../scripts/perf_routes/session.mjs', import.meta.url), 'utf8');
  const ready = source.indexOf('result.workerReady = await evaluate(');
  const check = source.indexOf('webgpuBackendProblem(spec.journey, result.workerReady, result.ortSessions)');
  assert.ok(check > ready && check < source.indexOf("result.status = 'ok'"), 'checked once workers have reported, before the session counts');
  assert.ok(source.indexOf("page.on('console'") < source.indexOf('await navigate(entryUrl())'), 'listening before the app loads');
  assert.match(source, /warmWebGpu\(browser, [^\n]*\{ signal \}\)/, 'warm-up observes the session shutdown signal');
});

test('a browser build that differs from the cohort is refused before measuring and re-runs on resume', () => {
  const source = readFileSync(new URL('../scripts/perf_routes/session.mjs', import.meta.url), 'utf8');
  const claim = source.indexOf('claimBrowser(browser.version()');
  assert.ok(claim > 0 && claim < source.indexOf('await warmWebGpu('), 'checked right after launch, before the warm-up');
  assert.ok(!/result\.browserVersion =/.test(source), 'only claimBrowser records an identity');
  const coordinator = readFileSync(new URL('../scripts/perf_routes/run.mjs', import.meta.url), 'utf8');
  assert.match(coordinator, /expectedBrowser: cohortBrowser\[cell\.browser\] \?\? null/);
  const attempt = coordinator.slice(coordinator.indexOf('const row = recordAttempt('));
  assert.ok(attempt.indexOf('if (row.browserMismatch') < attempt.indexOf("appendFileSync(join(out, 'runs.jsonl')"), 'never appended');
  const options = parseArgs(['--baseline', 'a', '--candidate', 'b', '--dry-run', '--browser', 'chrome', '--journey', 'navigation']);
  const cells = plan(options);
  const refused = { ...cells[0], status: 'invalid', error: 'Error: Browser build changed mid-cohort', browserMismatch: '150.0 @ /c/other/chrome' };
  const selected = selectSessions(cells, (id) => ({ result: id === sessionId(cells[0], 0) ? refused : null }));
  assert.equal(selected[0].reuse, null, 'a refused build is not an outcome');
});

test('the gate refuses to combine sessions from different browser builds (browser update or PERF_ROUTES_CHROMIUM change on resume)', () => {
  const options = parseArgs(['--baseline', 'a', '--candidate', 'b', '--dry-run', '--browser', 'chrome', '--journey', 'navigation']);
  const rows = (identity) => plan(options).map((cell, i) => ({ ...cell, platform: 'linux', status: 'ok', metrics: { jsBytes: 10 }, ...identity(i) }));
  const chromium = (version, executable) => ({ browserVersion: version, browserExecutable: executable });
  const same = chromium('149.0.7827.55', '/c/chromium-1228/chrome');
  assert.equal(browserIdentity(same), '149.0.7827.55 @ /c/chromium-1228/chrome');
  assert.equal(browserIdentity({ browserVersion: '27.0' }), '27.0', 'macOS rows carry no executable');
  assert.equal(browserIdentity({}), null);
  assert.doesNotThrow(() =>
    compareRuns(
      rows(() => same),
      options,
    ),
  );
  // A session that failed before launching a browser has no identity and does not count as a different build.
  assert.doesNotThrow(() =>
    compareRuns(
      rows((i) => (i ? same : {})),
      options,
    ),
  );
  assert.throws(
    () =>
      compareRuns(
        rows((i) => (i ? same : chromium('150.0.1.0', same.browserExecutable))),
        options,
      ),
    /different browser builds: 150\.0\.1\.0 @/,
  );
  assert.throws(
    () =>
      compareRuns(
        rows((i) => (i ? same : chromium(same.browserVersion, '/c/chromium-1223/chrome'))),
        options,
      ),
    /different browser builds/,
  );
});
