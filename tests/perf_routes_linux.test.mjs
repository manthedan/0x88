import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { chromeFootprint, compareRuns, footprintSum, parseArgs, plan, platformProblem } from '../scripts/perf_routes/gate.mjs';
import { LINUX_CHROME_ARGS, linuxFootprintReader, parseNvidiaComputeApps, parseSmapsRollup } from '../scripts/perf_routes/linux_memory.mjs';

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

test('nvidia-smi compute apps parse to MiB per PID; unavailable values are errors, not zero', () => {
  assert.deepEqual(parseNvidiaComputeApps('190094, 287\n'), new Map([[190094, 287]]));
  assert.deepEqual(parseNvidiaComputeApps(''), new Map());
  assert.deepEqual(
    parseNvidiaComputeApps('7, 10\n8, 1\n7, 5\n'),
    new Map([
      [7, 15],
      [8, 1],
    ]),
  );
  assert.throws(() => parseNvidiaComputeApps('190094, [N/A]\n'), /Unparseable/);
  assert.throws(() => parseNvidiaComputeApps('No running processes found\n'), /Unparseable/);
});

test('Linux reader adds VRAM to the GPU process, omits exited PIDs and tracks sampled peaks with reset', async () => {
  const smaps = new Map([
    [10, RENDERER_SMAPS],
    [20, GPU_SMAPS],
  ]);
  let vram = '20, 300\n';
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
  assert.deepEqual(rows.get(10), { cur: renderer, life: renderer, interval: renderer, vram: 0 });
  vram = '20, 100\n';
  rows = await read([20]);
  assert.deepEqual(rows.get(20), { cur: gpuHost + 100, life: gpuHost + 300, interval: gpuHost + 300, vram: 100 });
  rows = await read([20], { reset: true });
  assert.equal(rows.get(20).interval, gpuHost + 300, 'the resetting read still reports the interval it closes');
  rows = await read([20]);
  assert.equal(rows.get(20).interval, gpuHost + 100);
  assert.equal(rows.get(20).life, gpuHost + 300);
  assert.deepEqual(await read([]), new Map());
  vram = '20, [N/A]\n';
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
  assert.ok(LINUX_CHROME_ARGS.includes('--enable-unsafe-webgpu'));
  assert.ok(source.indexOf('No hardware WebGPU adapter') > source.indexOf('await uaMemory('), 'adapter check runs after natural observations');
});
