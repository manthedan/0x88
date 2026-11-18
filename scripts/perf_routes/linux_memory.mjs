// Linux (NVIDIA) stand-in for the macOS pidfp footprint reader. A second datapoint for Windows/NVIDIA-like
// users; its numbers are never comparable with macOS phys_footprint (the gate refuses mixed platforms).
//
// Host memory per PID = Pss_Dirty + SwapPss from /proc/<pid>/smaps_rollup (kB = KiB):
//  - phys_footprint excludes clean file-backed pages, so plain PSS is wrong: the GPU process alone maps
//    ~130 MiB of shared-library text (Pss_File).
//  - phys_footprint counts dirty memory the process shares. Private_Dirty drops every page also mapped
//    elsewhere: the renderer<->GPU shared-memory transfer/mapping buffers WebGPU uses, and zygote COW pages.
//    Pss_Dirty keeps them, split by mapper count, so the renderer+GPU sum counts a shared page once.
//  - phys_footprint includes compressed pages; SwapPss is the swapped-out share.
// The GPU process also gets its device allocations from nvidia-smi (MiB), because on Apple unified memory
// Metal allocations are part of the GPU process's phys_footprint. NVIDIA reports VRAM held by each process's
// contexts (driver granularity, not WebGPU buffer sizes); host-pinned staging memory lives in device-file
// mappings that smaps does not count.
//
// There is no kernel high-water mark for these counters: life/interval peaks are maxima of the reads
// (~100 ms apart), so sub-interval spikes are missed, unlike pidfp. smaps_rollup walks page tables under
// the target's mmap read lock (a few ms per process).
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';

const MiB = 1024; // smaps reports kB (KiB)

export function parseSmapsRollup(text) {
  const kb = {};
  for (const m of text.matchAll(/^(\w+):\s+(\d+) kB$/gm)) kb[m[1]] = Number(m[2]);
  // Pss_Dirty needs Linux >= 6.0; never substitute a different counter silently.
  for (const key of ['Pss_Dirty', 'SwapPss']) if (!Number.isFinite(kb[key])) throw Error(`smaps_rollup lacks ${key}`);
  return (kb.Pss_Dirty + kb.SwapPss) / MiB;
}

/** `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` -> Map(pid -> MiB). */
export function parseNvidiaComputeApps(text) {
  const vram = new Map();
  for (const line of text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)) {
    const m = /^(\d+),\s*(\d+)$/.exec(line);
    // "[N/A]" (no permission, WDDM-like drivers) must not read as zero.
    if (!m) throw Error(`Unparseable nvidia-smi compute-apps line: ${line}`);
    // One process may hold contexts on several GPUs.
    vram.set(Number(m[1]), (vram.get(Number(m[1])) ?? 0) + Number(m[2]));
  }
  return vram;
}

const queryNvidia = () =>
  new Promise((resolve, reject) =>
    execFile('nvidia-smi', ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], { encoding: 'utf8', timeout: 5000 }, (error, stdout) =>
      error ? reject(Error(`nvidia-smi failed (Linux GPU memory needs an NVIDIA driver): ${error.message}`)) : resolve(stdout),
    ),
  );

/**
 * Same contract as common.mjs readFootprint: Map(pid -> {cur, life, interval}) in MiB, unreadable PIDs
 * omitted; reset=true restarts each PID's interval peak at its current value. Rows also carry
 * `vram` (MiB, included in cur) so GPU allocations stay visible on their own.
 */
export function linuxFootprintReader({ readText = (path) => readFile(path, 'utf8'), queryVram = queryNvidia } = {}) {
  const peaks = new Map();
  return async (pids, { reset = false } = {}) => {
    const out = new Map();
    if (!pids.length) return out;
    const [texts, vramText] = await Promise.all([Promise.all(pids.map((pid) => readText(`/proc/${pid}/smaps_rollup`).catch(() => null))), queryVram()]);
    const vram = parseNvidiaComputeApps(vramText);
    pids.forEach((pid, i) => {
      if (texts[i] === null) return; // exited: same as pidfp's "err"
      const v = vram.get(pid) ?? 0;
      const cur = parseSmapsRollup(texts[i]) + v;
      const seen = peaks.get(pid) ?? { life: cur, interval: cur };
      const row = { cur, life: Math.max(seen.life, cur), interval: Math.max(seen.interval, cur), vram: v };
      peaks.set(pid, { life: row.life, interval: reset ? cur : row.interval });
      out.set(pid, row);
    });
    return out;
  };
}

// Headless Chromium on Linux exposes the NVIDIA Vulkan adapter only with these (verified on Yukon,
// Chromium 149 / driver 580, no display): WebGPU is blocklisted there without --enable-unsafe-webgpu,
// and ANGLE/GL would fall back to SwiftShader without Vulkan. No surface exists headless.
// requestAdapter() without powerPreference still returns null headless; the app always asks for
// high-performance, and the session asserts that adapter is hardware.
export const LINUX_CHROME_ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface'];

export const LINUX_LIMITATION =
  'Linux/NVIDIA: memory = smaps_rollup Pss_Dirty+SwapPss per process plus nvidia-smi VRAM for the GPU process; peaks are 100 ms sampled maxima (no kernel high-water). Chromium runs with --enable-unsafe-webgpu and ANGLE/Vulkan. Not comparable with macOS phys_footprint results.';
