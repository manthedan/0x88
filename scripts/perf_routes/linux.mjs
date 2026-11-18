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

/**
 * `nvidia-smi -q -d PIDS` -> Map(pid -> MiB) for every context type (C, G and C+G). The compute-only
 * `--query-compute-apps` list omits graphics-only contexts, and Chromium's Vulkan GPU process could be one
 * (Yukon's driver reports it as C+G). `-q -x` lists them too but takes ~130 ms against ~25 ms here.
 */
export function parseNvidiaProcesses(text) {
  const vram = new Map();
  let pid = null;
  for (const line of text.split('\n')) {
    const field = /^\s*([^:]+?)\s*:\s*(.*?)\s*$/.exec(line);
    if (!field) continue;
    const [, key, value] = field;
    if (key === 'Process ID') {
      if (pid !== null) throw Error(`nvidia-smi listed PID ${pid} without Used GPU Memory`);
      if (!/^\d+$/.test(value)) throw Error(`Unparseable nvidia-smi Process ID: ${value}`);
      pid = Number(value);
    } else if (key === 'Used GPU Memory') {
      const m = /^(\d+) MiB$/.exec(value);
      // "Not available"/"N/A" (no permission, WDDM-like drivers) must not read as zero.
      if (pid === null || !m) throw Error(`Unparseable nvidia-smi Used GPU Memory: ${value}`);
      // One process may hold contexts on several GPUs.
      vram.set(pid, (vram.get(pid) ?? 0) + Number(m[1]));
      pid = null;
    }
  }
  if (pid !== null) throw Error(`nvidia-smi listed PID ${pid} without Used GPU Memory`);
  // An empty map must come from a real report ("Processes : None"), not from unexpected output.
  if (!/^Attached GPUs\s*:/m.test(text)) throw Error('Unrecognized nvidia-smi -q -d PIDS output');
  return vram;
}

const queryNvidia = () =>
  new Promise((resolve, reject) =>
    execFile('nvidia-smi', ['-q', '-d', 'PIDS'], { encoding: 'utf8', timeout: 5000 }, (error, stdout) =>
      error ? reject(Error(`nvidia-smi failed (Linux GPU memory needs an NVIDIA driver): ${error.message}`)) : resolve(stdout),
    ),
  );

/**
 * Same contract as common.mjs readFootprint: Map(pid -> {cur, life, interval}) in MiB, unreadable PIDs
 * omitted; reset=true restarts each PID's interval peak at its current value. Rows also carry
 * `vram` (MiB, included in cur) so GPU allocations stay visible on their own. A PID nvidia-smi does
 * not list has `vram: null` (unavailable, not a measured zero); chromeFootprint rejects that for the
 * Chrome GPU process, which holds a Vulkan context from launch.
 */
export function linuxFootprintReader({ readText = (path) => readFile(path, 'utf8'), queryVram = queryNvidia } = {}) {
  const peaks = new Map();
  return async (pids, { reset = false } = {}) => {
    const out = new Map();
    if (!pids.length) return out;
    const [texts, vramText] = await Promise.all([Promise.all(pids.map((pid) => readText(`/proc/${pid}/smaps_rollup`).catch(() => null))), queryVram()]);
    const vram = parseNvidiaProcesses(vramText);
    pids.forEach((pid, i) => {
      if (texts[i] === null) return; // exited: same as pidfp's "err"
      const v = vram.get(pid) ?? null;
      const cur = parseSmapsRollup(texts[i]) + (v ?? 0);
      const seen = peaks.get(pid) ?? { life: cur, interval: cur };
      const row = { cur, life: Math.max(seen.life, cur), interval: Math.max(seen.interval, cur), vram: v };
      peaks.set(pid, { life: row.life, interval: reset ? cur : row.interval });
      out.set(pid, row);
    });
    return out;
  };
}

// Headless Chromium on Linux exposes the NVIDIA Vulkan adapter only with these (verified on Yukon,
// Chromium 149 / driver 580, no display). Without Vulkan, ANGLE and Dawn fall back to SwiftShader; no
// surface exists headless. --enable-unsafe-webgpu is not needed, so users' WebGPU blocklist applies.
// Dawn withholds shader-f16 on NVIDIA Vulkan drivers older than 615.71 (crbug.com/42251215) unless
// vulkan_enable_f16_on_nvidia is set. Without shader-f16 the app runs every ORT session (Lc0, Maia3)
// on WASM, so the run would measure the CPU fallback, not the WebGPU path that Windows/NVIDIA (D3D12)
// users get. Linux Chrome users on such drivers do get WASM; this datapoint stands in for Windows.
export const LINUX_CHROME_ARGS = ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface', '--enable-dawn-features=vulkan_enable_f16_on_nvidia'];

export const ADAPTER_INFO = `(async () => {
  const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  return a && { vendor: a.info.vendor, architecture: a.info.architecture, device: a.info.device, description: a.info.description, fallback: a.info.isFallbackAdapter, shaderF16: a.features.has('shader-f16') };
})()`;

/**
 * Headless Chromium returns null for requestAdapter() calls made in the first seconds after launch
 * (Yukon: the first call fails, later ones succeed; under Xvfb it never fails). The app's first
 * request happens right at page load and would silently fall back to WASM. Poll from a throwaway
 * context, closed before the measured context exists, until the GPU process returns a hardware adapter.
 * `isRendererGone` lets the caller wait for the throwaway renderer to exit so it is never sampled; the
 * warm-up fails if that exit is not confirmed.
 */
export async function warmWebGpu(browser, isRendererGone, { attempts = 40, delayMs = 250, exitPolls = 50, exitDelayMs = 100 } = {}) {
  const started = Date.now();
  const context = await browser.newContext();
  let adapter, tries;
  try {
    // A secure-context origin served by Playwright itself: no request reaches the counting proxy.
    await context.route('**/*', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>warm-up</title>' }));
    const page = await context.newPage();
    await page.goto('http://127.0.0.1/webgpu-warmup');
    for (tries = 1; tries <= attempts && !adapter; tries++) {
      adapter = await page.evaluate(ADAPTER_INFO);
      if (!adapter) await new Promise((r) => setTimeout(r, delayMs));
    }
  } finally {
    await context.close();
  }
  if (!adapter || adapter.fallback || /swiftshader|llvmpipe/i.test(JSON.stringify(adapter)))
    throw Error(`No hardware WebGPU adapter: ${JSON.stringify(adapter)}`);
  // The app treats an adapter without shader-f16 as no adapter: every model would silently run on WASM.
  if (!adapter.shaderF16) throw Error(`WebGPU adapter lacks shader-f16; the app would run its models on WASM: ${JSON.stringify(adapter)}`);
  // A lingering warm-up renderer would be sampled as part of the measured footprint.
  for (let i = 0; !(await isRendererGone()); i++) {
    if (i >= exitPolls) throw Error(`WebGPU warm-up renderer still running after ${exitPolls * exitDelayMs} ms`);
    await new Promise((r) => setTimeout(r, exitDelayMs));
  }
  return { ...adapter, attempts: tries - 1, ms: Date.now() - started };
}

export const LINUX_LIMITATION =
  'Linux/NVIDIA: memory = smaps_rollup Pss_Dirty+SwapPss per process plus nvidia-smi VRAM (all context types) for the GPU process; peaks are 100 ms sampled maxima (no kernel high-water). Chromium runs headless with ANGLE/Vulkan and Dawn vulkan_enable_f16_on_nvidia, after a WebGPU warm-up in a throwaway context. Not comparable with macOS phys_footprint results.';
