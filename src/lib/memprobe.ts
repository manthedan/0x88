/** Lab-only memory sample. The layout loads this module only for ?memprobe=1. */

type MemoryPerformance = Performance & {
  memory?: { usedJSHeapSize?: number; totalJSHeapSize?: number };
  measureUserAgentSpecificMemory?: () => Promise<{ bytes?: number; breakdown?: Array<{ attribution?: Array<{ scope?: string }> }> }>;
};

export type MemprobeSample = {
  label: string;
  href: string;
  crossOriginIsolated: boolean;
  measureLatencyMs: number;
  userAgentSpecificBytes: number | null;
  userAgentSpecificError: string | null;
  usedJSHeapSize: number | null;
  workerScopes: number | null;
  hookedWorkersLive: number | null;
};

function mib(bytes: number | null): string {
  return bytes == null ? 'n/a' : `${(bytes / 1048576).toFixed(1)} MiB`;
}

export async function sampleMemprobe(label = 'manual'): Promise<MemprobeSample> {
  const started = performance.now();
  const perf = performance as MemoryPerformance;
  let bytes: number | null = null;
  let error: string | null = null;
  let workerScopes: number | null = null;
  if (!crossOriginIsolated) error = 'not cross-origin isolated';
  else if (!perf.measureUserAgentSpecificMemory) error = 'measureUserAgentSpecificMemory unavailable';
  else {
    try {
      const measured = await perf.measureUserAgentSpecificMemory();
      bytes = typeof measured.bytes === 'number' ? measured.bytes : null;
      workerScopes = (measured.breakdown ?? []).filter((row) => (row.attribution ?? []).some((item) => /worker/i.test(item.scope ?? ''))).length;
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    }
  }
  const hooked = (globalThis as { __memprobeWorkers?: { live?: number } }).__memprobeWorkers;
  const sample: MemprobeSample = {
    label,
    href: location.href,
    crossOriginIsolated,
    measureLatencyMs: Math.round(performance.now() - started),
    userAgentSpecificBytes: bytes,
    userAgentSpecificError: error,
    usedJSHeapSize: perf.memory?.usedJSHeapSize ?? null,
    workerScopes,
    hookedWorkersLive: typeof hooked?.live === 'number' ? hooked.live : null,
  };
  console.log('[memprobe]', sample);
  return sample;
}

export function mountMemprobe(): void {
  if (document.getElementById('memprobe')) return;
  const root = document.createElement('aside');
  root.id = 'memprobe';
  root.setAttribute('aria-label', 'Memory probe');
  root.style.cssText =
    'position:fixed;right:8px;bottom:8px;z-index:99999;max-width:260px;padding:8px;background:#141414;color:#f4f1e8;font:12px/1.35 ui-monospace,monospace;border:1px solid #666';
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Sample memory';
  const out = document.createElement('pre');
  out.style.cssText = 'margin:6px 0 0;white-space:pre-wrap';
  button.addEventListener('click', () => {
    out.textContent = 'measuring (waits for GC)…';
    void sampleMemprobe('overlay').then((sample) => {
      const workers = sample.hookedWorkersLive ?? sample.workerScopes;
      out.textContent = `${sample.label}\nUA ${mib(sample.userAgentSpecificBytes)}\nheap ${mib(sample.usedJSHeapSize)}\nworkers ${workers ?? 'n/a'}\n${sample.measureLatencyMs} ms${sample.userAgentSpecificError ? `\n${sample.userAgentSpecificError}` : ''}`;
    });
  });
  root.append(button, out);
  document.body.append(root);
  (window as unknown as { __memprobe?: { sample: typeof sampleMemprobe } }).__memprobe = { sample: sampleMemprobe };
  console.log('[memprobe] ready. __memprobe.sample(label) waits on measureUserAgentSpecificMemory.');
}
