// Landing page: browser capability badges and the downloads/storage manager.

const SHELL_CACHE_NAME = '0x88-app-shell-v2';
const KNOWN_CACHES = [
  { name: 'lc0-browser-models-v1', label: 'Leela networks', detail: 'sha256-validated LC0 small and Queen Odds model cache' },
  { name: 'maia3-browser-models-v1', label: 'Maia3 human model', detail: 'sha256-validated Maia3 model cache' },
  { name: SHELL_CACHE_NAME, label: 'App shell', detail: 'offline cache: pages and local runtime files' },
];

function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing element #${id}`);
  return found;
}

function mb(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

async function detectCapabilities(signal: AbortSignal): Promise<void> {
  const caps = el('caps');
  const note = el('capNote');
  const out: string[] = [];
  let webgpu = false;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu;
  try { webgpu = !!(await gpu?.requestAdapter?.()); } catch { webgpu = false; }
  if (signal.aborted) return;
  out.push(`<span class="cap ${webgpu ? 'ok' : 'no'}">WebGPU ${webgpu ? '✓' : '✗'}</span>`);
  const wasm = typeof WebAssembly !== 'undefined';
  out.push(`<span class="cap ${wasm ? 'ok' : 'no'}">WebAssembly ${wasm ? '✓' : '✗'}</span>`);
  const isolated = globalThis.crossOriginIsolated === true;
  out.push(`<span class="cap ${isolated ? 'ok' : ''}">Threads ${isolated ? '✓' : '—'}</span>`);
  const cores = navigator.hardwareConcurrency ?? 1;
  out.push(`<span class="cap">${cores} cores</span>`);
  caps.innerHTML = out.join('');
  if (!wasm) note.textContent = 'This browser cannot run the engines — WebAssembly is unavailable.';
  else if (!webgpu) note.textContent = 'All CPU engines and the small Leela net will work here. The big Leela nets (t3, BT4) need WebGPU — available in current Chrome, Edge, and Safari.';
  else note.textContent = 'Everything works here, including the WebGPU-accelerated Leela nets.';
}

interface CacheUsage {
  name: string;
  label: string;
  detail: string;
  present: boolean;
  entries: number;
  /** Sum of Content-Length metadata, or null when any entry has no length. */
  bytes: number | null;
}

async function measureCache(name: string, label: string, detail: string, signal: AbortSignal): Promise<CacheUsage> {
  if (!(await caches.has(name))) return { name, label, detail, present: false, entries: 0, bytes: 0 };
  if (signal.aborted) return { name, label, detail, present: false, entries: 0, bytes: null };
  const cache = await caches.open(name);
  const keys = await cache.keys();
  let bytes = 0;
  let sizesKnown = true;
  for (const request of keys) {
    if (signal.aborted) break;
    const response = await cache.match(request);
    if (!response) continue;
    const raw = response.headers.get('content-length');
    const length = raw === null ? NaN : Number(raw);
    if (Number.isFinite(length) && length >= 0) bytes += length;
    else sizesKnown = false;
  }
  // Do not materialize cached model bodies just to measure them. Cache entries
  // can be hundreds of MB; reading each body made the landing page allocate and
  // scan the entire model cache.
  return { name, label, detail, present: true, entries: keys.length, bytes: sizesKnown ? bytes : null };
}

async function renderStorage(signal: AbortSignal): Promise<void> {
  const root = el('storage');
  if (typeof caches === 'undefined') {
    root.innerHTML = '<p class="capnote">Storage management needs a secure context (https or localhost).</p>';
    return;
  }
  root.innerHTML = '<p class="capnote">Measuring…</p>';
  const usages = await Promise.all(KNOWN_CACHES.map(({ name, label, detail }) => measureCache(name, label, detail, signal)));
  const estimate = await Promise.resolve(navigator.storage?.estimate?.()).catch(() => undefined);
  if (signal.aborted) return;
  const rows = usages.map((usage) => {
    const size = usage.bytes === null ? 'size unavailable' : mb(usage.bytes);
    return `
    <div class="store-row" data-cache="${usage.name}">
      <div class="store-info"><b>${usage.label}</b><span>${usage.detail}</span></div>
      <span class="store-size">${usage.present ? `${size} · ${usage.entries} file${usage.entries === 1 ? '' : 's'}` : 'empty'}</span>
      <button type="button" data-clear="${usage.name}" ${usage.present ? '' : 'disabled'}>Clear</button>
    </div>`;
  }).join('');
  const totalLine = estimate?.usage !== undefined
    ? `<p class="capnote">This site uses ${mb(estimate.usage)}${estimate.quota ? ` of the ${mb(estimate.quota)} the browser allows` : ''}. R2-hosted engine files such as Reckless, Berserk, Viridithas, and PlentyChess live in the browser HTTP cache, so they may not appear in these Cache Storage rows. Everything re-downloads automatically when needed.</p>`
    : '<p class="capnote">R2-hosted engine files such as Reckless, Berserk, Viridithas, and PlentyChess live in the browser HTTP cache, so they may not appear in these Cache Storage rows. Cleared files re-download automatically when needed.</p>';
  root.innerHTML = rows + totalLine;
  for (const button of root.querySelectorAll<HTMLButtonElement>('button[data-clear]')) {
    button.addEventListener('click', async () => {
      const cacheName = button.dataset.clear ?? '';
      const row = button.closest('.store-row');
      const sizeText = row?.querySelector('.store-size')?.textContent ?? '';
      const label = row?.querySelector('b')?.textContent ?? 'this cache';
      if (button.dataset.confirming !== '1') {
        button.dataset.confirming = '1';
        button.textContent = `Clear ${sizeText ? `(${sizeText})` : ''}?`;
        button.classList.add('clearing');
        const reset = (): void => {
          button.dataset.confirming = '0';
          button.textContent = 'Clear';
          button.classList.remove('clearing');
        };
        button.addEventListener('blur', reset, { once: true });
        setTimeout(() => { if (button.dataset.confirming === '1') reset(); }, 4000);
        return;
      }
      button.disabled = true;
      button.textContent = `Clearing ${label}…`;
      try {
        await caches.delete(cacheName);
        await renderStorage(signal);
      } catch (error) {
        if (!signal.aborted) button.textContent = `Clear failed: ${error instanceof Error ? error.message : String(error)}`;
      }
    });
  }
}

export function mountHomeBrowser(): () => void {
  const controller = new AbortController();
  const reportFailure = (error: unknown) => {
    if (controller.signal.aborted) return;
    console.error('[home] capability/storage initialization failed', error);
    const target = document.getElementById('capNote') ?? document.getElementById('storage');
    if (target) target.textContent = `Browser diagnostics unavailable: ${error instanceof Error ? error.message : String(error)}`;
  };
  void detectCapabilities(controller.signal).catch(reportFailure);
  void renderStorage(controller.signal).catch(reportFailure);
  return () => controller.abort();
}
