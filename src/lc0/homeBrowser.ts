// Landing page: browser capability badges and the downloads/storage manager.

const KNOWN_CACHES = [
  { name: 'lc0-browser-models-v1', label: 'Leela networks', detail: 'sha256-validated model cache' },
  { name: 'maia3-browser-models-v1', label: 'Maia3 human model', detail: 'sha256-validated Maia3 model cache' },
  { name: 'lc0-app-shell-v1', label: 'App & engine files', detail: 'offline cache: pages, WASM engines, runtimes' },
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

async function detectCapabilities(): Promise<void> {
  const caps = el('caps');
  const note = el('capNote');
  const out: string[] = [];
  let webgpu = false;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu;
  try { webgpu = !!(await gpu?.requestAdapter?.()); } catch { webgpu = false; }
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
  bytes: number;
}

async function measureCache(name: string, label: string, detail: string): Promise<CacheUsage> {
  if (!(await caches.has(name))) return { name, label, detail, present: false, entries: 0, bytes: 0 };
  const cache = await caches.open(name);
  const keys = await cache.keys();
  let bytes = 0;
  for (const request of keys) {
    const response = await cache.match(request);
    if (!response) continue;
    try {
      bytes += (await response.clone().blob()).size;
    } catch {
      // An unreadable entry still counts as present; size stays unknown.
    }
  }
  return { name, label, detail, present: true, entries: keys.length, bytes };
}

async function renderStorage(): Promise<void> {
  const root = el('storage');
  if (typeof caches === 'undefined') {
    root.innerHTML = '<p class="capnote">Storage management needs a secure context (https or localhost).</p>';
    return;
  }
  root.innerHTML = '<p class="capnote">Measuring…</p>';
  const usages = await Promise.all(KNOWN_CACHES.map(({ name, label, detail }) => measureCache(name, label, detail)));
  const estimate = await navigator.storage?.estimate?.().catch(() => undefined);
  const rows = usages.map((usage) => `
    <div class="store-row" data-cache="${usage.name}">
      <div class="store-info"><b>${usage.label}</b><span>${usage.detail}</span></div>
      <span class="store-size">${usage.present ? `${mb(usage.bytes)} · ${usage.entries} file${usage.entries === 1 ? '' : 's'}` : 'empty'}</span>
      <button type="button" data-clear="${usage.name}" ${usage.present ? '' : 'disabled'}>Clear</button>
    </div>`).join('');
  const totalLine = estimate?.usage !== undefined
    ? `<p class="capnote">This site uses ${mb(estimate.usage)}${estimate.quota ? ` of the ${mb(estimate.quota)} the browser allows` : ''}. Everything re-downloads automatically when needed.</p>`
    : '<p class="capnote">Cleared files re-download automatically when needed.</p>';
  root.innerHTML = rows + totalLine;
  for (const button of root.querySelectorAll<HTMLButtonElement>('button[data-clear]')) {
    button.addEventListener('click', async () => {
      button.disabled = true;
      button.textContent = 'Clearing…';
      await caches.delete(button.dataset.clear ?? '');
      await renderStorage();
    });
  }
}

void detectCapabilities();
void renderStorage();
