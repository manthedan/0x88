// Landing page: browser capability badges.

function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing element #${id}`);
  return found;
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

export function mountHomeBrowser(): () => void {
  void detectCapabilities();
  return () => undefined;
}
