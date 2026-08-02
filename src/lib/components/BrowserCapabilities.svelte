<script lang="ts">
import { onMount } from 'svelte';

let ready = false;
let webgpu = false;
let isolated = false;
let sharedMemory = false;
let threads = 1;
let cacheStorage = false;

onMount(() => {
  void (async () => {
    try {
      const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
      webgpu = !!gpu && !!(await gpu.requestAdapter());
    } catch {
      webgpu = false;
    }
    isolated = globalThis.crossOriginIsolated === true;
    sharedMemory = typeof SharedArrayBuffer !== 'undefined';
    threads = Math.max(1, navigator.hardwareConcurrency || 1);
    cacheStorage = 'caches' in globalThis;
    ready = true;
  })();
});
</script>

<details class="capabilities" data-testid="browser-capabilities">
  <summary>
    <span class="cap-label">Browser capabilities</span>
    {#if ready}
      <span class="desktop-status" class:ok={webgpu} class:warn={!webgpu}>WebGPU {webgpu ? 'ready' : 'unavailable'}</span>
      <span class="desktop-status" class:ok={isolated && sharedMemory} class:warn={!isolated || !sharedMemory}>WASM threads {isolated && sharedMemory ? 'ready' : 'single-thread fallback'}</span>
      <span class="mobile-status" class:ok={webgpu && isolated && sharedMemory} class:warn={!webgpu || !isolated || !sharedMemory}>
        {webgpu && isolated && sharedMemory ? 'WebGPU + threads ready' : webgpu ? 'WebGPU ready · CPU fallback' : 'CPU fallback active'}
      </span>
    {/if}
  </summary>
  {#if ready}
    <div class="capability-grid">
      <div><strong>WebGPU</strong><span>{webgpu ? 'Available. Neural engines can use the GPU.' : 'Unavailable. Supported engines fall back to WASM/CPU.'}</span></div>
      <div><strong>Shared memory</strong><span>{isolated && sharedMemory ? 'Available. Persistent and threaded WASM runtimes are enabled.' : 'Unavailable. CPU engines use safe single-thread or one-shot modes.'}</span></div>
      <div><strong>CPU capacity</strong><span>{threads} logical thread{threads === 1 ? '' : 's'} reported by this browser.</span></div>
      <div><strong>Model cache</strong><span>{cacheStorage ? 'Available. Downloaded models can be reused.' : 'Unavailable. Models may download again next visit.'}</span></div>
    </div>
  {:else}
    <div class="checking">Detecting browser runtime features…</div>
  {/if}
</details>

<style>
  .capabilities{max-width:1280px; margin:12px auto 0; padding:8px 12px; border:1px solid var(--rule); border-radius:10px; background:var(--panel)}
  summary{cursor:pointer; display:flex; align-items:center; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:12px; font-weight:650}
  summary .cap-label{padding:0; border:0; border-radius:0; font-family:var(--sans); font-size:12px}
  summary .desktop-status,summary .mobile-status{padding:2px 7px; border-radius:999px; border:1px solid var(--rule); font-family:var(--mono); font-size:10px}
  summary .ok{color:var(--accent); border-color:color-mix(in srgb, var(--accent) 45%, var(--rule))}
  summary .warn{color:var(--warn); border-color:color-mix(in srgb, var(--warn) 45%, var(--rule))}
  .mobile-status{display:none}
  .capability-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 18px; margin-top:10px; padding-top:9px; border-top:1px solid var(--rule); font-size:12px}
  .capability-grid div{display:grid; gap:2px}
  .capability-grid span,.checking{color:var(--muted)}
  @media(max-width:700px){
    .capability-grid{grid-template-columns:1fr}
    .capabilities{margin:10px 12px 0; padding:7px 10px}
    summary{min-height:28px; flex-wrap:nowrap}
    summary .cap-label{font-size:11px; margin-right:auto}
    .desktop-status{display:none}
    .mobile-status{display:inline-block; white-space:nowrap}
  }
</style>
