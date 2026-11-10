// Browser runtime feature detection shared by the workspace capability
// disclosure (BrowserCapabilities.svelte) and the landing page device check.
import { supportsWasmRelaxedSimd, supportsWasmSimd } from '../lc0/wasmFeatures.ts';

export interface BrowserCapabilityReport {
  /** null when the adapter request threw or did not answer in time. */
  webgpu: boolean | null;
  wasm: boolean;
  wasmSimd: boolean;
  wasmRelaxedSimd: boolean;
  /** crossOriginIsolated + SharedArrayBuffer: persistent and threaded WASM runtimes. */
  threadedWasm: boolean;
  isolated: boolean;
  sharedMemory: boolean;
  threads: number;
  cacheStorage: boolean;
}

const ADAPTER_TIMEOUT_MS = 4000;

async function probeWebGpu(): Promise<boolean | null> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await Promise.race([gpu.requestAdapter(), new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), ADAPTER_TIMEOUT_MS))]);
    return adapter === 'timeout' ? null : !!adapter;
  } catch {
    return false;
  }
}

let pending: Promise<BrowserCapabilityReport> | null = null;

/** Detect once per page load; every caller shares the same answer. */
export function detectBrowserCapabilities(): Promise<BrowserCapabilityReport> {
  pending ??= (async () => {
    const webgpu = await probeWebGpu();
    const wasm = typeof WebAssembly !== 'undefined';
    const isolated = globalThis.crossOriginIsolated === true;
    const sharedMemory = typeof SharedArrayBuffer !== 'undefined';
    return {
      webgpu,
      wasm,
      wasmSimd: wasm && supportsWasmSimd(),
      wasmRelaxedSimd: wasm && supportsWasmRelaxedSimd(),
      threadedWasm: isolated && sharedMemory,
      isolated,
      sharedMemory,
      threads: Math.max(1, navigator.hardwareConcurrency || 1),
      cacheStorage: 'caches' in globalThis,
    };
  })();
  return pending;
}
