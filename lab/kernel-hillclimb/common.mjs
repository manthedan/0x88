// Shared helpers for kernel hill-climb bench pages: device identity, sample
// statistics, and the result contract (window.__kernelBenchResult /
// window.__kernelBenchError). See scripts/kernel_hillclimb/README.md.
import { mad, median } from '../../scripts/lib/perf_canary_stats.mjs';

export const RESULT_SCHEMA = 'kernel-hillclimb.bench-result.v1';

export const params = new URLSearchParams(location.search);

export function intParam(name, fallback, min = 1, max = 1e9) {
  const raw = params.get(name);
  if (raw === null || raw === '') return fallback;
  const value = Math.floor(Number(raw));
  if (!Number.isFinite(value) || value < min || value > max) throw new Error(`invalid ${name}=${raw} (expected integer ${min}..${max})`);
  return value;
}

export function listParam(name, fallback) {
  const raw = params.get(name);
  if (raw === null || raw === '') return fallback;
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

export function log(...parts) {
  const text = parts.map((p) => (typeof p === 'string' ? p : JSON.stringify(p))).join(' ');
  const el = document.getElementById('log');
  if (el) el.textContent += `${text}\n`;
  console.log('[kernel-bench]', text);
}

/** Quantile by linear interpolation on the sorted sample (q in [0,1]). */
export function quantile(values, q) {
  const s = [...values].sort((a, b) => a - b);
  if (!s.length) return NaN;
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/**
 * Noise-aware summary of a timing sample. Uses the perf canary's median/MAD so
 * candidate comparisons follow the same "max(floor, 3*MAD/median)" rule.
 */
export function summarize(values) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return { n: 0 };
  const m = median(finite);
  const spread = mad(finite);
  return {
    n: finite.length,
    median: m,
    mad: spread,
    madRatio: spread / Math.max(Math.abs(m), 1e-12),
    mean: finite.reduce((a, b) => a + b, 0) / finite.length,
    min: Math.min(...finite),
    p10: quantile(finite, 0.1),
    p90: quantile(finite, 0.9),
    max: Math.max(...finite),
    noisy: finite.length >= 3 && spread / Math.max(Math.abs(m), 1e-12) > 0.1,
  };
}

/** Everything a remote run needs to identify the device and browser. */
export async function deviceIdentity(adapter, device) {
  const info = adapter?.info ?? {};
  let uaData = null;
  try {
    if (navigator.userAgentData?.getHighEntropyValues) {
      uaData = await navigator.userAgentData.getHighEntropyValues(['platform', 'platformVersion', 'architecture', 'model', 'fullVersionList', 'bitness']);
    }
  } catch {
    uaData = null;
  }
  const limits = {};
  const limitSource = device?.limits ?? adapter?.limits;
  if (limitSource) {
    for (const key of [
      'maxBufferSize',
      'maxStorageBufferBindingSize',
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupSizeX',
      'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupsPerDimension',
      'maxStorageBuffersPerShaderStage',
    ])
      limits[key] = limitSource[key];
  }
  const subgroupMatrixConfigs = (adapter?.info?.subgroupMatrixConfigs ?? adapter?.subgroupMatrixConfigs ?? []).map?.((c) => ({
    componentType: c.componentType,
    resultComponentType: c.resultComponentType,
    M: c.M,
    N: c.N,
    K: c.K,
  }));
  return {
    adapter: {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      description: info.description ?? '',
      device: info.device ?? '',
      subgroupMinSize: info.subgroupMinSize,
      subgroupMaxSize: info.subgroupMaxSize,
      isFallbackAdapter: info.isFallbackAdapter,
      subgroupMatrixConfigs: subgroupMatrixConfigs ?? [],
    },
    adapterFeatures: adapter ? [...adapter.features].sort() : [],
    deviceFeatures: device ? [...device.features].sort() : [],
    limits,
    wgslLanguageFeatures: [...(navigator.gpu?.wgslLanguageFeatures ?? [])].sort(),
    userAgent: navigator.userAgent,
    userAgentData: uaData,
    platform: navigator.platform,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: navigator.deviceMemory,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    page: location.pathname,
    query: Object.fromEntries(params.entries()),
  };
}

export async function requestAdapter() {
  if (!navigator.gpu) throw new Error('navigator.gpu missing (WebGPU unavailable)');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  return adapter;
}

/** Publish the final JSON for the runner (local agent-browser or remote ssh). */
export function publishResult(result) {
  const full = { schema: RESULT_SCHEMA, generatedAt: new Date().toISOString(), ...result };
  window.__kernelBenchResult = full;
  log('RESULT_READY', { bench: full.bench, ok: full.ok });
  const status = document.getElementById('status');
  if (status) status.textContent = full.ok ? 'done' : 'done (gate failed)';
  return full;
}

export function publishError(error) {
  const message = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
  window.__kernelBenchError = message;
  log('BENCH_ERROR', message);
  const status = document.getElementById('status');
  if (status) status.textContent = 'error';
}

export function runMain(main) {
  window.__kernelBenchResult = undefined;
  window.__kernelBenchError = undefined;
  main().then(
    (result) => {
      if (result) publishResult(result);
    },
    (error) => publishError(error),
  );
}

/** FNV-1a 32-bit hash for shader identities (cheap, synchronous). */
export function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
