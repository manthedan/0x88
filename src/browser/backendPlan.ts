export type RuntimeStrategy = 'webgpu_only' | 'wasm_fallback' | 'progressive_fallback';

export interface BrowserProfile {
  id: string;
  hasWebGpu: boolean;
  hasWasmSimd: boolean;
  hasIndexedDb: boolean;
  deviceClass: 'mobile' | 'desktop';
}

export interface BackendPlan {
  backend: 'webgpu' | 'wasm' | 'none';
  cacheModel: boolean;
  progressiveLoad: boolean;
}

export function chooseBackend(profile: BrowserProfile, strategy: RuntimeStrategy): BackendPlan {
  if (strategy === 'webgpu_only') {
    return profile.hasWebGpu
      ? { backend: 'webgpu', cacheModel: false, progressiveLoad: false }
      : { backend: 'none', cacheModel: false, progressiveLoad: false };
  }

  if (strategy === 'wasm_fallback') {
    if (profile.hasWebGpu) return { backend: 'webgpu', cacheModel: false, progressiveLoad: false };
    if (profile.hasWasmSimd) return { backend: 'wasm', cacheModel: false, progressiveLoad: false };
    return { backend: 'none', cacheModel: false, progressiveLoad: false };
  }

  if (strategy === 'progressive_fallback') {
    if (profile.hasWebGpu) return { backend: 'webgpu', cacheModel: profile.hasIndexedDb, progressiveLoad: true };
    if (profile.hasWasmSimd) return { backend: 'wasm', cacheModel: profile.hasIndexedDb, progressiveLoad: profile.deviceClass === 'desktop' };
    return { backend: 'none', cacheModel: false, progressiveLoad: false };
  }

  const exhaustive: never = strategy;
  return exhaustive;
}
