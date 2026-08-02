export type BrowserRuntimeSelector = 'auto' | 'ort' | 'tvmjs-webgpu' | 'custom-webgpu';
export type RuntimeRegistryStatus = 'experimental' | 'beta' | 'promoted' | 'disabled';

export type RuntimeRegistryEntry = {
  modelId: string;
  runtime: Exclude<BrowserRuntimeSelector, 'auto'>;
  runtimeConfigId: string;
  status: RuntimeRegistryStatus;
  artifact: { manifestUrl: string };
  fallback: { runtime: 'ort'; modelId: string; onnxUrl: string; metaUrl: string };
  requiredFeatures: string[];
  promotionGate: string;
};

export const BT4_ANNEAL_MUON_BEST_MODEL_ID = 'bt4-anneal-muon-best';
export const BT4_ANNEAL_MUON_BEST_TVM_HYBRID_LEGACY_MODEL_ID = 'bt4-anneal-muon-best-tvm-hybrid';
export const BT4_SOAP_REM_C19000_FINAL_MODEL_ID = 'bt4-soap-rem-c19000-final';

export const CENTIPAWN_TVMJS_WEBGPU_V2: RuntimeRegistryEntry = {
  modelId: BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
  runtime: 'tvmjs-webgpu',
  runtimeConfigId: 'centipawn-tvmjs-webgpu-f32-v2-shape-k16',
  status: 'promoted',
  artifact: { manifestUrl: '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json' },
  fallback: {
    runtime: 'ort',
    modelId: BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
    onnxUrl: '/models/bt4_soap_rem_c19000_final.onnx',
    metaUrl: '/models/bt4_soap_rem_c19000_final.meta.json',
  },
  requiredFeatures: ['webgpu'],
  promotionGate: 'centipawn-tvmjs-webgpu-v2',
};

export const SQUAREFORMER_TVM_HYBRID_V1: RuntimeRegistryEntry = {
  modelId: BT4_ANNEAL_MUON_BEST_MODEL_ID,
  runtime: 'custom-webgpu',
  runtimeConfigId: 'squareformer-tvm-webgpu-hybrid-v1',
  status: 'promoted',
  artifact: { manifestUrl: '/runtimes/squareformer-tvm-hybrid/bt4-anneal-muon-best/v1/manifest.json' },
  fallback: {
    runtime: 'ort',
    modelId: BT4_ANNEAL_MUON_BEST_MODEL_ID,
    onnxUrl: '/models/bt4_anneal_muon_best.onnx',
    metaUrl: '/models/bt4_anneal_muon_best.meta.json',
  },
  requiredFeatures: ['webgpu'],
  promotionGate: 'squareformer-rc-v1',
};

const PROMOTED_RUNTIMES: RuntimeRegistryEntry[] = [CENTIPAWN_TVMJS_WEBGPU_V2, SQUAREFORMER_TVM_HYBRID_V1].filter((entry) => entry.status === 'promoted');

export function normalizeRuntimeModelKey(modelKey: string): string {
  return modelKey === BT4_ANNEAL_MUON_BEST_TVM_HYBRID_LEGACY_MODEL_ID ? BT4_ANNEAL_MUON_BEST_MODEL_ID : modelKey;
}

export function parseBrowserRuntimeSelector(value: string | null | undefined, options: { legacyHybridModelKey?: string | null } = {}): BrowserRuntimeSelector {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/_/g, '-');
  if (normalized === '' && options.legacyHybridModelKey === BT4_ANNEAL_MUON_BEST_TVM_HYBRID_LEGACY_MODEL_ID) return 'custom-webgpu';
  if (normalized === '' || normalized === 'auto') return 'auto';
  if (normalized === 'ort' || normalized === 'onnx' || normalized === 'onnxruntime' || normalized === 'baseline') return 'ort';
  if (normalized === 'tvmjs' || normalized === 'tvmjs-webgpu' || normalized === 'webgpu-tvmjs' || normalized === 'compiled-webgpu') return 'tvmjs-webgpu';
  if (normalized === 'custom-webgpu' || normalized === 'webgpu-custom' || normalized === 'tvm-hybrid' || normalized === 'hybrid') return 'custom-webgpu';
  return 'auto';
}

function boolRuntimeParam(value: string | null | undefined, fallback: boolean): boolean {
  if (value === null || value === undefined) return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return fallback;
}

export function runtimeFallbackEnabled(params: URLSearchParams, fallback = true): boolean {
  return boolRuntimeParam(params.get('runtimeFallback') ?? params.get('hybridFallback'), fallback);
}

export function promotedCustomRuntimeForModel(modelId: string): RuntimeRegistryEntry | undefined {
  return PROMOTED_RUNTIMES.find((entry) => entry.modelId === modelId && entry.runtime === 'custom-webgpu');
}

export function promotedTvmjsRuntimeForModel(modelId: string): RuntimeRegistryEntry | undefined {
  return PROMOTED_RUNTIMES.find((entry) => entry.modelId === modelId && entry.runtime === 'tvmjs-webgpu');
}

export function shouldAttemptCustomRuntime(modelId: string, selector: BrowserRuntimeSelector): boolean {
  if (selector === 'ort') return false;
  if (selector === 'custom-webgpu') return true;
  return promotedCustomRuntimeForModel(modelId)?.status === 'promoted';
}

export function shouldAttemptTvmjsRuntime(modelId: string, selector: BrowserRuntimeSelector): boolean {
  if (selector === 'ort' || selector === 'custom-webgpu') return false;
  if (selector === 'tvmjs-webgpu') return true;
  return promotedTvmjsRuntimeForModel(modelId)?.status === 'promoted';
}
