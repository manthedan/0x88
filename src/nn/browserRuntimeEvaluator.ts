import { SquareFormerEvaluator, type SquareFormerMeta } from './squareformerEvaluator.ts';
import { SquareformerTvmHybridEvaluator } from './squareformerTvmHybridEvaluator.ts';
import type { Evaluator } from './evaluator.ts';
import { publishBrowserRuntimeAudit, type BrowserRuntimeAuditDetail } from './runtimeAudit.ts';
import {
  BT4_ANNEAL_MUON_BEST_MODEL_ID,
  normalizeRuntimeModelKey,
  parseBrowserRuntimeSelector,
  promotedCustomRuntimeForModel,
  runtimeFallbackEnabled,
  shouldAttemptCustomRuntime,
  type BrowserRuntimeSelector,
} from './runtimeRegistry.ts';

type NavigatorGpu = Navigator & { gpu?: { requestAdapter(opts?: unknown): Promise<{ requestDevice(): Promise<unknown> } | null> } };

export type BrowserSquareformerRuntimeSpec = {
  id?: string;
  label?: string;
  modelId?: string;
  onnx: string;
  meta: string;
  runtime?: BrowserRuntimeSelector | string;
  manifestUrl?: string;
  kernelBase?: string;
  fixtureRoot?: string;
};

export type BrowserRuntimeEvaluatorResult = {
  evaluator: Evaluator;
  meta: SquareFormerMeta;
  modelId: string;
  requestedRuntime: BrowserRuntimeSelector;
  resolvedRuntime: 'ort' | 'custom-webgpu' | 'custom-webgpu-fallback-ort';
  runtimeConfigId?: string;
  manifestUrl?: string;
  fallbackReason?: string;
};

async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}

function inferRuntimeModelId(spec: BrowserSquareformerRuntimeSpec): string {
  if (spec.modelId) return normalizeRuntimeModelKey(spec.modelId);
  if (spec.id) {
    const normalizedId = normalizeRuntimeModelKey(spec.id);
    if (normalizedId === BT4_ANNEAL_MUON_BEST_MODEL_ID) return normalizedId;
  }
  const urls = `${spec.onnx} ${spec.meta}`;
  if (urls.includes('bt4_anneal_muon_best') || urls.includes('bt4-anneal-muon-best')) return BT4_ANNEAL_MUON_BEST_MODEL_ID;
  return spec.id ?? spec.onnx;
}

async function createHybridEvaluator(meta: SquareFormerMeta, manifestUrl: string | undefined, kernelBase: string | undefined, fixtureRoot: string | undefined): Promise<Evaluator> {
  const gpu = (navigator as NavigatorGpu).gpu;
  if (!gpu) throw new Error('WebGPU is unavailable for custom WebGPU runtime');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable for custom WebGPU runtime');
  const device = await adapter.requestDevice();
  return await SquareformerTvmHybridEvaluator.create(device as never, meta, { manifestUrl, kernelBase, fixtureRoot });
}

export async function createBrowserSquareformerRuntimeEvaluator(
  spec: BrowserSquareformerRuntimeSpec,
  options: {
    params?: URLSearchParams;
    runtime?: string | null;
    manifestUrl?: string | null;
    kernelBase?: string | null;
    fixtureRoot?: string | null;
    fallback?: boolean;
    audit?: Partial<BrowserRuntimeAuditDetail>;
  } = {},
): Promise<BrowserRuntimeEvaluatorResult> {
  const params = options.params;
  const meta = await loadJson<SquareFormerMeta>(spec.meta);
  const modelId = inferRuntimeModelId(spec);
  const requestedRuntime = parseBrowserRuntimeSelector(options.runtime ?? spec.runtime ?? params?.get('runtime'));
  const fallback = options.fallback ?? (params ? runtimeFallbackEnabled(params, true) : true);
  const entry = promotedCustomRuntimeForModel(modelId);
  const manifestUrl = options.manifestUrl ?? spec.manifestUrl ?? params?.get('manifest') ?? params?.get('manifestUrl') ?? entry?.artifact.manifestUrl;
  const kernelBase = options.kernelBase ?? spec.kernelBase ?? params?.get('kernelBase') ?? undefined;
  const fixtureRoot = options.fixtureRoot ?? spec.fixtureRoot ?? params?.get('fixtureRoot') ?? undefined;
  const customRequested = shouldAttemptCustomRuntime(modelId, requestedRuntime);
  const auditBase = {
    source: 'createBrowserSquareformerRuntimeEvaluator',
    family: 'tiny',
    engineLabel: spec.label,
    modelId,
    modelUrl: spec.onnx,
    metaUrl: spec.meta,
    requestedRuntime,
    runtimeConfigId: entry?.runtimeConfigId,
    manifestUrl,
    ...options.audit,
  };

  if (requestedRuntime === 'custom-webgpu' && !entry && !manifestUrl) {
    throw new Error(`No promoted custom WebGPU runtime is registered for ${modelId}`);
  }

  if (customRequested) {
    try {
      const evaluator = await createHybridEvaluator(meta, manifestUrl, kernelBase, fixtureRoot);
      const result = {
        evaluator,
        meta,
        modelId,
        requestedRuntime,
        resolvedRuntime: 'custom-webgpu' as const,
        runtimeConfigId: entry?.runtimeConfigId,
        manifestUrl,
      };
      publishBrowserRuntimeAudit({ ...auditBase, resolvedRuntime: result.resolvedRuntime });
      return result;
    } catch (err) {
      if (!fallback) throw err;
      const fallbackReason = err instanceof Error ? err.message : String(err);
      console.warn('Custom WebGPU runtime unavailable; falling back to ORT SquareFormer.', { label: spec.label ?? spec.id, modelId, fallbackReason });
      const evaluator = await SquareFormerEvaluator.create(spec.onnx, meta);
      const result = {
        evaluator,
        meta,
        modelId,
        requestedRuntime,
        resolvedRuntime: 'custom-webgpu-fallback-ort' as const,
        runtimeConfigId: entry?.runtimeConfigId,
        manifestUrl,
        fallbackReason,
      };
      publishBrowserRuntimeAudit({ ...auditBase, resolvedRuntime: result.resolvedRuntime, fallbackReason });
      return result;
    }
  }

  const evaluator = await SquareFormerEvaluator.create(spec.onnx, meta);
  const result = {
    evaluator,
    meta,
    modelId,
    requestedRuntime,
    resolvedRuntime: 'ort' as const,
  };
  publishBrowserRuntimeAudit({ ...auditBase, resolvedRuntime: result.resolvedRuntime });
  return result;
}
