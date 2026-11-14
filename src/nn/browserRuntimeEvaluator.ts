import * as v from 'valibot';
import { resolvePublicAssetUrl } from '../lc0/assetUrls.ts';
import type { Evaluator } from './evaluator.ts';
import { type BrowserRuntimeAuditDetail, publishBrowserRuntimeAudit } from './runtimeAudit.ts';
import { RuntimeFallbackEvaluator } from './runtimeFallbackEvaluator.ts';
import {
  type BrowserRuntimeSelector,
  BT4_ANNEAL_MUON_BEST_MODEL_ID,
  BT4_SOAP_REM_C19000_FINAL_MODEL_ID,
  centipawnModelIdForUrls,
  normalizeRuntimeModelKey,
  parseBrowserRuntimeSelector,
  promotedCustomRuntimeForModel,
  promotedTvmjsRuntimeForModel,
  runtimeFallbackEnabled,
  S4FIX_V_EMA_MODEL_ID,
  shouldAttemptCustomRuntime,
  shouldAttemptTvmjsRuntime,
  squareformerModelSettings,
} from './runtimeRegistry.ts';
import { SquareFormerMetaSchema } from './squareFormerMetaSchema.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './squareformerEvaluator.ts';
import { SquareformerTvmHybridEvaluator } from './squareformerTvmHybridEvaluator.ts';
import { SquareformerTvmjsWebgpuEvaluator } from './squareformerTvmjsWebgpuEvaluator.ts';

type NavigatorGpu = Navigator & { gpu?: { requestAdapter(opts?: unknown): Promise<{ requestDevice(): Promise<{ destroy(): void }> } | null> } };

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
  resolvedRuntime: 'ort' | 'tvmjs-webgpu' | 'tvmjs-webgpu-fallback-ort' | 'custom-webgpu' | 'custom-webgpu-fallback-ort';
  runtimeConfigId?: string;
  manifestUrl?: string;
  fallbackReason?: string;
};

async function loadJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return await res.json();
}

function inferRuntimeModelId(spec: BrowserSquareformerRuntimeSpec): string {
  if (centipawnModelIdForUrls(spec.onnx, spec.meta) === S4FIX_V_EMA_MODEL_ID) return S4FIX_V_EMA_MODEL_ID;
  if (spec.modelId) return normalizeRuntimeModelKey(spec.modelId);
  if (spec.id) {
    const normalizedId = normalizeRuntimeModelKey(spec.id);
    if (normalizedId === BT4_ANNEAL_MUON_BEST_MODEL_ID || normalizedId === BT4_SOAP_REM_C19000_FINAL_MODEL_ID) return normalizedId;
  }
  return centipawnModelIdForUrls(spec.onnx, spec.meta) ?? spec.id ?? spec.onnx;
}

async function createHybridEvaluator(
  meta: SquareFormerMeta,
  manifestUrl: string | undefined,
  kernelBase: string | undefined,
  fixtureRoot: string | undefined,
): Promise<Evaluator> {
  const gpu = (navigator as NavigatorGpu).gpu;
  if (!gpu) throw new Error('WebGPU is unavailable for custom WebGPU runtime');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable for custom WebGPU runtime');
  const device = await adapter.requestDevice();
  try {
    // On success the evaluator owns the device and destroys it in destroy().
    return await SquareformerTvmHybridEvaluator.create(device as never, meta, { manifestUrl, kernelBase, fixtureRoot });
  } catch (error) {
    device.destroy();
    throw error;
  }
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
  const meta = v.parse(SquareFormerMetaSchema, await loadJson(spec.meta));
  const modelId = inferRuntimeModelId(spec);
  const requestedRuntime = parseBrowserRuntimeSelector(options.runtime ?? spec.runtime ?? params?.get('runtime'));
  const createOrtEvaluator = () => SquareFormerEvaluator.create(spec.onnx, meta, squareformerModelSettings(modelId));
  const fallback = options.fallback ?? (params ? runtimeFallbackEnabled(params, true) : true);
  const tvmjsEntry = promotedTvmjsRuntimeForModel(modelId);
  const customEntry = promotedCustomRuntimeForModel(modelId);
  const entry = shouldAttemptTvmjsRuntime(modelId, requestedRuntime) ? tvmjsEntry : customEntry;
  const manifestPath = options.manifestUrl ?? spec.manifestUrl ?? params?.get('manifest') ?? params?.get('manifestUrl') ?? entry?.artifact.manifestUrl;
  const manifestUrl = manifestPath ? resolvePublicAssetUrl(manifestPath) : undefined;
  const kernelBase = options.kernelBase ?? spec.kernelBase ?? params?.get('kernelBase') ?? undefined;
  const fixtureRoot = options.fixtureRoot ?? spec.fixtureRoot ?? params?.get('fixtureRoot') ?? undefined;
  const customRequested = shouldAttemptCustomRuntime(modelId, requestedRuntime);
  const auditBase = {
    source: 'createBrowserSquareformerRuntimeEvaluator',
    family: 'centipawn',
    engineLabel: spec.label,
    modelId,
    modelUrl: spec.onnx,
    metaUrl: spec.meta,
    requestedRuntime,
    runtimeConfigId: entry?.runtimeConfigId,
    manifestUrl,
    ...options.audit,
  };

  // The compiled graphs belong to the old model, even if a persisted strict selector supplies a manifest.
  if (modelId === S4FIX_V_EMA_MODEL_ID) {
    const compiledRequested = requestedRuntime === 'tvmjs-webgpu' || requestedRuntime === 'custom-webgpu';
    const fallbackReason = compiledRequested ? 's4fix-v-ema is ORT-only; compiled runtimes target an older graph' : undefined;
    const resolvedRuntime = compiledRequested ? (`${requestedRuntime}-fallback-ort` as const) : 'ort';
    if (fallbackReason) console.warn('Centipawn compiled runtime unavailable; falling back to ORT SquareFormer.', { modelId, fallbackReason });
    const evaluator = await createOrtEvaluator();
    publishBrowserRuntimeAudit({ ...auditBase, resolvedRuntime, fallbackReason });
    return { evaluator, meta, modelId, requestedRuntime, resolvedRuntime, fallbackReason };
  }

  if (requestedRuntime === 'custom-webgpu' && !entry && !manifestUrl) {
    throw new Error(`No promoted custom WebGPU runtime is registered for ${modelId}`);
  }

  if (requestedRuntime === 'tvmjs-webgpu' && !entry && !manifestUrl) {
    throw new Error(`No promoted TVMJS WebGPU runtime is registered for ${modelId}`);
  }

  if (shouldAttemptTvmjsRuntime(modelId, requestedRuntime)) {
    try {
      if (!manifestUrl) throw new Error(`No TVMJS manifest is registered for ${modelId}`);
      const primary = await SquareformerTvmjsWebgpuEvaluator.create(manifestUrl, meta);
      const selectedManifestUrl = primary.manifestUrl ?? manifestUrl;
      const runtimeConfigId = new URL(selectedManifestUrl, location.href).pathname.endsWith('/f32/v3-subgroups/manifest.json')
        ? 'centipawn-tvmjs-webgpu-f32-v3-subgroups'
        : entry?.runtimeConfigId;
      const selectedAudit = { ...auditBase, manifestUrl: selectedManifestUrl, runtimeConfigId };
      const evaluator = fallback ? new RuntimeFallbackEvaluator(primary, () => SquareFormerEvaluator.create(spec.onnx, meta), selectedAudit) : primary;
      const result = {
        evaluator,
        meta,
        modelId,
        requestedRuntime,
        resolvedRuntime: 'tvmjs-webgpu' as const,
        runtimeConfigId,
        manifestUrl: selectedManifestUrl,
      };
      publishBrowserRuntimeAudit({ ...selectedAudit, resolvedRuntime: result.resolvedRuntime });
      return result;
    } catch (err) {
      if (!fallback) throw err;
      const fallbackReason = err instanceof Error ? err.message : String(err);
      console.warn('TVMJS WebGPU runtime unavailable; falling back to ORT SquareFormer.', { label: spec.label ?? spec.id, modelId, fallbackReason });
      const evaluator = await SquareFormerEvaluator.create(spec.onnx, meta);
      const result = {
        evaluator,
        meta,
        modelId,
        requestedRuntime,
        resolvedRuntime: 'tvmjs-webgpu-fallback-ort' as const,
        runtimeConfigId: entry?.runtimeConfigId,
        manifestUrl,
        fallbackReason,
      };
      publishBrowserRuntimeAudit({ ...auditBase, resolvedRuntime: result.resolvedRuntime, fallbackReason });
      return result;
    }
  }

  if (customRequested) {
    try {
      const primary = await createHybridEvaluator(meta, manifestUrl, kernelBase, fixtureRoot);
      const evaluator = fallback
        ? new RuntimeFallbackEvaluator(primary, () => SquareFormerEvaluator.create(spec.onnx, meta), auditBase, 'custom-webgpu-fallback-ort')
        : primary;
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
