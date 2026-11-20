import * as v from 'valibot';
import { resolvePublicAssetUrl } from '../lc0/assetUrls.ts';
import { isPinnedCentipawnOnnx } from './centipawnModel.ts';
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

export async function createBrowserSquareformerRuntimeEvaluator(
  spec: BrowserSquareformerRuntimeSpec,
  options: {
    params?: URLSearchParams;
    runtime?: string | null;
    manifestUrl?: string | null;
    kernelBase?: string | null;
    fixtureRoot?: string | null;
    fallback?: boolean;
    signal?: AbortSignal;
    audit?: Partial<BrowserRuntimeAuditDetail>;
  } = {},
): Promise<BrowserRuntimeEvaluatorResult> {
  const { params, signal } = options;
  const res = await fetch(spec.meta, { signal });
  if (!res.ok) throw new Error(`${spec.meta}: ${res.status} ${res.statusText}`);
  const meta = v.parse(SquareFormerMetaSchema, await res.json());
  signal?.throwIfAborted();

  const urlModelId = centipawnModelIdForUrls(spec.onnx, spec.meta);
  const normalizedSpecId = spec.id ? normalizeRuntimeModelKey(spec.id) : undefined;
  const modelId =
    isPinnedCentipawnOnnx(spec.onnx) || urlModelId === S4FIX_V_EMA_MODEL_ID
      ? S4FIX_V_EMA_MODEL_ID
      : spec.modelId
        ? normalizeRuntimeModelKey(spec.modelId)
        : normalizedSpecId === BT4_ANNEAL_MUON_BEST_MODEL_ID || normalizedSpecId === BT4_SOAP_REM_C19000_FINAL_MODEL_ID
          ? normalizedSpecId
          : (urlModelId ?? spec.id ?? spec.onnx);

  const requestedRuntime = parseBrowserRuntimeSelector(options.runtime ?? spec.runtime ?? params?.get('runtime'));
  const createOrtEvaluator = () =>
    SquareFormerEvaluator.create(spec.onnx, meta, squareformerModelSettings(modelId), {
      verifiedCentipawn: isPinnedCentipawnOnnx(spec.onnx),
      signal,
    });
  const fallback = options.fallback ?? (params ? runtimeFallbackEnabled(params, true) : true);
  const attemptTvmjs = shouldAttemptTvmjsRuntime(modelId, requestedRuntime);
  const attemptCustom = !attemptTvmjs && shouldAttemptCustomRuntime(modelId, requestedRuntime);
  const entry = attemptTvmjs ? promotedTvmjsRuntimeForModel(modelId) : promotedCustomRuntimeForModel(modelId);
  const manifestPath = options.manifestUrl ?? spec.manifestUrl ?? params?.get('manifest') ?? params?.get('manifestUrl') ?? entry?.artifact.manifestUrl;
  const manifestUrl = manifestPath ? resolvePublicAssetUrl(manifestPath) : undefined;
  const kernelBase = options.kernelBase ?? spec.kernelBase ?? params?.get('kernelBase') ?? undefined;
  const fixtureRoot = options.fixtureRoot ?? spec.fixtureRoot ?? params?.get('fixtureRoot') ?? undefined;
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
  const finish = (
    evaluator: Evaluator,
    resolvedRuntime: BrowserRuntimeEvaluatorResult['resolvedRuntime'],
    extra: Partial<Pick<BrowserRuntimeEvaluatorResult, 'runtimeConfigId' | 'manifestUrl' | 'fallbackReason'>> = {},
    auditOverride: Partial<BrowserRuntimeAuditDetail> = {},
  ): BrowserRuntimeEvaluatorResult => {
    publishBrowserRuntimeAudit({ ...auditBase, ...auditOverride, resolvedRuntime, fallbackReason: extra.fallbackReason });
    return { evaluator, meta, modelId, requestedRuntime, resolvedRuntime, ...extra };
  };

  // The compiled graphs belong to the old model, even if a persisted strict selector supplies a manifest.
  if (modelId === S4FIX_V_EMA_MODEL_ID) {
    const compiledRequested = requestedRuntime === 'tvmjs-webgpu' || requestedRuntime === 'custom-webgpu';
    const fallbackReason = compiledRequested ? 's4fix-v-ema is ORT-only; compiled runtimes target an older graph' : undefined;
    const resolvedRuntime = compiledRequested ? (`${requestedRuntime}-fallback-ort` as const) : 'ort';
    if (fallbackReason) console.warn('Centipawn compiled runtime unavailable; falling back to ORT SquareFormer.', { modelId, fallbackReason });
    return finish(await createOrtEvaluator(), resolvedRuntime, { fallbackReason });
  }

  if ((requestedRuntime === 'custom-webgpu' || requestedRuntime === 'tvmjs-webgpu') && !entry && !manifestUrl) {
    throw new Error(`No promoted ${requestedRuntime === 'custom-webgpu' ? 'custom' : 'TVMJS'} WebGPU runtime is registered for ${modelId}`);
  }

  if (attemptTvmjs || attemptCustom) {
    const resolvedRuntime = attemptTvmjs ? ('tvmjs-webgpu' as const) : ('custom-webgpu' as const);
    const fallbackRuntime = attemptTvmjs ? ('tvmjs-webgpu-fallback-ort' as const) : ('custom-webgpu-fallback-ort' as const);
    try {
      let primary: Evaluator;
      let selectedManifestUrl = manifestUrl;
      let runtimeConfigId = entry?.runtimeConfigId;
      if (attemptTvmjs) {
        if (!manifestUrl) throw new Error(`No TVMJS manifest is registered for ${modelId}`);
        const tvmjs = await SquareformerTvmjsWebgpuEvaluator.create(manifestUrl, meta, signal);
        primary = tvmjs;
        selectedManifestUrl = tvmjs.manifestUrl ?? manifestUrl;
        if (new URL(selectedManifestUrl, location.href).pathname.endsWith('/f32/v3-subgroups/manifest.json')) {
          runtimeConfigId = 'centipawn-tvmjs-webgpu-f32-v3-subgroups';
        }
      } else {
        const gpu = navigator.gpu;
        if (!gpu) throw new Error('WebGPU is unavailable for custom WebGPU runtime');
        const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('WebGPU adapter unavailable for custom WebGPU runtime');
        const device = await adapter.requestDevice();
        try {
          signal?.throwIfAborted();
          // On success the evaluator owns the device and destroys it in destroy().
          primary = await SquareformerTvmHybridEvaluator.create(device, meta, { manifestUrl, kernelBase, fixtureRoot }, signal);
        } catch (error) {
          device.destroy();
          throw error;
        }
      }
      const selectedAudit = { ...auditBase, manifestUrl: selectedManifestUrl, runtimeConfigId };
      const evaluator = fallback ? new RuntimeFallbackEvaluator(primary, createOrtEvaluator, selectedAudit, fallbackRuntime) : primary;
      return finish(evaluator, resolvedRuntime, { runtimeConfigId, manifestUrl: selectedManifestUrl }, selectedAudit);
    } catch (err) {
      signal?.throwIfAborted();
      if ((err instanceof Error && err.name === 'AbortError') || !fallback) throw err;
      const fallbackReason = err instanceof Error ? err.message : String(err);
      console.warn(`${attemptTvmjs ? 'TVMJS' : 'Custom'} WebGPU runtime unavailable; falling back to ORT SquareFormer.`, {
        label: spec.label ?? spec.id,
        modelId,
        fallbackReason,
      });
      return finish(await createOrtEvaluator(), fallbackRuntime, { runtimeConfigId: entry?.runtimeConfigId, manifestUrl, fallbackReason });
    }
  }

  return finish(await createOrtEvaluator(), 'ort');
}
