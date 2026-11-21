import { type OrtSessionPlacement, resolveOrtSessionPlacement } from '../nn/ortRuntime.ts';
import {
  LC0_BIG_NET_SPLITK_FILES,
  LC0_SMALL_BATCH1_QDQ8_FILE,
  LC0_SMALL_BATCH1_QDQ8_PATH,
  LC0_SMALL_BATCH1_SPLITK_FILE,
  type Lc0ModelVariant,
  lc0SmallModelRequest,
  lc0SmallSplitKUrl,
  lc0SplitKAdapterEligible,
  lc0SplitKUrl,
  selectLc0OrtModelUrl,
} from './lc0ModelUrls.ts';
import {
  clearLc0ModelCacheEntry,
  type Lc0ModelLoadInfo,
  type Lc0ModelLoadOptions,
  type Lc0ModelLoadResult,
  loadLc0ModelForOrt,
  untilAborted,
} from './modelCache.ts';
import { defaultLc0OrtIoMode, Lc0OnnxEvaluator, type Lc0OnnxEvaluatorOptions } from './onnxEvaluator.ts';
import { isMetadataTrustedModel, isOnnxModelParseError } from './onnxModelRecovery.ts';

export {
  LC0_BIG_NET_SPLITK_FILES,
  LC0_SMALL_BATCH1_QDQ8_FILE,
  LC0_SMALL_BATCH1_QDQ8_PATH,
  LC0_SMALL_BATCH1_SPLITK_FILE,
  type Lc0ModelVariant,
  lc0SmallModelRequest,
  lc0SmallSplitKUrl,
  lc0SplitKAdapterEligible,
  lc0SplitKUrl,
  selectLc0OrtModelUrl,
};

type ModelBytes = string | Uint8Array | ArrayBuffer;
type LoadedModel = { info: Lc0ModelLoadInfo; input: ModelBytes | undefined };

function loadedModel(result: Lc0ModelLoadResult, revalidated = false): LoadedModel {
  const { model, ...info } = result;
  if (revalidated) info.revalidated = true;
  return { info, input: model };
}

function takeModel(load: LoadedModel): ModelBytes | undefined {
  const input = load.input;
  load.input = undefined;
  return input;
}

export type Lc0OrtEvaluatorFromUrlOptions = {
  load?: Lc0ModelLoadOptions;
  evaluator?: Omit<Lc0OnnxEvaluatorOptions, 'signal' | 'wasmFallbackModel' | 'reloadModel' | 'rejectModelParseErrors'>;
  /**
   * Apply the device rule above. Product surfaces set it from
   * lc0SmallModelRequest (no ?model= override); explicit overrides and lab
   * tools leave it off so they get exactly the bytes they named.
   */
  deviceSelection?: boolean;
  selectedModelUrl?: string;
};

export type Lc0OrtEvaluatorFromUrl = {
  evaluator: Lc0OnnxEvaluator;
  /** Provenance of the model actually run; no retained model buffer. */
  modelLoad: Lc0ModelLoadInfo;
  requestedUrl: string;
  /** URL of the model the session runs: the cache identity for its evaluations. */
  servedUrl: string;
  variant: Lc0ModelVariant;
  /** Why the original model is served when split-K was selected (load or WebGPU failure). */
  fallbackReason?: string;
};

/** Injection points for unit tests. */
export type Lc0OrtEvaluatorDeps = {
  placement: () => Promise<Pick<OrtSessionPlacement, 'providers' | 'adapterVendor'>>;
  load: (url: string, options: Lc0ModelLoadOptions) => Promise<Lc0ModelLoadResult>;
  createEvaluator: (model: ModelBytes, options: Lc0OnnxEvaluatorOptions) => Promise<Lc0OnnxEvaluator>;
  clearCacheEntry?: typeof clearLc0ModelCacheEntry;
};

export const DEFAULT_LC0_ORT_EVALUATOR_DEPS: Lc0OrtEvaluatorDeps = {
  placement: resolveOrtSessionPlacement,
  load: loadLc0ModelForOrt,
  createEvaluator: (model, options) => Lc0OnnxEvaluator.create(model, options),
};

/**
 * Load an Lc0 ONNX model and create its ORT session, applying the device rule
 * when asked. Every product path that loads Lc0 Small goes through here (the
 * search worker, and the Play/Arena/Analysis main-thread loaders).
 *
 * - Split-K selected: its bytes are loaded under their own URL, so the model
 *   cache stores and verifies them as a separate entry.
 * - Split-K load fails (for example the CDN does not have it yet): warn and
 *   serve the original.
 * - The WebGPU session fails and createOrtSession falls back to WASM: that
 *   attempt loads the ORIGINAL URL, not the split-K bytes.
 */
export async function createLc0OnnxEvaluatorFromUrl(
  requestedUrl: string,
  options: Lc0OrtEvaluatorFromUrlOptions = {},
  deps: Lc0OrtEvaluatorDeps = DEFAULT_LC0_ORT_EVALUATOR_DEPS,
): Promise<Lc0OrtEvaluatorFromUrl> {
  const loadOptions = options.load ?? {};
  loadOptions.signal?.throwIfAborted();
  let selectedUrl = options.selectedModelUrl ?? requestedUrl;
  if (selectedUrl !== requestedUrl) {
    const { providers, adapterVendor } = await untilAborted(deps.placement(), loadOptions.signal);
    if (providers[0] !== 'webgpu' || (adapterVendor !== undefined && !lc0SplitKAdapterEligible(adapterVendor))) selectedUrl = requestedUrl;
  } else if (options.selectedModelUrl === undefined && options.deviceSelection) {
    selectedUrl = selectLc0OrtModelUrl(requestedUrl, await untilAborted(deps.placement(), loadOptions.signal));
  }
  loadOptions.signal?.throwIfAborted();
  let fallbackReason: string | undefined;
  let modelLoad: LoadedModel | undefined;
  let servedUrl = requestedUrl;
  if (selectedUrl !== requestedUrl) {
    try {
      modelLoad = loadedModel(await deps.load(selectedUrl, loadOptions));
      servedUrl = selectedUrl;
    } catch (error) {
      if ((error as { name?: unknown } | null)?.name === 'AbortError') throw error;
      fallbackReason = `split-K model unavailable: ${error instanceof Error ? error.message : String(error)}`;
      console.warn(`Lc0: ${fallbackReason}; loading the original model.`);
    }
  }
  modelLoad ??= loadedModel(await deps.load(requestedUrl, loadOptions));
  let originalLoad: LoadedModel | undefined;
  let usedOriginal = false;
  const reloadPrimary = async (): Promise<ModelBytes> => {
    modelLoad = loadedModel(await deps.load(servedUrl, { ...loadOptions, allowUnverified: false }), modelLoad?.info.revalidated);
    usedOriginal = false;
    loadOptions.signal?.throwIfAborted();
    return takeModel(modelLoad)!;
  };
  const wasmFallbackModel =
    servedUrl === requestedUrl
      ? undefined
      : async () => {
          if (originalLoad?.input === undefined) {
            originalLoad = loadedModel(await deps.load(requestedUrl, loadOptions), originalLoad?.info.revalidated);
          }
          usedOriginal = true;
          loadOptions.signal?.throwIfAborted();
          return takeModel(originalLoad)!;
        };
  // Defined outside create's activation: these callbacks capture provenance/
  // empty holders, never its transient model argument.
  const rejectModelParseErrors = () => isMetadataTrustedModel((usedOriginal ? originalLoad : modelLoad)?.info);
  const create = async () => {
    loadOptions.signal?.throwIfAborted();
    usedOriginal = false;
    let input = takeModel(modelLoad!);
    input ??= await reloadPrimary();
    const creating = deps.createEvaluator(input, {
      ...options.evaluator,
      ioMode: options.evaluator?.ioMode ?? defaultLc0OrtIoMode(options.deviceSelection === true),
      signal: loadOptions.signal,
      ...(wasmFallbackModel ? { wasmFallbackModel } : {}),
      ...(modelLoad?.info.sha256Valid === true ? { reloadModel: reloadPrimary } : {}),
      rejectModelParseErrors,
    });
    input = undefined; // Do not pin the unfolded bytes while ORT builds the folded session.
    return creating;
  };
  let evaluator: Lc0OnnxEvaluator;
  try {
    evaluator = await create();
  } catch (error) {
    loadOptions.signal?.throwIfAborted();
    const failedOnOriginal = usedOriginal;
    const failed = (failedOnOriginal ? originalLoad : modelLoad)?.info;
    if (!isOnnxModelParseError(error) || !isMetadataTrustedModel(failed) || !failed) throw error;
    console.warn(`Lc0: cached model could not be parsed; evicting and verifying one fresh load. ${failed.url}`);
    await (deps.clearCacheEntry ?? clearLc0ModelCacheEntry)(failed.url, loadOptions.cacheName, loadOptions.metadataStore);
    loadOptions.signal?.throwIfAborted();
    const reloaded = loadedModel(
      await deps.load(failedOnOriginal ? requestedUrl : servedUrl, { ...loadOptions, forceReload: true, allowUnverified: false }),
      true,
    );
    if (reloaded.info.verification !== 'hashed' || reloaded.info.sha256Valid !== true)
      throw new Error('LC0 model recovery requires freshly SHA-256-verified bytes');
    if (failedOnOriginal) {
      originalLoad = reloaded;
      // Keep only provenance until WASM is actually needed. An uncacheable
      // recovery must retain its verified bytes to avoid another model download.
      if (reloaded.info.mode === 'cache') reloaded.input = undefined;
    } else modelLoad = reloaded;
    // Deliberately not recursive: a second parse failure propagates without another refetch.
    evaluator = await create();
  }
  if (usedOriginal && originalLoad) {
    // createOrtSession only loads the fallback model when the WebGPU attempts failed
    // and it is about to run WASM; a returned session is therefore the original model.
    fallbackReason = 'WebGPU session failed; WASM runs the original model';
    return { evaluator, modelLoad: originalLoad.info, requestedUrl, servedUrl: requestedUrl, variant: 'original', fallbackReason };
  }
  return {
    evaluator,
    modelLoad: modelLoad!.info,
    requestedUrl,
    servedUrl,
    variant: servedUrl === requestedUrl ? 'original' : 'splitk',
    ...(fallbackReason ? { fallbackReason } : {}),
  };
}
