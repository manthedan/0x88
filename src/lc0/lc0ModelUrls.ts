import type { OrtSessionPlacement } from '../nn/ortConfig.ts';

/** The shipped Lc0 Small model: batch-1 f16 export with weight-only int8 QDQ. */
export const LC0_SMALL_BATCH1_QDQ8_FILE = 't1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';
export const LC0_SMALL_BATCH1_QDQ8_PATH = `/models/lc0/${LC0_SMALL_BATCH1_QDQ8_FILE}`;
/**
 * Split-K rewrite of the same model (docs/lc0_small_splitk_2026-11.md). Same
 * weights, outputs equal up to f16 summation order. About 1.5x faster per
 * invoke on Apple M4 ORT WebGPU, 1.21x slower on ORT WASM, and not shown to
 * help on NVIDIA, so it is served only to WebGPU sessions on Apple adapters.
 */
export const LC0_SMALL_BATCH1_SPLITK_FILE = 't1-256x10-distilled-swa-2432500.batch1.f16.qdq8.splitk.onnx';

export type Lc0ModelVariant = 'original' | 'splitk';

/** Adapters the split-K model was measured faster on. */
export function lc0SplitKAdapterEligible(adapterVendor: string | undefined): boolean {
  return adapterVendor?.trim().toLowerCase() === 'apple';
}

/**
 * Smolgen-compress split-K rewrites of the big nets (scripts/kernel_hillclimb/splitk_rewrite.py
 * --include 'smolgen/dense1/w$' --assume-m 8; docs/bignet_smolgen_splitk_2026-11.md). Only the
 * 15 smolgen dense1 MatMuls change. A net is listed only where its paired search A/B held up
 * on Apple M4 WebGPU: t3 (~6-10% faster per move). BT4 was within noise, and LQO gained only
 * 3-5% with its gate WDL deviation at the limit, so both keep their shipped model.
 */
export const LC0_BIG_NET_SPLITK_FILES: Readonly<Record<string, string>> = {
  't3-512x15x16h-distill-swa-2767500.batch8.f16.onnx': 't3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx',
};

/** Shipped file name -> its split-K rewrite, for every net that has one. */
const SPLITK_FILES: Readonly<Record<string, string>> = { [LC0_SMALL_BATCH1_QDQ8_FILE]: LC0_SMALL_BATCH1_SPLITK_FILE, ...LC0_BIG_NET_SPLITK_FILES };

function replaceFileName(modelUrl: string, files: Readonly<Record<string, string>>): string | undefined {
  const match = /(^|\/)([^/?#]+)([?#].*)?$/.exec(modelUrl);
  const replacement = match && Object.hasOwn(files, match[2]) ? files[match[2]] : undefined;
  if (!match || !replacement) return undefined;
  return `${modelUrl.slice(0, match.index)}${match[1]}${replacement}${match[3] ?? ''}`;
}

/** The split-K URL for a URL naming the shipped Lc0 Small QDQ8 file (any origin/prefix, query kept), else undefined. */
export function lc0SmallSplitKUrl(modelUrl: string): string | undefined {
  return replaceFileName(modelUrl, { [LC0_SMALL_BATCH1_QDQ8_FILE]: LC0_SMALL_BATCH1_SPLITK_FILE });
}

/** The split-K URL for any shipped model that has one (Lc0 Small or a listed big net), else undefined. */
export function lc0SplitKUrl(modelUrl: string): string | undefined {
  return replaceFileName(modelUrl, SPLITK_FILES);
}

/**
 * The one selection rule. A split-K model replaces its shipped model only when
 * the session will run on ORT WebGPU (first resolved provider) AND the adapter
 * vendor is Apple. Everything else, and every other URL, is unchanged.
 */
export function selectLc0OrtModelUrl(modelUrl: string, placement: Pick<OrtSessionPlacement, 'providers' | 'adapterVendor'>): string {
  if (placement.providers[0] !== 'webgpu') return modelUrl;
  if (!lc0SplitKAdapterEligible(placement.adapterVendor)) return modelUrl;
  return lc0SplitKUrl(modelUrl) ?? modelUrl;
}

/**
 * The Lc0 Small model a product page loads, and whether the device rule may
 * replace it. Any `?model=` override (even one naming the default file) is
 * explicit and loads exactly that URL; only the page default opts in.
 */
export function lc0SmallModelRequest(
  defaultUrl: string,
  override: string | null | undefined,
  resolveUrl: (url: string) => string = (url) => url,
): { modelUrl: string; deviceSelection: boolean } {
  if (override === null || override === undefined) return { modelUrl: defaultUrl, deviceSelection: true };
  return { modelUrl: resolveUrl(override), deviceSelection: false };
}
