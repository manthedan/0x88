import type { EngineVariantOption } from './engineCatalog.ts';

export type BrowserVariantAssetStatus = 'unknown' | 'checking' | 'present' | 'ok' | 'missing';

export interface BrowserVariantOptionState {
  assetStatus: BrowserVariantAssetStatus;
  unsupportedReason?: string | null;
  /** Disable and annotate an option until its generated artifact probe succeeds. */
  requirePresent?: boolean;
}

/** Convert family-specific probe results into consistent selector behavior. */
export function browserVariantOption(value: string, label: string, state: BrowserVariantOptionState): EngineVariantOption {
  const status = state.assetStatus === 'ok' ? 'present' : state.assetStatus;
  const unsupportedReason = state.unsupportedReason ?? null;
  const checkingRequiredAsset = state.requirePresent === true && status !== 'present' && status !== 'missing';
  const disabled = Boolean(unsupportedReason) || status === 'missing' || checkingRequiredAsset;
  const suffix = unsupportedReason ? ` (${unsupportedReason})` : status === 'missing' ? ' (asset missing)' : checkingRequiredAsset ? ' (checking asset)' : '';
  return { value, label: `${label}${suffix}`, disabled };
}

type BrowserVariant = { key: string };

export function availableBrowserVariants<T extends BrowserVariant>(builtIns: readonly T[], requested: T, usable: (variant: T) => boolean = () => true): T[] {
  const variants = builtIns.filter(usable);
  if (usable(requested) && !variants.some((variant) => variant.key === requested.key)) variants.push(requested);
  return variants;
}

export function resolveBrowserVariant<T extends BrowserVariant, K extends string>(
  variantKey: string,
  requested: T,
  normalize: (value: string | null) => K,
  byKey: (key: K) => T,
  builtIns: readonly T[],
  usable: (variant: T) => boolean = () => true,
): T {
  const key = normalize(variantKey);
  if (key === 'custom' && requested.key === 'custom' && usable(requested)) return requested;
  const variant = byKey(key);
  if (usable(variant)) return variant;
  const fallback = builtIns.find(usable);
  if (!fallback) throw new Error(`No usable browser engine variant for ${variantKey}`);
  return fallback;
}

export interface BrowserVariantSelector<T extends BrowserVariant> {
  available(): T[];
  resolve(variantKey: string): T;
}

export function createBrowserVariantSelector<T extends BrowserVariant, K extends string>(options: {
  builtIns: readonly T[];
  requested: () => T;
  normalize: (value: string | null) => K;
  byKey: (key: K) => T;
  usable?: (variant: T) => boolean;
}): BrowserVariantSelector<T> {
  const usable = options.usable ?? (() => true);
  return {
    available: () => availableBrowserVariants(options.builtIns, options.requested(), usable),
    resolve: (variantKey) => resolveBrowserVariant(variantKey, options.requested(), options.normalize, options.byKey, options.builtIns, usable),
  };
}
