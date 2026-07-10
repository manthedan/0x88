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
  const suffix = unsupportedReason
    ? ` (${unsupportedReason})`
    : status === 'missing'
      ? ' (asset missing)'
      : checkingRequiredAsset
        ? ' (checking asset)'
        : '';
  return { value, label: `${label}${suffix}`, disabled };
}
