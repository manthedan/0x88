// Shared CPU-engine construction for the play / arena / analysis pages.
//
// Each page keeps its own variant-selection policy (URL params, custom
// variants, async asset fallback) and its own engine cache, but the
// CONSTRUCTION — default options, constructor wiring, and the cache-key
// derivation those caches rely on — is single-sourced here. Before this
// module the three pages carried parallel copies that had already drifted
// (arena's reckless cache key included the backend, analysis's did not, so
// two backends of the same variant collided in the analysis cache).
import type { BrowserUciEngine } from './browserUciEngine.ts';
import { RecklessEngine } from './recklessEngine.ts';
import { defaultRecklessVariantKey, recklessVariantByKey, resolveDefaultRecklessVariantAssetFallback, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, type ViridithasRuntimeOptions } from './viridithasEngine.ts';
import { defaultViridithasVariantKey, resolveDefaultViridithasVariantAssetFallback, viridithasVariantByKey, type ViridithasVariant } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { berserkVariantByKey, defaultBerserkVariantKey, resolveDefaultBerserkVariantAssetFallback, type BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { defaultPlentyChessVariantKey, plentyChessVariantByKey, resolveDefaultPlentyChessVariantAssetFallback, type PlentyChessVariant } from './plentychessVariants.ts';
import { StormphraxEngine } from './stormphraxEngine.ts';
import { defaultStormphraxVariantKey, resolveDefaultStormphraxVariantAssetFallback, stormphraxVariantByKey, type StormphraxVariant } from './stormphraxVariants.ts';

/** Construction-time search defaults; pages re-tune per move via setOptions. */
const CPU_ENGINE_DEFAULTS = { depth: 4, hashMb: 16 } as const;

export function recklessCacheKey(variant: RecklessVariant): string {
  return `${variant.key}:${variant.wasmUrl}:${variant.nnueUrl ?? ''}:${variant.nnueExpectedBytes ?? ''}:${variant.backend ?? 'wasi'}`;
}

export function viridithasCacheKey(variant: ViridithasVariant): string {
  return `${variant.key}:${variant.wasmUrl}`;
}

export function berserkCacheKey(variant: BerserkVariant): string {
  return `${variant.key}:${variant.jsUrl ?? ''}:${variant.wasmUrl}:${variant.dataUrl ?? ''}`;
}

export function plentyChessCacheKey(variant: PlentyChessVariant): string {
  return `${variant.key}:${variant.jsUrl}:${variant.wasmUrl}:${variant.dataUrl}`;
}

export function stormphraxCacheKey(variant: StormphraxVariant): string {
  return `${variant.key}:${variant.jsUrl}:${variant.wasmUrl}:${variant.dataUrl}`;
}

export function createRecklessEngine(variant: RecklessVariant, onStatus?: () => void): RecklessEngine {
  return new RecklessEngine({ ...CPU_ENGINE_DEFAULTS }, variant.wasmUrl, {
    backend: variant.backend ?? 'wasi',
    nnueUrl: variant.nnueUrl,
    nnueExpectedBytes: variant.nnueExpectedBytes,
    ...(onStatus ? { onStatus } : {}),
  });
}

export function createViridithasEngine(variant: ViridithasVariant, runtimeOptions: ViridithasRuntimeOptions = {}): ViridithasEngine {
  return new ViridithasEngine({ ...CPU_ENGINE_DEFAULTS }, variant.wasmUrl, runtimeOptions);
}

export function createBerserkEngine(variant: BerserkVariant): BerserkEngine {
  return new BerserkEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createPlentyChessEngine(variant: PlentyChessVariant): PlentyChessEngine {
  return new PlentyChessEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createStormphraxEngine(variant: StormphraxVariant): StormphraxEngine {
  return new StormphraxEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export type DefaultBrowserUciFamily = 'reckless' | 'viridithas' | 'berserk' | 'plentychess' | 'stormphrax';

type DefaultBrowserUciProvisioner = (onStatus?: () => void) => Promise<BrowserUciEngine>;

/**
 * Default-variant provisioning used by Play. Analysis and Arena still resolve
 * explicit variants themselves, but a standard UCI family no longer needs a
 * new Play-specific construction branch.
 */
export const DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS: Record<DefaultBrowserUciFamily, DefaultBrowserUciProvisioner> = {
  reckless: async (onStatus) => createRecklessEngine(
    await resolveDefaultRecklessVariantAssetFallback(recklessVariantByKey(defaultRecklessVariantKey()), false),
    onStatus,
  ),
  viridithas: async () => createViridithasEngine(
    await resolveDefaultViridithasVariantAssetFallback(viridithasVariantByKey(defaultViridithasVariantKey()), false),
  ),
  berserk: async () => createBerserkEngine(
    await resolveDefaultBerserkVariantAssetFallback(berserkVariantByKey(defaultBerserkVariantKey()), false),
  ),
  plentychess: async () => createPlentyChessEngine(
    await resolveDefaultPlentyChessVariantAssetFallback(plentyChessVariantByKey(defaultPlentyChessVariantKey()), false),
  ),
  stormphrax: async () => createStormphraxEngine(
    await resolveDefaultStormphraxVariantAssetFallback(stormphraxVariantByKey(defaultStormphraxVariantKey()), false),
  ),
};

export function isDefaultBrowserUciFamily(family: string): family is DefaultBrowserUciFamily {
  return Object.hasOwn(DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS, family);
}

export function createDefaultBrowserUciEngine(family: DefaultBrowserUciFamily, onStatus?: () => void): Promise<BrowserUciEngine> {
  return DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS[family](onStatus);
}
