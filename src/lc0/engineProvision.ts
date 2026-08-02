// Shared CPU-engine construction for the play / arena / analysis pages.
//
// Each page keeps its own variant-selection policy (URL params, custom
// variants, async asset fallback) and its own engine cache, but the
// CONSTRUCTION — default options, constructor wiring, and the cache-key
// derivation those caches rely on — is single-sourced here. Before this
// module the three pages carried parallel copies that had already drifted
// (arena's reckless cache key included the backend, analysis's did not, so
// two backends of the same variant collided in the analysis cache).

import { BerserkEngine } from './berserkEngine.ts';
import { type BerserkVariant, berserkVariantByKey, defaultBerserkVariantKey, resolveDefaultBerserkVariantAssetFallback } from './berserkVariants.ts';
import type { BrowserUciEngine } from './browserUciEngine.ts';
import { type CpuEngineSurface, cpuEngineHashMbForSurface } from './cpuEngineMemory.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import {
  defaultPlentyChessVariantKey,
  type PlentyChessVariant,
  plentyChessVariantByKey,
  resolveDefaultPlentyChessVariantAssetFallback,
} from './plentychessVariants.ts';
import { RecklessEngine } from './recklessEngine.ts';
import { defaultRecklessVariantKey, type RecklessVariant, recklessVariantByKey, resolveDefaultRecklessVariantAssetFallback } from './recklessVariants.ts';
import { StormphraxEngine } from './stormphraxEngine.ts';
import {
  defaultStormphraxVariantKey,
  resolveDefaultStormphraxVariantAssetFallback,
  type StormphraxVariant,
  stormphraxVariantByKey,
} from './stormphraxVariants.ts';
import { ViridithasEngine, type ViridithasRuntimeOptions } from './viridithasEngine.ts';
import {
  defaultViridithasVariantKey,
  resolveDefaultViridithasVariantAssetFallback,
  type ViridithasVariant,
  viridithasVariantByKey,
} from './viridithasVariants.ts';

export type { CpuEngineSurface } from './cpuEngineMemory.ts';
export { CPU_ENGINE_HASH_MB_BY_SURFACE, cpuEngineHashMbForSurface, DEFAULT_CPU_ENGINE_HASH_MB } from './cpuEngineMemory.ts';

export interface CpuEngineProvisionOptions {
  surface?: CpuEngineSurface;
}

function cpuEngineDefaults(options: CpuEngineProvisionOptions): { depth: number; hashMb: number } {
  return { depth: 4, hashMb: cpuEngineHashMbForSurface(options.surface ?? 'play') };
}

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

export function createRecklessEngine(variant: RecklessVariant, onStatus?: () => void, options: CpuEngineProvisionOptions = {}): RecklessEngine {
  return new RecklessEngine(cpuEngineDefaults(options), variant.wasmUrl, {
    backend: variant.backend ?? 'wasi',
    nnueUrl: variant.nnueUrl,
    nnueExpectedBytes: variant.nnueExpectedBytes,
    ...(onStatus ? { onStatus } : {}),
  });
}

export function createViridithasEngine(
  variant: ViridithasVariant,
  runtimeOptions: ViridithasRuntimeOptions = {},
  options: CpuEngineProvisionOptions = {},
): ViridithasEngine {
  return new ViridithasEngine(cpuEngineDefaults(options), variant.wasmUrl, runtimeOptions);
}

export function createBerserkEngine(variant: BerserkVariant, options: CpuEngineProvisionOptions = {}): BerserkEngine {
  return new BerserkEngine({ ...cpuEngineDefaults(options), threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createPlentyChessEngine(variant: PlentyChessVariant, options: CpuEngineProvisionOptions = {}): PlentyChessEngine {
  return new PlentyChessEngine({ ...cpuEngineDefaults(options), threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createStormphraxEngine(variant: StormphraxVariant, options: CpuEngineProvisionOptions = {}): StormphraxEngine {
  return new StormphraxEngine({ ...cpuEngineDefaults(options), threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export type DefaultBrowserUciFamily = 'reckless' | 'viridithas' | 'berserk' | 'plentychess' | 'stormphrax';

type DefaultBrowserUciProvisioner = (onStatus?: () => void) => Promise<BrowserUciEngine>;

/**
 * Default-variant provisioning used by Play. Analysis and Arena still resolve
 * explicit variants themselves, but a standard UCI family no longer needs a
 * new Play-specific construction branch.
 */
export const DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS: Record<DefaultBrowserUciFamily, DefaultBrowserUciProvisioner> = {
  reckless: async (onStatus) =>
    createRecklessEngine(await resolveDefaultRecklessVariantAssetFallback(recklessVariantByKey(defaultRecklessVariantKey()), false), onStatus),
  viridithas: async () =>
    createViridithasEngine(await resolveDefaultViridithasVariantAssetFallback(viridithasVariantByKey(defaultViridithasVariantKey()), false)),
  berserk: async () => createBerserkEngine(await resolveDefaultBerserkVariantAssetFallback(berserkVariantByKey(defaultBerserkVariantKey()), false)),
  plentychess: async () =>
    createPlentyChessEngine(await resolveDefaultPlentyChessVariantAssetFallback(plentyChessVariantByKey(defaultPlentyChessVariantKey()), false)),
  stormphrax: async () =>
    createStormphraxEngine(await resolveDefaultStormphraxVariantAssetFallback(stormphraxVariantByKey(defaultStormphraxVariantKey()), false)),
};

export function isDefaultBrowserUciFamily(family: string): family is DefaultBrowserUciFamily {
  return Object.hasOwn(DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS, family);
}

export function createDefaultBrowserUciEngine(family: DefaultBrowserUciFamily, onStatus?: () => void): Promise<BrowserUciEngine> {
  return DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS[family](onStatus);
}
