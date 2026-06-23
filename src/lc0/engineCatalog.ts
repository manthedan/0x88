export type EngineFamily = 'lc0' | 'tiny' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'plentychess';
export type EngineSurface = 'arena' | 'analysis';
export type EngineStrengthUnit = 'visits' | 'depth';

type ImportMetaWithEnv = ImportMeta & { env?: Record<string, string | undefined> };
const env = (import.meta as ImportMetaWithEnv).env ?? {};

export interface EngineRow {
  family: EngineFamily;
  variant: string;
  strength: number;
}

export interface EngineVariantOption {
  value: string;
  label: string;
  disabled?: boolean;
  experimental?: boolean;
}

export interface EngineStrengthMeta {
  unit: EngineStrengthUnit;
  min: number;
  max: number;
  def: number;
}

export interface EngineFamilyCatalogEntry {
  id: EngineFamily;
  label: string;
  shortLabel: string;
  status: 'stable' | 'experimental' | 'mixed';
  docHref: string;
  note: string;
}

export const ENGINE_FAMILY_PRIORITY: readonly EngineFamily[] = ['lc0', 'tiny', 'sf', 'reckless', 'viridithas', 'berserk', 'plentychess'];
const V0_ENGINE_FAMILY_PRIORITY: readonly EngineFamily[] = ['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess'];
const V0_RECKLESS_VARIANTS = new Set(['full', 'simd', 'relaxed-simd']);
const V0_BERSERK_VARIANTS = new Set(['emscripten', 'emscripten-simd', 'emscripten-relaxed']);
const V0_VIRIDITHAS_VARIANTS = new Set(['default', 'simd', 'relaxed-simd']);
const V0_PLENTYCHESS_VARIANTS = new Set(['emscripten', 'emscripten-sse41', 'emscripten-relaxed']);

export const ENGINE_FAMILY_CATALOG: Record<EngineFamily, EngineFamilyCatalogEntry> = {
  lc0: {
    id: 'lc0',
    label: 'Lc0',
    shortLabel: 'Lc0',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#lc0-family',
    note: 'Browser-native neural/search lane; small model is stable, BT4 and runtime experiments are gated.',
  },
  tiny: {
    id: 'tiny',
    label: 'Tiny Leela',
    shortLabel: 'TL',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#tiny-family',
    note: 'Tiny SquareFormer family; ORT is baseline and promoted custom WebGPU can be selected separately from LC0.',
  },
  sf: {
    id: 'sf',
    label: 'Stockfish',
    shortLabel: 'SF',
    status: 'stable',
    docHref: 'docs/engine_catalog.md#stockfish-family',
    note: 'NPM Stockfish 18 JS/WASM UCI baseline.',
  },
  reckless: {
    id: 'reckless',
    label: 'Reckless',
    shortLabel: 'Reck',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#reckless-family',
    note: 'Patched browser/WASI UCI engine; SIMD WASI is the strongest current candidate, browser API variants are experimental.',
  },
  viridithas: {
    id: 'viridithas',
    label: 'Viridithas',
    shortLabel: 'Viri',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#viridithas-family',
    note: 'Patched browser/WASI UCI engine; integration remains experimental.',
  },
  berserk: {
    id: 'berserk',
    label: 'Berserk',
    shortLabel: 'Berserk',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#berserk-family',
    note: 'Patched single-thread Emscripten UCI worker; early smoke passed, lifecycle remains experimental.',
  },
  plentychess: {
    id: 'plentychess',
    label: 'PlentyChess',
    shortLabel: 'Plenty',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#plentychess-family',
    note: 'Patched single-thread Emscripten UCI worker; smoked and benchmarked, but large .data sidecar keeps it experimental.',
  },
};

export interface EngineFamilyResourceProfile {
  resourceClass: 'cpu' | 'gpu';
  /** Max UCI threads the current browser build supports; the resource broker clamps grants to this. */
  maxThreads: number;
}

// Single-threaded entries become elastic by shipping a threaded build and
// raising maxThreads here; the broker needs no other change.
export const ENGINE_RESOURCE_PROFILES: Record<EngineFamily, EngineFamilyResourceProfile> = {
  lc0: { resourceClass: 'gpu', maxThreads: 1 },
  tiny: { resourceClass: 'gpu', maxThreads: 1 },
  sf: { resourceClass: 'cpu', maxThreads: 32 },
  reckless: { resourceClass: 'cpu', maxThreads: 1 },
  viridithas: { resourceClass: 'cpu', maxThreads: 1 },
  berserk: { resourceClass: 'cpu', maxThreads: 1 },
  plentychess: { resourceClass: 'cpu', maxThreads: 1 },
};

export function engineResourceProfile(family: EngineFamily): EngineFamilyResourceProfile {
  return ENGINE_RESOURCE_PROFILES[family];
}

export const LC0_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'small', label: 'Small' },
  { value: 't3', label: 't3-512 distill' },
  { value: 'bt4', label: 'BT4-it332', experimental: true },
];

/** Lc0 variants backed by the lazy WebGPU big-net worker (bt4Engine.ts). */
export function isLc0BigNetVariant(variant: string): variant is 'bt4' | 't3' {
  return variant === 'bt4' || variant === 't3';
}

export const TINY_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'bt4-auto', label: 'BT4 Anneal Muon Best · runtime auto' },
  { value: 'bt4-ort', label: 'BT4 Anneal Muon Best · ORT baseline' },
  { value: 'bt4-custom', label: 'BT4 Anneal Muon Best · custom WebGPU strict', experimental: true },
];

export const STOCKFISH_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'lite', label: 'Lite' },
  { value: 'full', label: 'Full' },
];

const ENGINE_STRENGTH: Record<EngineSurface, Record<EngineFamily, EngineStrengthMeta>> = {
  arena: {
    lc0: { unit: 'visits', min: 1, max: 100000, def: 100 },
    tiny: { unit: 'visits', min: 1, max: 100000, def: 100 },
    sf: { unit: 'depth', min: 1, max: 40, def: 8 },
    reckless: { unit: 'depth', min: 1, max: 30, def: 4 },
    viridithas: { unit: 'depth', min: 1, max: 20, def: 6 },
    berserk: { unit: 'depth', min: 1, max: 20, def: 4 },
    plentychess: { unit: 'depth', min: 1, max: 20, def: 4 },
  },
  analysis: {
    lc0: { unit: 'visits', min: 1, max: 100000, def: 400 },
    tiny: { unit: 'visits', min: 1, max: 100000, def: 400 },
    sf: { unit: 'depth', min: 1, max: 30, def: 14 },
    reckless: { unit: 'depth', min: 1, max: 30, def: 14 },
    viridithas: { unit: 'depth', min: 1, max: 20, def: 8 },
    berserk: { unit: 'depth', min: 1, max: 20, def: 8 },
    plentychess: { unit: 'depth', min: 1, max: 20, def: 8 },
  },
};

export function engineFamilyOptions(): { value: EngineFamily; label: string }[] {
  const families = isV0DeployProfile() ? V0_ENGINE_FAMILY_PRIORITY : ENGINE_FAMILY_PRIORITY;
  return families.map((family) => ({ value: family, label: ENGINE_FAMILY_CATALOG[family].label }));
}

export function isV0DeployProfile(): boolean {
  return env.VITE_BROWSER_CHESS_DEPLOY_PROFILE === 'v0';
}

export function engineStrengthMeta(family: EngineFamily, surface: EngineSurface): EngineStrengthMeta {
  return ENGINE_STRENGTH[surface][family];
}

export function defaultEngineStrength(family: EngineFamily, surface: EngineSurface): number {
  return engineStrengthMeta(family, surface).def;
}

export function normalizeDeployEngineRow(row: EngineRow, surface: EngineSurface, index = 0): EngineRow {
  const next: EngineRow = isV0DeployProfile()
    ? row.family === 'lc0'
      ? { ...row, family: 'lc0', variant: 'small' }
      : row.family === 'sf'
        ? { ...row, family: 'sf', variant: 'lite' }
        : row.family === 'reckless'
          ? { ...row, family: 'reckless', variant: V0_RECKLESS_VARIANTS.has(row.variant) ? row.variant : 'full' }
        : row.family === 'berserk'
          ? { ...row, family: 'berserk', variant: V0_BERSERK_VARIANTS.has(row.variant) ? row.variant : 'emscripten' }
        : row.family === 'viridithas'
          ? { ...row, family: 'viridithas', variant: V0_VIRIDITHAS_VARIANTS.has(row.variant) ? row.variant : 'default' }
        : row.family === 'plentychess'
          ? { ...row, family: 'plentychess', variant: V0_PLENTYCHESS_VARIANTS.has(row.variant) ? row.variant : 'emscripten' }
        : index % 2 === 0
          ? { family: 'lc0', variant: 'small', strength: defaultEngineStrength('lc0', surface) }
          : { family: 'sf', variant: 'lite', strength: defaultEngineStrength('sf', surface) }
    : { ...row };
  const meta = engineStrengthMeta(next.family, surface);
  next.strength = Math.max(meta.min, Math.min(meta.max, Math.floor(Number(next.strength) || meta.def)));
  return next;
}

export function lc0VariantOptions(bt4Supported: boolean): EngineVariantOption[] {
  // Both big-net variants need the same WebGPU support probe.
  const variants = isV0DeployProfile() ? LC0_ENGINE_VARIANTS.filter((option) => option.value === 'small') : LC0_ENGINE_VARIANTS;
  return variants.map((option) => ({ ...option, disabled: isLc0BigNetVariant(option.value) ? !bt4Supported : option.disabled }));
}

export function tinyVariantOptions(): EngineVariantOption[] {
  return TINY_ENGINE_VARIANTS.map((option) => ({ ...option }));
}

export function stockfishVariantOptions(): EngineVariantOption[] {
  const variants = isV0DeployProfile() ? STOCKFISH_ENGINE_VARIANTS.filter((option) => option.value === 'lite') : STOCKFISH_ENGINE_VARIANTS;
  return variants.map((option) => ({ ...option }));
}

export function defaultStaticEngineVariant(family: 'lc0' | 'tiny' | 'sf' | 'berserk' | 'plentychess' | 'viridithas'): string {
  if (family === 'tiny') return TINY_ENGINE_VARIANTS[0].value;
  if (family === 'sf') return STOCKFISH_ENGINE_VARIANTS[0].value;
  if (family === 'berserk' || family === 'plentychess') return 'emscripten';
  if (family === 'viridithas') return 'default';
  return LC0_ENGINE_VARIANTS[0].value;
}

export function lc0EngineLabel(variant: string): string {
  if (variant === 'bt4') return 'Lc0 BT4-it332';
  if (variant === 't3') return 'Lc0 t3-512';
  return 'Lc0';
}

export function stockfishEngineLabel(variant: string, surface: EngineSurface): string {
  if (surface === 'analysis') return variant === 'lite' ? 'SF Lite' : 'SF';
  return variant === 'lite' ? 'Stockfish Lite' : 'Stockfish';
}

export function tinyEngineLabel(variant: string): string {
  if (variant === 'bt4-ort') return 'Tiny Leela · ORT';
  if (variant === 'bt4-custom') return 'Tiny Leela · custom WebGPU';
  return 'Tiny Leela · auto';
}

export function isEngineFamily(value: string): value is EngineFamily {
  return value === 'lc0' || value === 'tiny' || value === 'sf' || value === 'reckless' || value === 'viridithas' || value === 'berserk' || value === 'plentychess';
}
