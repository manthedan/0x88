export type EngineFamily = 'lc0' | 'sf' | 'reckless' | 'viridithas';
export type EngineSurface = 'arena' | 'analysis';
export type EngineStrengthUnit = 'visits' | 'depth';

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

export const ENGINE_FAMILY_PRIORITY: readonly EngineFamily[] = ['lc0', 'sf', 'reckless', 'viridithas'];

export const ENGINE_FAMILY_CATALOG: Record<EngineFamily, EngineFamilyCatalogEntry> = {
  lc0: {
    id: 'lc0',
    label: 'Lc0',
    shortLabel: 'Lc0',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#lc0-family',
    note: 'Browser-native neural/search lane; small model is stable, BT4 and runtime experiments are gated.',
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
};

export const LC0_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'small', label: 'Small' },
  { value: 'bt4', label: 'BT4', experimental: true },
];

export const STOCKFISH_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'lite', label: 'Lite' },
  { value: 'full', label: 'Full' },
];

const ENGINE_STRENGTH: Record<EngineSurface, Record<EngineFamily, EngineStrengthMeta>> = {
  arena: {
    lc0: { unit: 'visits', min: 1, max: 100000, def: 100 },
    sf: { unit: 'depth', min: 1, max: 40, def: 8 },
    reckless: { unit: 'depth', min: 1, max: 30, def: 4 },
    viridithas: { unit: 'depth', min: 1, max: 20, def: 6 },
  },
  analysis: {
    lc0: { unit: 'visits', min: 1, max: 100000, def: 400 },
    sf: { unit: 'depth', min: 1, max: 30, def: 14 },
    reckless: { unit: 'depth', min: 1, max: 30, def: 14 },
    viridithas: { unit: 'depth', min: 1, max: 20, def: 8 },
  },
};

export function engineFamilyOptions(): { value: EngineFamily; label: string }[] {
  return ENGINE_FAMILY_PRIORITY.map((family) => ({ value: family, label: ENGINE_FAMILY_CATALOG[family].label }));
}

export function engineStrengthMeta(family: EngineFamily, surface: EngineSurface): EngineStrengthMeta {
  return ENGINE_STRENGTH[surface][family];
}

export function defaultEngineStrength(family: EngineFamily, surface: EngineSurface): number {
  return engineStrengthMeta(family, surface).def;
}

export function lc0VariantOptions(bt4Supported: boolean): EngineVariantOption[] {
  return LC0_ENGINE_VARIANTS.map((option) => ({ ...option, disabled: option.value === 'bt4' ? !bt4Supported : option.disabled }));
}

export function stockfishVariantOptions(): EngineVariantOption[] {
  return STOCKFISH_ENGINE_VARIANTS.map((option) => ({ ...option }));
}

export function defaultStaticEngineVariant(family: 'lc0' | 'sf'): string {
  if (family === 'sf') return STOCKFISH_ENGINE_VARIANTS[0].value;
  return LC0_ENGINE_VARIANTS[0].value;
}

export function lc0EngineLabel(variant: string): string {
  return variant === 'bt4' ? 'Lc0 BT4' : 'Lc0';
}

export function stockfishEngineLabel(variant: string, surface: EngineSurface): string {
  if (surface === 'analysis') return variant === 'lite' ? 'SF Lite' : 'SF';
  return variant === 'lite' ? 'Stockfish Lite' : 'Stockfish';
}

export function isEngineFamily(value: string): value is EngineFamily {
  return value === 'lc0' || value === 'sf' || value === 'reckless' || value === 'viridithas';
}
