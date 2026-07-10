export type EngineFamily = 'lc0' | 'centipawn' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'plentychess' | 'stormphrax';
export type EngineSurface = 'arena' | 'analysis';
export type EngineStrengthUnit = 'visits' | 'depth';
export type EngineRuntimeKind = 'neural' | 'uci';

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

export interface EngineFamilyResourceProfile {
  resourceClass: 'cpu' | 'gpu';
  /** Max UCI threads the current browser build supports; the resource broker clamps grants to this. */
  maxThreads: number;
}

export interface EnginePlayOptionDefinition {
  id: string;
  label: string;
  variant: string;
  group: 'odds' | 'engine';
  order: number;
  v0: boolean;
}

export interface EnginePlayDefinition {
  options: readonly EnginePlayOptionDefinition[];
  /** Five Play strength levels; Stockfish uses its separate skill/depth ladder. */
  levels?: readonly [number, number, number, number, number];
}

export interface EngineFamilyDefinition extends EngineFamilyCatalogEntry {
  runtime: EngineRuntimeKind;
  resource: EngineFamilyResourceProfile;
  strength: Record<EngineSurface, EngineStrengthMeta>;
  order: Record<'default' | 'v0', number>;
  variants: {
    default: string;
    /** Static options live here; families with runtime-probed variants supply their options in their typed variant module. */
    options?: readonly EngineVariantOption[];
    /** Production variants are an explicit allowlist. */
    v0Allowed: readonly string[];
    v0Fallback: string;
    label?: (variant: string, surface: EngineSurface) => string;
  };
  play?: EnginePlayDefinition;
  aliases?: readonly string[];
}

export const LC0_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'small', label: 'Small' },
  { value: 't3', label: 't3-512 distill' },
  { value: 'bt4', label: 'BT4-it332', experimental: true },
];

export const CENTIPAWN_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'bt4-ort', label: 'BT4 SOAP REM c19000 · ORT' },
  { value: 'bt4-auto', label: 'Legacy BT4 Anneal Muon · runtime auto', experimental: true },
  { value: 'bt4-custom', label: 'Legacy BT4 Anneal Muon · custom WebGPU strict', experimental: true },
];

export const STOCKFISH_ENGINE_VARIANTS: readonly EngineVariantOption[] = [
  { value: 'lite', label: 'Lite' },
  { value: 'full', label: 'Full' },
];

/**
 * Single source of truth for metadata shared by Analysis and Arena.
 *
 * Family-specific runtime adapters and feature probes remain in their typed
 * modules, but adding a family no longer requires parallel edits for selector
 * order, production allowlists, resources, strengths, labels, and aliases.
 */
export const ENGINE_FAMILY_DEFINITIONS = {
  lc0: {
    id: 'lc0',
    label: 'Lc0',
    shortLabel: 'Lc0',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#lc0-family',
    note: 'Browser-native neural/search lane; small model is stable, BT4 and runtime experiments are gated.',
    runtime: 'neural',
    resource: { resourceClass: 'gpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'visits', min: 1, max: 100000, def: 100 },
      analysis: { unit: 'visits', min: 1, max: 100000, def: 400 },
    },
    order: { default: 0, v0: 0 },
    variants: {
      default: 'small',
      options: LC0_ENGINE_VARIANTS,
      v0Allowed: ['small', 'bt4'],
      v0Fallback: 'small',
      label: (variant) => variant === 'bt4' ? 'Lc0 BT4-it332' : variant === 't3' ? 'Lc0 t3-512' : 'Lc0',
    },
    play: {
      options: [
        { id: 'leela-queen-odds', label: 'Leela Queen Odds', variant: 'lqo', group: 'odds', order: 0, v0: true },
        { id: 'lc0-small', label: 'Lc0 · Small net', variant: 'small', group: 'engine', order: 3, v0: true },
        { id: 'lc0-t3', label: 'Lc0 · t3-512 distill', variant: 't3', group: 'engine', order: 4, v0: false },
        { id: 'lc0-bt4', label: 'Lc0 · BT4-it332', variant: 'bt4', group: 'engine', order: 5, v0: true },
      ],
      levels: [8, 32, 100, 400, 1600],
    },
  },
  sf: {
    id: 'sf',
    label: 'Stockfish',
    shortLabel: 'SF',
    status: 'stable',
    docHref: 'docs/engine_catalog.md#stockfish-family',
    note: 'NPM Stockfish 18 JS/WASM UCI baseline.',
    runtime: 'uci',
    resource: { resourceClass: 'cpu', maxThreads: 32 },
    strength: {
      arena: { unit: 'depth', min: 1, max: 40, def: 8 },
      analysis: { unit: 'depth', min: 1, max: 30, def: 12 },
    },
    order: { default: 1, v0: 1 },
    variants: {
      default: 'lite',
      options: STOCKFISH_ENGINE_VARIANTS,
      v0Allowed: ['lite'],
      v0Fallback: 'lite',
      label: (variant, surface) => surface === 'analysis'
        ? variant === 'lite' ? 'SF Lite' : 'SF'
        : variant === 'lite' ? 'Stockfish Lite' : 'Stockfish',
    },
    play: {
      options: [
        { id: 'sf-lite', label: 'Stockfish Lite', variant: 'lite', group: 'engine', order: 1, v0: true },
        { id: 'sf-full', label: 'Stockfish', variant: 'full', group: 'engine', order: 2, v0: false },
      ],
    },
  },
  reckless: {
    id: 'reckless',
    label: 'Reckless',
    shortLabel: 'Reck',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#reckless-family',
    note: 'Patched browser/WASI UCI engine; SIMD WASI is the strongest current candidate, browser API variants are experimental.',
    runtime: 'uci',
    resource: { resourceClass: 'cpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'depth', min: 1, max: 30, def: 4 },
      analysis: { unit: 'depth', min: 1, max: 30, def: 12 },
    },
    order: { default: 2, v0: 2 },
    variants: {
      default: 'relaxed-simd',
      v0Allowed: ['full', 'simd', 'relaxed-simd'],
      v0Fallback: 'full',
    },
    play: {
      options: [{ id: 'reckless', label: 'Reckless', variant: 'default', group: 'engine', order: 6, v0: true }],
      levels: [2, 4, 6, 10, 14],
    },
  },
  viridithas: {
    id: 'viridithas',
    label: 'Viridithas',
    shortLabel: 'Viri',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#viridithas-family',
    note: 'Patched browser/WASI UCI engine; integration remains experimental.',
    runtime: 'uci',
    resource: { resourceClass: 'cpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'depth', min: 1, max: 20, def: 6 },
      analysis: { unit: 'depth', min: 1, max: 20, def: 12 },
    },
    order: { default: 3, v0: 4 },
    variants: {
      default: 'default',
      v0Allowed: ['default', 'simd', 'relaxed-simd'],
      v0Fallback: 'default',
    },
    play: {
      options: [{ id: 'viridithas', label: 'Viridithas', variant: 'default', group: 'engine', order: 7, v0: true }],
      levels: [2, 4, 6, 9, 12],
    },
  },
  berserk: {
    id: 'berserk',
    label: 'Berserk',
    shortLabel: 'Berserk',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#berserk-family',
    note: 'Patched single-thread Emscripten UCI worker; early smoke passed, lifecycle remains experimental.',
    runtime: 'uci',
    resource: { resourceClass: 'cpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'depth', min: 1, max: 20, def: 4 },
      analysis: { unit: 'depth', min: 1, max: 20, def: 12 },
    },
    order: { default: 4, v0: 3 },
    variants: {
      default: 'emscripten',
      v0Allowed: ['emscripten', 'emscripten-simd', 'emscripten-relaxed'],
      v0Fallback: 'emscripten',
    },
    play: {
      options: [{ id: 'berserk', label: 'Berserk', variant: 'default', group: 'engine', order: 8, v0: true }],
      levels: [2, 4, 6, 9, 12],
    },
  },
  plentychess: {
    id: 'plentychess',
    label: 'PlentyChess',
    shortLabel: 'Plenty',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#plentychess-family',
    note: 'Patched single-thread Emscripten UCI worker; smoked and benchmarked, but large .data sidecar keeps it experimental.',
    runtime: 'uci',
    resource: { resourceClass: 'cpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'depth', min: 1, max: 20, def: 4 },
      analysis: { unit: 'depth', min: 1, max: 20, def: 12 },
    },
    order: { default: 5, v0: 5 },
    variants: {
      default: 'emscripten',
      v0Allowed: ['emscripten', 'emscripten-sse41', 'emscripten-relaxed'],
      v0Fallback: 'emscripten',
    },
    play: {
      options: [{ id: 'plentychess', label: 'PlentyChess', variant: 'default', group: 'engine', order: 9, v0: true }],
      levels: [2, 4, 6, 9, 12],
    },
  },
  stormphrax: {
    id: 'stormphrax',
    label: 'Stormphrax',
    shortLabel: 'Storm',
    status: 'experimental',
    docHref: 'docs/engine_catalog.md#stormphrax-family',
    note: 'Stormphrax 8.0.0 single-thread Emscripten UCI worker with its undertown NNUE.',
    runtime: 'uci',
    resource: { resourceClass: 'cpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'depth', min: 1, max: 20, def: 4 },
      analysis: { unit: 'depth', min: 1, max: 20, def: 12 },
    },
    order: { default: 6, v0: 6 },
    variants: {
      default: 'emscripten',
      v0Allowed: ['emscripten'],
      v0Fallback: 'emscripten',
    },
    play: {
      options: [{ id: 'stormphrax', label: 'Stormphrax', variant: 'emscripten', group: 'engine', order: 10, v0: true }],
      levels: [2, 4, 6, 9, 12],
    },
  },
  centipawn: {
    id: 'centipawn',
    label: 'Centipawn',
    shortLabel: 'Centi',
    status: 'mixed',
    docHref: 'docs/engine_catalog.md#centipawn-family',
    note: 'Centipawn SquareFormer neural family; ORT is baseline and promoted custom WebGPU can be selected separately from LC0.',
    runtime: 'neural',
    resource: { resourceClass: 'gpu', maxThreads: 1 },
    strength: {
      arena: { unit: 'visits', min: 1, max: 100000, def: 100 },
      analysis: { unit: 'visits', min: 1, max: 100000, def: 400 },
    },
    order: { default: 7, v0: 7 },
    variants: {
      default: 'bt4-ort',
      options: CENTIPAWN_ENGINE_VARIANTS,
      v0Allowed: ['bt4-ort'],
      v0Fallback: 'bt4-ort',
      label: (variant) => variant === 'bt4-ort'
        ? 'Centipawn'
        : variant === 'bt4-custom' ? 'Centipawn · custom WebGPU' : 'Centipawn · auto',
    },
    play: {
      options: [{ id: 'centipawn', label: 'Centipawn', variant: 'bt4-ort', group: 'engine', order: 11, v0: true }],
      levels: [8, 32, 100, 400, 1600],
    },
    aliases: ['tiny'],
  },
} as const satisfies Record<EngineFamily, EngineFamilyDefinition>;

function familyDefinitions(): EngineFamilyDefinition[] {
  return Object.values(ENGINE_FAMILY_DEFINITIONS) as EngineFamilyDefinition[];
}

function orderedFamilies(profile: 'default' | 'v0'): readonly EngineFamily[] {
  return Object.freeze(familyDefinitions()
    .slice()
    .sort((a, b) => a.order[profile] - b.order[profile])
    .map((definition) => definition.id));
}

export const ENGINE_FAMILY_PRIORITY = orderedFamilies('default');
export const V0_ENGINE_FAMILY_PRIORITY = orderedFamilies('v0');

export const ENGINE_FAMILY_CATALOG = Object.fromEntries(familyDefinitions().map((definition) => [definition.id, {
  id: definition.id,
  label: definition.label,
  shortLabel: definition.shortLabel,
  status: definition.status,
  docHref: definition.docHref,
  note: definition.note,
}])) as Record<EngineFamily, EngineFamilyCatalogEntry>;

export const ENGINE_RESOURCE_PROFILES = Object.fromEntries(familyDefinitions().map((definition) => [definition.id, definition.resource])) as Record<EngineFamily, EngineFamilyResourceProfile>;

export function engineFamilyDefinition(family: EngineFamily): EngineFamilyDefinition {
  return ENGINE_FAMILY_DEFINITIONS[family] as EngineFamilyDefinition;
}

export function engineResourceProfile(family: EngineFamily): EngineFamilyResourceProfile {
  return engineFamilyDefinition(family).resource;
}

/** Lc0 variants backed by the lazy WebGPU big-net worker (bt4Engine.ts). */
export function isLc0BigNetVariant(variant: string): variant is 'bt4' | 't3' {
  return variant === 'bt4' || variant === 't3';
}

export function engineFamilyOptions(): { value: EngineFamily; label: string }[] {
  const families = isV0DeployProfile() ? V0_ENGINE_FAMILY_PRIORITY : ENGINE_FAMILY_PRIORITY;
  return families.map((family) => ({ value: family, label: engineFamilyDefinition(family).label }));
}

export function enginePlayOptions(): Array<EnginePlayOptionDefinition & { family: EngineFamily }> {
  return familyDefinitions()
    .flatMap((definition) => (definition.play?.options ?? []).map((option) => ({ ...option, family: definition.id })))
    .filter((option) => !isV0DeployProfile() || option.v0)
    .sort((a, b) => a.order - b.order);
}

export function enginePlayLevels(family: EngineFamily): readonly [number, number, number, number, number] {
  const levels = engineFamilyDefinition(family).play?.levels;
  if (!levels) throw new Error(`Engine family ${family} does not use the shared Play strength ladder`);
  return levels;
}

export function isV0DeployProfile(): boolean {
  return env.VITE_BROWSER_CHESS_DEPLOY_PROFILE === 'v0';
}

export function engineStrengthMeta(family: EngineFamily, surface: EngineSurface): EngineStrengthMeta {
  return engineFamilyDefinition(family).strength[surface];
}

export function defaultEngineStrength(family: EngineFamily, surface: EngineSurface): number {
  return engineStrengthMeta(family, surface).def;
}

export function normalizeDeployEngineRow(row: EngineRow, surface: EngineSurface, index = 0): EngineRow {
  const definition = (ENGINE_FAMILY_DEFINITIONS as Partial<Record<string, EngineFamilyDefinition>>)[row.family];
  let next: EngineRow;
  if (!definition) {
    next = index % 2 === 0
      ? { family: 'lc0', variant: engineFamilyDefinition('lc0').variants.default, strength: defaultEngineStrength('lc0', surface) }
      : { family: 'sf', variant: engineFamilyDefinition('sf').variants.default, strength: defaultEngineStrength('sf', surface) };
  } else {
    const variant = isV0DeployProfile() && !definition.variants.v0Allowed.includes(row.variant)
      ? definition.variants.v0Fallback
      : row.variant;
    next = { ...row, variant };
  }
  const meta = engineStrengthMeta(next.family, surface);
  next.strength = Math.max(meta.min, Math.min(meta.max, Math.floor(Number(next.strength) || meta.def)));
  return next;
}

export function lc0VariantOptions(bt4Supported: boolean): EngineVariantOption[] {
  const definition = engineFamilyDefinition('lc0');
  const variants = isV0DeployProfile()
    ? definition.variants.options!.filter((option) => definition.variants.v0Allowed.includes(option.value))
    : definition.variants.options!;
  return variants.map((option) => ({ ...option, disabled: isLc0BigNetVariant(option.value) ? !bt4Supported : option.disabled }));
}

export function centipawnVariantOptions(): EngineVariantOption[] {
  const definition = engineFamilyDefinition('centipawn');
  const variants = isV0DeployProfile()
    ? definition.variants.options!.filter((option) => definition.variants.v0Allowed.includes(option.value))
    : definition.variants.options!;
  return variants.map((option) => ({ ...option }));
}

export function stockfishVariantOptions(): EngineVariantOption[] {
  const definition = engineFamilyDefinition('sf');
  const variants = isV0DeployProfile()
    ? definition.variants.options!.filter((option) => definition.variants.v0Allowed.includes(option.value))
    : definition.variants.options!;
  return variants.map((option) => ({ ...option }));
}

export function defaultStaticEngineVariant(family: EngineFamily): string {
  return engineFamilyDefinition(family).variants.default;
}

export function lc0EngineLabel(variant: string): string {
  return engineFamilyDefinition('lc0').variants.label!(variant, 'arena');
}

export function stockfishEngineLabel(variant: string, surface: EngineSurface): string {
  return engineFamilyDefinition('sf').variants.label!(variant, surface);
}

export function centipawnEngineLabel(variant: string): string {
  return engineFamilyDefinition('centipawn').variants.label!(variant, 'arena');
}

export function isEngineFamily(value: string): value is EngineFamily {
  return Object.hasOwn(ENGINE_FAMILY_DEFINITIONS, value);
}

const LEGACY_ENGINE_FAMILY_ALIASES = Object.fromEntries(familyDefinitions().flatMap((definition) =>
  (definition.aliases ?? []).map((alias) => [alias, definition.id]),
)) as Record<string, EngineFamily>;

/** Map a possibly-legacy family string to its canonical key, or null when unknown. */
export function canonicalEngineFamily(value: string): EngineFamily | null {
  const aliased = LEGACY_ENGINE_FAMILY_ALIASES[value] ?? value;
  return isEngineFamily(aliased) ? aliased : null;
}
