import type { EngineFamily } from './engineCatalog.ts';

export type EngineLogoFamily = Extract<EngineFamily, 'lc0' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'centipawn'> | 'generic';

interface EngineLogoAsset {
  family: EngineLogoFamily;
  url: string;
}

const ENGINE_LOGO_ASSETS: Record<EngineLogoFamily, EngineLogoAsset> = {
  lc0: { family: 'lc0', url: '/engine-logos/lc0.svg' },
  centipawn: { family: 'centipawn', url: '/engine-logos/centipawn.svg' },
  sf: { family: 'sf', url: '/engine-logos/stockfish.png' },
  reckless: { family: 'reckless', url: '/engine-logos/reckless.png' },
  viridithas: { family: 'viridithas', url: '/engine-logos/viridithas.png' },
  berserk: { family: 'berserk', url: '/engine-logos/berserk.jpg' },
  generic: { family: 'generic', url: '/engine-logos/generic.svg' },
};

export function engineLogoFamilyForName(name: string): EngineLogoFamily {
  const n = name.toLowerCase();
  if (n.includes('centipawn')) return 'centipawn';
  if (n.includes('bt4') || n.includes('lc0') || n.includes('leela')) return 'lc0';
  if (n.includes('reckless')) return 'reckless';
  if (n.includes('viridithas')) return 'viridithas';
  if (n.includes('berserk')) return 'berserk';
  if (n.includes('stockfish') || /\bsf\b/.test(n)) return 'sf';
  return 'generic';
}

export function engineLogoFamilyForEngineFamily(family: EngineFamily): EngineLogoFamily {
  if (family === 'lc0') return 'lc0';
  if (family === 'centipawn') return 'centipawn';
  if (family === 'sf') return 'sf';
  if (family === 'reckless' || family === 'viridithas' || family === 'berserk') return family;
  return 'generic';
}

export function engineLogoUrl(family: EngineLogoFamily | undefined): string | undefined {
  return family ? ENGINE_LOGO_ASSETS[family]?.url : undefined;
}

export function engineLogoHtml(family: EngineLogoFamily | undefined, className = 'engine-logo'): string {
  const { url } = ENGINE_LOGO_ASSETS[family ?? 'generic'];
  return `<img class="${className}" src="${url}" alt="" loading="lazy">`;
}

export function engineLogoHtmlForName(name: string, className = 'engine-logo'): string {
  return engineLogoHtml(engineLogoFamilyForName(name), className);
}
