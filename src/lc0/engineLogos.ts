import type { EngineFamily } from './engineCatalog.ts';

export type EngineLogoFamily = Extract<EngineFamily, 'lc0' | 'sf' | 'reckless' | 'viridithas' | 'berserk'>;

interface EngineLogoAsset {
  family: EngineLogoFamily;
  url: string;
}

const ENGINE_LOGO_ASSETS: Record<EngineLogoFamily, EngineLogoAsset> = {
  lc0: { family: 'lc0', url: '/engine-logos/lc0.svg' },
  sf: { family: 'sf', url: '/engine-logos/stockfish.png' },
  reckless: { family: 'reckless', url: '/engine-logos/reckless.png' },
  viridithas: { family: 'viridithas', url: '/engine-logos/viridithas.png' },
  berserk: { family: 'berserk', url: '/engine-logos/berserk.jpg' },
};

const availableEngineLogos = new Set<EngineLogoFamily>();
const pendingProbeCallbacks = new Set<() => void>();
let probed = false;
let probing: Promise<void> | null = null;

export function engineLogoFamilyForName(name: string): EngineLogoFamily | undefined {
  const n = name.toLowerCase();
  if (n.includes('tiny leela')) return undefined;
  if (n.includes('bt4') || n.includes('lc0') || n.includes('leela')) return 'lc0';
  if (n.includes('reckless')) return 'reckless';
  if (n.includes('viridithas')) return 'viridithas';
  if (n.includes('berserk')) return 'berserk';
  if (n.includes('stockfish') || /\bsf\b/.test(n)) return 'sf';
  return undefined;
}

export function engineLogoFamilyForEngineFamily(family: EngineFamily): EngineLogoFamily | undefined {
  if (family === 'lc0') return 'lc0';
  if (family === 'sf') return 'sf';
  if (family === 'reckless' || family === 'viridithas' || family === 'berserk') return family;
  return undefined;
}

export function engineLogoUrl(family: EngineLogoFamily | undefined): string | undefined {
  return family ? ENGINE_LOGO_ASSETS[family]?.url : undefined;
}

export function engineLogoHtml(family: EngineLogoFamily | undefined, className = 'engine-logo'): string {
  if (!family) return '';
  const url = engineLogoUrl(family);
  return url && availableEngineLogos.has(family) ? `<img class="${className}" src="${url}" alt="">` : '';
}

export function engineLogoHtmlForName(name: string, className = 'engine-logo'): string {
  return engineLogoHtml(engineLogoFamilyForName(name), className);
}

function notifyProbeCallbacks(): void {
  const callbacks = [...pendingProbeCallbacks];
  pendingProbeCallbacks.clear();
  if (!availableEngineLogos.size) return;
  for (const callback of callbacks) {
    try {
      callback();
    } catch {
      // A logo-triggered re-render is best-effort; do not fail the shared probe.
    }
  }
}

export async function probeEngineLogos(onChange?: () => void): Promise<void> {
  if (onChange) pendingProbeCallbacks.add(onChange);
  if (probed) {
    notifyProbeCallbacks();
    return;
  }
  probing ??= Promise.all(Object.values(ENGINE_LOGO_ASSETS).map(async ({ family, url }) => {
    try {
      const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      if (response.ok && (response.headers.get('content-type') ?? '').startsWith('image/')) availableEngineLogos.add(family);
    } catch {
      // Logo assets are decorative; missing files fall back to text-only labels.
    }
  })).then(() => {
    probed = true;
    notifyProbeCallbacks();
  });
  await probing;
}
