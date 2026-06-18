type ImportMetaWithEnv = ImportMeta & { env?: Record<string, string | undefined> };

const env = (import.meta as ImportMetaWithEnv).env ?? {};

function cleanBase(raw: string | undefined): string | undefined {
  if (!raw) return undefined;
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  try {
    const url = new URL(trimmed);
    if (url.protocol !== 'https:' && url.protocol !== 'http:') return undefined;
    return url.href.replace(/\/+$/, '');
  } catch {
    return undefined;
  }
}

function queryAssetBase(): string | undefined {
  try {
    const params = new URLSearchParams(location.search);
    return cleanBase(params.get('assetBase') ?? params.get('modelBase') ?? undefined);
  } catch {
    return undefined;
  }
}

function configuredAssetBase(): string | undefined {
  const globalBase = (globalThis as { LC0_BROWSER_ASSET_BASE_URL?: string }).LC0_BROWSER_ASSET_BASE_URL;
  return queryAssetBase()
    ?? cleanBase(globalBase)
    ?? cleanBase(env.VITE_LC0_BROWSER_ASSET_BASE_URL)
    ?? cleanBase(env.VITE_LC0_MODEL_BASE_URL);
}

/**
 * Resolve public model/runtime assets. Local development keeps same-origin
 * /models paths; hosted builds can point the same paths at an R2-backed origin.
 */
export function resolvePublicAssetUrl(pathOrUrl: string): string {
  try {
    const url = new URL(pathOrUrl);
    if (url.protocol === 'https:' || url.protocol === 'http:') return url.href;
  } catch {
    // Relative/same-origin path handled below.
  }
  const base = configuredAssetBase();
  if (!base || !pathOrUrl.startsWith('/models/')) return pathOrUrl;
  return `${base}${pathOrUrl}`;
}
