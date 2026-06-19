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

function queryAssetBase(allowModelBaseAlias: boolean): string | undefined {
  try {
    const params = new URLSearchParams(location.search);
    return cleanBase(params.get('assetBase') ?? (allowModelBaseAlias ? params.get('modelBase') : undefined) ?? undefined);
  } catch {
    return undefined;
  }
}

function trustedConfiguredAssetBase(allowModelBaseAlias: boolean): string | undefined {
  const globals = globalThis as { LC0_BROWSER_ASSET_BASE_URL?: string; TINY_LEELA_ASSET_BASE_URL?: string };
  return cleanBase(globals.LC0_BROWSER_ASSET_BASE_URL)
    ?? cleanBase(globals.TINY_LEELA_ASSET_BASE_URL)
    ?? cleanBase(env.VITE_LC0_BROWSER_ASSET_BASE_URL)
    ?? cleanBase(env.VITE_TINY_LEELA_ASSET_BASE_URL)
    ?? (allowModelBaseAlias ? cleanBase(env.VITE_LC0_MODEL_BASE_URL) : undefined);
}

function configuredAssetBase(allowModelBaseAlias: boolean): string | undefined {
  return queryAssetBase(allowModelBaseAlias) ?? trustedConfiguredAssetBase(allowModelBaseAlias);
}

const R2_RESOLVED_PREFIXES = [
  '/models/',
  '/berserk/',
  '/plentychess/',
  '/viridithas/',
  '/reckless/',
  '/runtimes/squareformer-tvm-hybrid/',
];

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
  const isModelAsset = pathOrUrl.startsWith('/models/');
  const base = configuredAssetBase(isModelAsset);
  if (!base || !R2_RESOLVED_PREFIXES.some((prefix) => pathOrUrl.startsWith(prefix))) return pathOrUrl;
  return `${base}${pathOrUrl}`;
}

export function isTrustedExecutableAssetUrl(pathOrUrl: string): boolean {
  try {
    const base = typeof location !== 'undefined' ? location.href : 'http://localhost';
    const url = new URL(pathOrUrl, base);
    if (typeof location !== 'undefined' && url.origin === location.origin) return true;
    const trustedBase = trustedConfiguredAssetBase(false);
    return trustedBase ? url.origin === new URL(trustedBase).origin : false;
  } catch {
    return false;
  }
}
