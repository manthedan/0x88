type ImportMetaWithEnv = ImportMeta & { env?: Record<string, string | boolean | undefined> };

const env = (import.meta as ImportMetaWithEnv).env ?? {};

function cleanBase(raw: string | boolean | undefined): string | undefined {
  if (typeof raw !== 'string') return undefined;
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

/** Loopback dev servers, static builds and lab pages; also usable in workers. */
export function isLocalDevelopmentOrigin(): boolean {
  if (typeof location === 'undefined') return true;
  try {
    const hostname = location.hostname || new URL(location.origin).hostname;
    return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(hostname) || hostname.endsWith('.localhost');
  } catch {
    return false;
  }
}

export function assetPathname(raw: string): string {
  try {
    return new URL(raw, typeof location === 'undefined' ? 'http://localhost/' : location.href).pathname;
  } catch {
    return raw;
  }
}

/** Dev-server query overrides also honour the build's DEV flag. */
function isLocalDevelopment(): boolean {
  return env.DEV === true || isLocalDevelopmentOrigin();
}

/**
 * ?assetBase= / ?modelBase= decide where WASM, NNUE, ONNX and model manifests
 * are fetched from, and a manifest from an arbitrary origin brings its own
 * sha256, so verification would pass for whatever it serves. On a deployed
 * site an override is therefore honoured only when it names the build-time
 * asset host or this page's own origin; any origin is allowed locally.
 */
function overrideAllowed(base: string, allowModelBaseAlias: boolean): boolean {
  if (isLocalDevelopment()) return true;
  try {
    const origin = new URL(base).origin;
    if (origin === location.origin) return true;
    const trusted = trustedConfiguredAssetBase(allowModelBaseAlias);
    return trusted !== undefined && origin === new URL(trusted).origin;
  } catch {
    return false;
  }
}

function queryAssetBase(allowModelBaseAlias: boolean): string | undefined {
  try {
    const params = new URLSearchParams(location.search);
    const base = cleanBase(params.get('assetBase') ?? (allowModelBaseAlias ? params.get('modelBase') : undefined) ?? undefined);
    return base && overrideAllowed(base, allowModelBaseAlias) ? base : undefined;
  } catch {
    return undefined;
  }
}

function trustedConfiguredAssetBase(allowModelBaseAlias: boolean): string | undefined {
  const globals = globalThis as { LC0_BROWSER_ASSET_BASE_URL?: string; TINY_LEELA_ASSET_BASE_URL?: string };
  return (
    cleanBase(globals.LC0_BROWSER_ASSET_BASE_URL) ??
    cleanBase(globals.TINY_LEELA_ASSET_BASE_URL) ??
    cleanBase(env.VITE_LC0_BROWSER_ASSET_BASE_URL) ??
    cleanBase(env.VITE_TINY_LEELA_ASSET_BASE_URL) ??
    (allowModelBaseAlias ? cleanBase(env.VITE_LC0_MODEL_BASE_URL) : undefined)
  );
}

function configuredAssetBase(allowModelBaseAlias: boolean): string | undefined {
  return queryAssetBase(allowModelBaseAlias) ?? trustedConfiguredAssetBase(allowModelBaseAlias);
}

const R2_RESOLVED_PREFIXES = [
  '/artifacts/sha256/',
  '/models/',
  '/berserk/',
  '/plentychess/',
  '/stormphrax/',
  '/viridithas/',
  '/reckless/',
  '/stockfish/',
  '/runtimes/squareformer-tvm-hybrid/',
  '/runtimes/centipawn-tvmjs-webgpu/',
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

/** Apply the same trust boundary to individual query-selected assets as bases. */
export function resolveAssetOverride(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    const base = typeof location === 'undefined' ? 'http://localhost/' : location.href;
    const url = new URL(raw, base);
    if (url.protocol !== 'https:' && url.protocol !== 'http:') return undefined;
    if (typeof location === 'undefined' ? url.origin !== 'http://localhost' : !overrideAllowed(url.href, false)) return undefined;
    // A double-slash pathname becomes a different host if shortened to a
    // relative URL. The parser also normalises backslashes into these paths.
    return url.origin === new URL(base).origin && !url.pathname.startsWith('//') ? `${url.pathname}${url.search}` : url.href;
  } catch {
    return undefined;
  }
}

export function isTrustedExecutableAssetUrl(pathOrUrl: string): boolean {
  try {
    const base = typeof location !== 'undefined' ? location.href : 'http://localhost';
    const url = new URL(pathOrUrl, base);
    if (typeof location !== 'undefined' && url.origin === location.origin) return true;
    if (isLocalDevelopment() && resolveAssetOverride(pathOrUrl)) return true;
    const trustedBase = trustedConfiguredAssetBase(false);
    return trustedBase ? url.origin === new URL(trustedBase).origin : false;
  } catch {
    return false;
  }
}
