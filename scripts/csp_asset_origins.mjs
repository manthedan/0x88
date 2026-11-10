// Mirror assetUrls.ts's build-time base precedence (runtime globals are not
// build configuration). Stockfish, Berserk, PlentyChess and Stormphrax load JS
// from the executable base via blob workers/importScripts. The Squareformer
// TVMJS loader also inserts a trusted script from that base. Reckless and
// Viridithas use bundled workers and fetch WASM/NNUE; model-only bases and the
// artifact channel serve data. /ort/ glue and bundled workers stay same-origin.
const DEFAULT_ASSET_ORIGIN = 'https://assets.0x88.app';

function configuredOrigin(env, settings) {
  for (const setting of settings) {
    try {
      const url = new URL(env[setting]?.trim());
      if (url.protocol === 'https:' || url.protocol === 'http:') return { origin: url.origin, setting };
    } catch {
      // Like cleanBase in assetUrls.ts, ignore empty or invalid bases.
    }
  }
}

export function configuredAssetOrigins(env) {
  const executable = configuredOrigin(env, ['VITE_LC0_BROWSER_ASSET_BASE_URL', 'VITE_TINY_LEELA_ASSET_BASE_URL']);
  const fallback = { origin: DEFAULT_ASSET_ORIGIN, setting: 'VITE_LC0_BROWSER_ASSET_BASE_URL' };
  return {
    executable: executable ?? fallback,
    model: executable ?? configuredOrigin(env, ['VITE_LC0_MODEL_BASE_URL']) ?? fallback,
  };
}

// Keep the shipped policy intact and add the configured executable origin.
export function executableAssetSources(env) {
  return [...new Set([DEFAULT_ASSET_ORIGIN, configuredAssetOrigins(env).executable.origin])];
}
