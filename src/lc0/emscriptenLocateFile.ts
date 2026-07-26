/**
 * Shared `Module.locateFile` resolution for the Emscripten engine workers
 * (Berserk, PlentyChess, Stormphrax).
 *
 * This is load-bearing for the shared-preload design. Emscripten bakes the
 * package name of the build that produced it into each glue file, so the
 * relaxed-SIMD glue asks for `<engine>-emscripten-relaxed-simd128.data` even
 * though only the canonical `<engine>-emscripten.data` is published (the tiers
 * emit byte-identical preloads, so exactly one copy ships). `locateFile` is the
 * hook that redirects that request to the canonical URL. Drop the `.data`
 * branch and every non-default SIMD tier 404s at load time while the URLs in
 * the variant registry still look correct — which is precisely the regression
 * this module exists to make testable.
 */

export interface EmscriptenAssetUrls {
  /** Glue URL; the base for anything not explicitly redirected. */
  jsUrl: string;
  /** Per-variant wasm sidecar, when the caller pins one. */
  wasmUrl?: string | null;
  /** Canonical shared preload package, when the caller pins one. */
  dataUrl?: string | null;
}

/**
 * Resolve one Emscripten-requested filename to a concrete URL. Extension-based
 * rather than name-based on purpose: the requested name carries the *building*
 * variant's basename, which is exactly what must not be honoured.
 */
export function resolveEmscriptenAssetUrl(file: string, urls: EmscriptenAssetUrls): string {
  const name = String(file);
  if (urls.wasmUrl && name.endsWith('.wasm')) return urls.wasmUrl;
  if (urls.dataUrl && name.endsWith('.data')) return urls.dataUrl;
  return new URL(name, urls.jsUrl).href;
}
