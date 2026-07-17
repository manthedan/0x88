import type { EngineFamily } from './engineCatalog.ts';

const VARIANT_MENU_LABELS: Partial<Record<EngineFamily, Record<string, string>>> = {
  reckless: {
    full: 'Scalar',
    simd: 'SIMD',
    'relaxed-simd': 'Relaxed SIMD',
    lite: 'Lite',
    'wasi-simd-external': 'SIMD · external NNUE',
    'browser-api': 'Browser API',
    'browser-api-simd': 'Browser API · SIMD',
    'browser-api-simd-external': 'Browser API · SIMD · external NNUE',
    custom: 'Custom',
  },
  viridithas: {
    default: 'Scalar',
    simd: 'SIMD',
    'relaxed-simd': 'Relaxed SIMD',
    custom: 'Custom',
  },
  berserk: {
    emscripten: 'Scalar',
    'emscripten-simd': 'SIMD',
    'emscripten-relaxed': 'Relaxed SIMD',
    default: 'Scalar WASI · planned',
    simd: 'SIMD WASI · planned',
    custom: 'Custom',
  },
  plentychess: {
    emscripten: 'Standard',
    'emscripten-sse41': 'SSE4.1',
    'emscripten-relaxed': 'Relaxed SIMD',
    custom: 'Custom',
  },
  stormphrax: {
    emscripten: '8',
    'emscripten-relaxed': '8 · Relaxed SIMD',
    custom: 'Custom',
  },
};

/** Compact labels for a variant dropdown whose adjacent control already names the family. */
export function engineVariantMenuLabel(family: EngineFamily, variant: string, fallback: string): string {
  return VARIANT_MENU_LABELS[family]?.[variant] ?? fallback;
}
