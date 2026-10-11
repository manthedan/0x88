import { runEmscriptenBuiltin, runNativeBuiltin } from './bench.mjs';
import { emscriptenTiers } from './tiers.mjs';

export default {
  tiers: (context) => emscriptenTiers(context, ['', 'relaxed-simd128']).map((tier) => ({ ...tier, requiresSimd: true })),
  native: runNativeBuiltin,
  bench: runEmscriptenBuiltin,
  nativeBuild: 'make -j3 (locked source, matching undertown.nnue; not run automatically)',
  parity: 'emscripten',
};
