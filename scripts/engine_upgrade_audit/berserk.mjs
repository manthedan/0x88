import { runEmscriptenBuiltin, runNativeBuiltin } from './bench.mjs';
import { emscriptenTiers } from './tiers.mjs';

export default {
  tiers: (context) => emscriptenTiers(context, ['', 'simd128', 'relaxed-simd128']),
  native: runNativeBuiltin,
  bench: runEmscriptenBuiltin,
  nativeBuild: 'make -C src -j3 (locked source, matching network; not run automatically)',
  parity: 'emscripten',
};
