import { readdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { benchDefaults } from '../stockfish_upgrade_bench.mjs';
import { nativeCell, runReplay, stageEmscripten } from './bench.mjs';
import { benchSource } from './source.mjs';

export default {
  tiers: (context) => {
    const files = [
      ...readdirSync(context.artifacts).filter((file) => file.endsWith('.wasm')),
      ...(context.provenance.manifest?.artifacts ?? []).map((artifact) => basename(artifact.path)).filter((file) => file.endsWith('.wasm')),
    ];
    // The supported release layout has five required flavors. Derive the
    // candidate's versioned stem, never this checkout's historical version.
    const stems = new Set(files.map((file) => file.match(/^(stockfish-\d+(?:\.\d+)*)(?:-(?:lite(?:-single(?:-relaxed)?)?|single))?\.wasm$/)?.[1]).filter(Boolean));
    if (stems.size !== 1) throw new Error('Expected one versioned Stockfish release layout; adapter review required');
    const [stem] = stems;
    for (const suffix of ['', '-single', '-lite', '-lite-single', '-lite-single-relaxed']) files.push(`${stem}${suffix}.wasm`);
    return [...new Set(files)].sort().map((file) =>
      ({ name: file.replace(/^stockfish-/, '').replace(/\.wasm$/, ''), wasm: join(context.artifacts, file),
        js: join(context.artifacts, file.replace(/\.wasm$/, '.js')), flavor: file.includes('lite') ? 'lite' : 'full', requiresSimd: true }));
  },
  prepare: async (context) => { context.defaults = benchDefaults(await benchSource(context, 'src/benchmark.cpp')); },
  native: async (context, tier) => {
    const command = tier.flavor === 'lite' ? context.options.nativeLiteCmd : context.options.nativeCmd;
    return command ? { ...await runReplay(context, nativeCell(command), context.defaults, 10, true), source: '--native-cmd', command } : null;
  },
  bench: async (context, tier) => ({ ...await runReplay(context, stageEmscripten(context, tier, true), context.defaults, 10),
    method: 'full 51-position depth-10 bench replay (Stockfish.js compiles out bench)' }),
  nativeBuild: 'make -C src build ARCH=x86-64-avx2 -j3 (same Stockfish.js ref; LITE_NET=yes for lite; patches/stockfish-js-native-audit.patch; not run automatically)',
  // A commit-message bench uses the upstream default depth, not our depth 10;
  // it is provenance only, never a valid replay reference for this adapter.
  commitBenchCompatible: false,
  parity: 'stockfish',
};
