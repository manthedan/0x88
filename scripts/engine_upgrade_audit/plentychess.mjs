import { nativeCell, runReplay, stageEmscripten } from './bench.mjs';
import { benchSource } from './source.mjs';
import { emscriptenTiers } from './tiers.mjs';

export function parsePlentyChessFens(source) {
  const block = source.match(/BENCH_POSITIONS\s*=\s*\{([\s\S]*?)\};/)?.[1];
  if (!block) throw new Error('Upstream BENCH_POSITIONS changed');
  const fens = [...block.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
  if (fens.length !== 46) throw new Error(`Unexpected PlentyChess bench size: ${fens.length} (expected 46; Chess960 starts at zero-based 44)`);
  return fens;
}

export function plentyChessScript(fens) {
  return ['uci', 'setoption name Threads value 1', 'setoption name Hash value 16', 'ucinewgame',
    ...fens.flatMap((fen, index) => [...(index === 44 ? ['setoption name UCI_Chess960 value true'] : []), `position fen ${fen}`, 'go depth 13'])];
}

export function plentyChessDefaults(fens) {
  // The shared bench-replay driver supplies the same setup and readiness
  // barriers. Keep options in the corpus sequence, not resets per position.
  return fens.flatMap((fen, index) => [...(index === 44 ? ['setoption name UCI_Chess960 value true'] : []), fen]);
}

export default {
  tiers: (context) => emscriptenTiers(context, ['', 'sse41', 'relaxed-simd128']),
  prepare: async (context) => { context.defaults = plentyChessDefaults(parsePlentyChessFens(await benchSource(context, 'src/bench.h'))); },
  native: async (context) => context.options.nativeCmd ?
    { ...await runReplay(context, nativeCell(context.options.nativeCmd), context.defaults, 13), source: '--native-cmd', command: context.options.nativeCmd } : null,
  bench: async (context, tier) => ({ ...await runReplay(context, stageEmscripten(context, tier), context.defaults, 13),
    method: 'full 46-position depth-13 replay (built-in bench deadlocks); Chess960 before index 44' }),
  nativeBuild: 'make -j3 (locked upstream source; process the matching net first; not run automatically)',
  // Replay includes the source's "moves ..." suffixes. Upstream's built-in
  // bench only constructs a board from the FEN, so its commit signature is a
  // different protocol and must not be used as this replay's numeric oracle.
  commitBenchCompatible: false,
  parity: 'emscripten',
};
