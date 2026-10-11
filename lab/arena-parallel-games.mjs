import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { berserkVariantByKey, defaultBerserkVariantKey } from '../src/lc0/berserkVariants.ts';
import { gameOutcome } from '../src/lc0/engineBattle.ts';
import { createBerserkEngine } from '../src/lc0/engineProvision.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';

// The product adapters expose depth/movetime, not UCI nodes. Lab-only transport
// interception exercises the same workers with `go nodes` without adding a
// product budget mode. Never run another surface in this document.
const NativeWorker = window.Worker;
const liveWorkers = new Set();
let nodeBudget;
window.Worker = class extends NativeWorker {
  constructor(...args) {
    super(...args);
    liveWorkers.add(this);
  }
  postMessage(message, ...rest) {
    if (nodeBudget !== undefined) {
      if (typeof message === 'string' && /^go\s/.test(message)) message = `go nodes ${nodeBudget}`;
      else if (message?.type === 'command' && /^go\s/.test(message.command)) message = { ...message, command: `go nodes ${nodeBudget}` };
    }
    return super.postMessage(message, ...rest);
  }
  terminate() {
    liveWorkers.delete(this);
    return super.terminate();
  }
};

const openings = [START_FEN, 'rnbqkbnr/ppp2ppp/8/3pp3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3'];
function createPair() {
  return [
    new StockfishEngine({ threads: 1, hashMb: 64, skillLevel: 20 }),
    createBerserkEngine(berserkVariantByKey(defaultBerserkVariantKey()), { surface: 'arena' }),
  ];
}
async function play(pair, index, config, signal) {
  let board = parseFen(openings[index % openings.length]);
  const priorFens = [];
  const trace = [];
  const samples = [];
  for (const engine of pair) {
    engine.setOptions({ threads: 1, hashMb: 64, depth: config.depth ?? 8, movetimeMs: config.limit === 'movetime' ? (config.movetimeMs ?? 500) : undefined });
    await engine.newGame(signal);
  }
  const start = performance.now();
  const cap = config.limit === 'movetime' ? (config.plies ?? 16) : 300;
  let outcome;
  for (let ply = 0; ply < cap; ply++) {
    outcome = gameOutcome(board, priorFens);
    if (outcome) break;
    const seat = board.turn === 'w' ? 0 : 1;
    const engine = pair[seat];
    const fen = boardToFen(board);
    const started = performance.now();
    const uci = await engine.bestMove(fen, signal);
    const elapsedMs = performance.now() - started;
    if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    const info = engine.lastInfo()[0];
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) throw new Error(`Illegal move: ${engine.name} ${uci} ${fen}`);
    samples.push({ engine: seat === 0 ? 'Stockfish' : 'Berserk', nodes: info?.nodes, nps: info?.nps, elapsedMs });
    trace.push({ uci, nodes: info?.nodes, depth: info?.depth, scoreCp: info?.scoreCp, mateIn: info?.mateIn, pv: info?.pvUci });
    priorFens.push(fen);
    board = makeMove(board, move);
  }
  outcome ??= gameOutcome(board, priorFens) ?? { result: '1/2-1/2', reason: 'max plies' };
  return { index, opening: openings[index % openings.length], ...outcome, trace, samples, elapsedMs: performance.now() - start };
}

window.arenaConcurrencyTrial = async (config = {}) => {
  if (window.arenaConcurrencyRunning) throw new Error('Trial already running');
  window.arenaConcurrencyRunning = true;
  const controller = new AbortController();
  window.cancelArenaConcurrencyTrial = () => controller.abort();
  const k = Math.max(1, Math.min(3, Math.floor(config.k ?? 1)));
  const count = config.games ?? k;
  const pairs = [];
  const results = new Array(count);
  const status = document.querySelector('#status');
  status.textContent = `Running K=${k}, ${config.limit ?? 'depth'}`;
  try {
    nodeBudget = config.limit === 'nodes' ? (config.nodes ?? 10000) : undefined;
    for (let lane = 0; lane < k; lane++) pairs.push(createPair());
    // Startup/warmup is excluded from the throughput clock.
    await Promise.all(pairs.flatMap((pair) => pair.map((engine) => engine.prewarm(controller.signal))));
    for (const pair of pairs)
      for (const engine of pair) {
        engine.setOptions({ depth: 1, movetimeMs: undefined });
        await engine.bestMove(START_FEN, controller.signal);
      }
    let next = 0;
    const started = performance.now();
    await Promise.all(
      pairs.map(async (pair) => {
        for (;;) {
          const index = next++;
          if (index >= count) return;
          results[index] = await play(pair, index, config, controller.signal);
        }
      }),
    );
    const report = {
      config,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemory: navigator.deviceMemory,
      isolated: crossOriginIsolated,
      elapsedMs: performance.now() - started,
      games: results,
    };
    window.arenaConcurrencyLast = report;
    document.querySelector('#output').textContent = JSON.stringify(
      { ...report, games: results.map(({ trace, samples, ...game }) => ({ ...game, plies: trace.length })) },
      null,
      2,
    );
    status.textContent = 'Done';
    return report;
  } catch (error) {
    controller.abort();
    status.textContent = `Failed: ${error.message}`;
    throw error;
  } finally {
    for (const pair of pairs) for (const engine of pair) engine.dispose();
    nodeBudget = undefined;
    window.arenaConcurrencyRunning = false;
    window.arenaConcurrencyLiveWorkers = liveWorkers.size;
  }
};
window.arenaConcurrencyMemory = async (k) => {
  if (window.arenaConcurrencyRunning) throw new Error('Trial already running');
  window.arenaConcurrencyRunning = true;
  k = Math.max(1, Math.min(3, Math.floor(k)));
  const pairs = [];
  try {
    for (let lane = 0; lane < k; lane++) pairs.push(createPair());
    await Promise.all(pairs.flatMap((pair) => pair.map((engine) => engine.prewarm())));
    for (const pair of pairs) for (const engine of pair) await engine.bestMove(START_FEN);
    return { k, workers: liveWorkers.size, memory: await performance.measureUserAgentSpecificMemory() };
  } finally {
    for (const pair of pairs) for (const engine of pair) engine.dispose();
    window.arenaConcurrencyRunning = false;
  }
};
