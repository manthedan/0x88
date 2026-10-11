// Research-only probe. Bundle with esbuild into the built static app; no product defaults change.
import { type BoardState, boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { CachedEvaluator, type EvaluationContext, type Evaluator } from '../src/nn/evaluator.ts';
import { SquareformerTvmjsWebgpuEvaluator } from '../src/nn/squareformerTvmjsWebgpuEvaluator.ts';
import { advanceSearchRoot, chooseMove, montyLitePuctPolicy, type Node } from '../src/search/puct.ts';

const lines = [
  'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6b8 d2d4 b8d7',
  'd2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5 g1f3 c7c5 e1g1 b8c6 a2a3 b4c3 b2c3 d5c4 d3c4 d8c7',
  'e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8 e1c1 b8d7',
];
const games = lines.map((line) => {
  const positions = [parseFen(START_FEN)];
  for (const uci of line.split(' ')) {
    const board = positions[positions.length - 1];
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) throw new Error(`Illegal replay move ${uci}`);
    positions.push(makeMove(board, move));
  }
  return positions;
});
const history = (positions: BoardState[]) => positions.slice(0, -1).map(boardToFen).reverse();
const bounded = (context: EvaluationContext = {}): EvaluationContext => ({
  ...context,
  historyFens: context.historyFens?.slice(0, 16),
  historyBoards: context.historyBoards?.slice(0, 16),
});
function memory(root: Node | null | undefined) {
  let nodes = 0;
  let edges = 0;
  let historyRefs = 0;
  let estimateBytes = 0;
  const seen = new Set<Node>();
  const stack = root ? [root] : [];
  while (stack.length) {
    const node = stack.pop()!;
    if (seen.has(node)) continue;
    seen.add(node);
    nodes++;
    edges += node.edges.length;
    historyRefs += node.historyFens.length + node.historyBoards.length;
    // Conservative accounting proxy, not a heap/RSS measurement. Includes maps, FEN text and board arrays.
    estimateBytes += 1024 + node.edges.length * 512 + node.historyFens.reduce((sum, fen) => sum + 64 + fen.length * 2, 0) + node.historyBoards.length * 8;
    for (const edge of node.edges) if (edge.child) stack.push(edge.child);
  }
  return { nodes, edges, historyRefs, estimateBytes };
}

let evaluator: CachedEvaluator;
async function init() {
  if (evaluator) return;
  const meta = await (await fetch('/models/bt4_soap_rem_c19000_final.meta.json')).json();
  const inner = await SquareformerTvmjsWebgpuEvaluator.create(
    '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json',
    meta,
  );
  const windowed: Evaluator = {
    evaluate: (board, context) => inner.evaluate(board, bounded(context)),
    evaluateBatch: (boards, contexts = []) =>
      inner.evaluateBatch(
        boards,
        boards.map((_, i) => bounded(contexts[i])),
      ),
  };
  // The identical cache is used in both arms; only the retained search tree differs.
  evaluator = new CachedEvaluator(windowed, { maxEntries: 4096, includeHistory: true, includeLegalMoves: true });
  await evaluator.evaluate(games[0][0]);
}
async function run(reuse: boolean, visits = 400, yieldEveryMs = 0, plies = 20, sideStride = 2) {
  await init();
  evaluator.clear();
  const rows = [];
  for (let game = 0; game < games.length; game++) {
    let root: Node | null = null;
    for (let ply = 0; ply < Math.min(plies, games[game].length - 1); ply++) {
      const positions = games[game].slice(0, ply + 1);
      const board = positions[positions.length - 1];
      if (reuse && ply > 0 && root) {
        const previous = games[game][ply - 1];
        const played = legalMoves(previous).find((move) => boardToFen(makeMove(previous, move)) === boardToFen(board));
        root = played ? advanceSearchRoot(root, played, board, history(positions)) : null;
      }
      if (ply % sideStride !== 0) continue;
      const inherited = root?.visits ?? 0;
      const before = evaluator.metrics();
      const started = performance.now();
      const result = await chooseMove(board, evaluator, {
        visits,
        batchSize: Math.min(32, visits),
        historyFens: history(positions),
        searchPolicy: montyLitePuctPolicy,
        root: reuse ? root : undefined,
        yieldEveryMs,
      });
      const wallMs = performance.now() - started;
      const after = evaluator.metrics();
      const retained = memory(result.root);
      rows.push({
        game,
        ply,
        fen: boardToFen(board),
        historyFens: history(positions),
        move: result.move ? moveToUci(result.move) : null,
        rootReused: result.stats?.rootReused,
        inherited: result.stats?.rootReused ? inherited : 0,
        completedVisits: result.stats?.completedVisits,
        totalVisits: result.visits,
        neuralMisses: after.misses - before.misses,
        cacheHits: after.hits - before.hits,
        cacheEntries: after.entries,
        wallMs,
        retained,
        heapBytes: (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize,
      });
      root = reuse && retained.estimateBytes <= 16 * 1024 * 1024 ? (result.root ?? null) : null;
    }
  }
  return { reuse, visits, yieldEveryMs, sideStride, rows };
}
async function responsiveness(yieldEveryMs: number, visits = 1600) {
  await init();
  evaluator.clear();
  const options = { visits, batchSize: 32, historyFens: [], searchPolicy: montyLitePuctPolicy };
  await chooseMove(games[0][0], evaluator, options); // all-cache workload on the next identical search
  let handlerMs: number | null = null;
  let abortMs: number | null = null;
  const controller = new AbortController();
  const started = performance.now();
  const timer = setTimeout(() => {
    handlerMs = performance.now() - started;
    controller.abort();
  }, 20);
  try {
    await chooseMove(games[0][0], evaluator, { ...options, yieldEveryMs, signal: controller.signal });
  } catch (error) {
    if (!(error instanceof Error) || error.name !== 'AbortError') throw error;
    abortMs = performance.now() - started;
  }
  clearTimeout(timer);
  return { yieldEveryMs, requestedInputAtMs: 20, handlerMs, abortMs, totalMs: performance.now() - started };
}
async function cachedWall(yieldEveryMs: number, visits = 1600) {
  await init();
  evaluator.clear();
  const options = { visits, batchSize: 32, historyFens: [], searchPolicy: montyLitePuctPolicy };
  await chooseMove(games[0][0], evaluator, options);
  const started = performance.now();
  const result = await chooseMove(games[0][0], evaluator, { ...options, yieldEveryMs });
  return { yieldEveryMs, wallMs: performance.now() - started, misses: result.stats?.neuralEvalMisses, move: result.move ? moveToUci(result.move) : null };
}
Object.assign(globalThis, { centipawnReuseProbe: { init, run, responsiveness, cachedWall } });
