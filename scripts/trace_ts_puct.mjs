import fs from 'node:fs';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';

const artifactPath = process.argv[2] ?? 'artifacts/student_distill_benchmark.json';
const fen = process.argv[3] ?? START_FEN;
const sims = Number(process.argv[4] ?? 8);
const evaluator = new StudentEvaluator(JSON.parse(fs.readFileSync(artifactPath, 'utf8')));
const board = parseFen(fen);
const moves = legalMoves(board);
const evaln = await evaluator.evaluate(board);
const value = (wdl) => wdl[0] - wdl[2];
const raw = moves.map((move) => Math.max(0, evaln.policy.get(moveToActionId(move)) ?? 0));
const total = raw.reduce((a, b) => a + b, 0);
const fallback = moves.length ? 1 / moves.length : 0;
const edges = moves.map((move, i) => ({ move, prior: total > 0 ? raw[i] / total : fallback, visits: 0, valueSum: 0 }));
const q = (edge) => edge.visits ? -edge.valueSum / edge.visits : 0;
console.log(`root_value=${value(evaln.wdl).toFixed(9)}`);
edges.slice(0, 8).forEach((e, i) => console.log(`PRIOR ${i} ${moveToUci(e.move)} ${e.prior.toFixed(9)}`));
for (let sim = 0; sim < sims; sim++) {
  const parentVisits = edges.reduce((sum, edge) => sum + edge.visits, 0);
  const sqrtParent = Math.sqrt(parentVisits + 1);
  let best = edges[0], bestScore = -Infinity;
  for (const edge of edges) {
    const score = q(edge) + 1.5 * edge.prior * sqrtParent / (1 + edge.visits);
    if (score > bestScore) { best = edge; bestScore = score; }
  }
  const childValue = value((await evaluator.evaluate(makeMove(board, best.move))).wdl);
  const before = best.visits;
  const qBefore = before ? -best.valueSum / before : 0;
  best.visits += 1;
  best.valueSum += childValue;
  console.log(`TRACE sim=${sim} move=${moveToUci(best.move)} prior=${best.prior.toFixed(9)} q_before=${qBefore.toFixed(9)} score=${bestScore.toFixed(9)} visits_before=${before} child_value=${childValue.toFixed(9)} q_after=${q(best).toFixed(9)}`);
}
