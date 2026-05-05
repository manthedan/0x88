import type { BoardState } from '../chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import type { Evaluator } from '../nn/evaluator.ts';

export interface SearchResult { move: Move | null; visits: number; value: number; }
export interface SearchOptions { visits?: number; cpuct?: number; }

interface Edge {
  move: Move;
  prior: number;
  child: Node | null;
  visits: number;
  valueSum: number; // from the child side-to-move perspective
}

interface Node {
  board: BoardState;
  expanded: boolean;
  terminalValue: number | null;
  edges: Edge[];
}

function valueFromWdl(wdl: [number, number, number]): number {
  return wdl[0] - wdl[2];
}

function edgeQForParent(edge: Edge): number {
  return edge.visits ? -edge.valueSum / edge.visits : 0;
}

async function expand(node: Node, evaluator: Evaluator): Promise<number> {
  const moves = legalMoves(node.board);
  if (!moves.length) {
    node.expanded = true;
    // No legal moves: checkmate is loss for the side to move, stalemate is draw.
    node.terminalValue = inCheck(node.board) ? -1 : 0;
    node.edges = [];
    return node.terminalValue;
  }
  const evaln = await evaluator.evaluate(node.board);
  const raw = moves.map((move) => Math.max(0, evaln.policy.get(moveToActionId(move)) ?? 0));
  const total = raw.reduce((a, b) => a + b, 0);
  const fallback = 1 / moves.length;
  node.edges = moves.map((move, i) => ({
    move,
    prior: total > 0 ? raw[i] / total : fallback,
    child: null,
    visits: 0,
    valueSum: 0,
  }));
  node.expanded = true;
  node.terminalValue = null;
  return valueFromWdl(evaln.wdl);
}

async function simulate(node: Node, evaluator: Evaluator, cpuct: number): Promise<number> {
  if (!node.expanded) return expand(node, evaluator);
  if (node.terminalValue !== null) return node.terminalValue;
  const parentVisits = node.edges.reduce((sum, edge) => sum + edge.visits, 0);
  const sqrtParent = Math.sqrt(parentVisits + 1);
  let best = node.edges[0];
  let bestScore = -Infinity;
  for (const edge of node.edges) {
    const q = edgeQForParent(edge);
    const u = cpuct * edge.prior * sqrtParent / (1 + edge.visits);
    const score = q + u;
    if (score > bestScore) { best = edge; bestScore = score; }
  }
  if (!best.child) best.child = { board: makeMove(node.board, best.move), expanded: false, terminalValue: null, edges: [] };
  const childValue = await simulate(best.child, evaluator, cpuct);
  best.visits += 1;
  best.valueSum += childValue;
  return -childValue;
}

export async function chooseMove(board: BoardState, evaluator: Evaluator, options: SearchOptions = {}): Promise<SearchResult> {
  const visits = Math.max(1, Math.floor(options.visits ?? 8));
  const cpuct = options.cpuct ?? 1.5;
  const root: Node = { board, expanded: false, terminalValue: null, edges: [] };
  const rootValue = await expand(root, evaluator);
  if (!root.edges.length) return { move: null, visits: 0, value: rootValue };
  for (let i = 0; i < visits; i++) await simulate(root, evaluator, cpuct);
  const best = root.edges.reduce((a, b) => b.visits > a.visits ? b : a);
  return { move: best.move, visits: root.edges.reduce((sum, edge) => sum + edge.visits, 0), value: edgeQForParent(best) };
}
