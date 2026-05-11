import { boardToFen, type BoardState } from '../chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import type { Evaluation, Evaluator } from '../nn/evaluator.ts';

export interface SearchPolicyEntry { move: Move; visits: number; prior: number; q: number; probability: number; }
export interface SearchResult { move: Move | null; visits: number; value: number; policy: SearchPolicyEntry[]; stats?: SearchStats; }
export interface SearchOptions { visits?: number; cpuct?: number; fpu?: number; temperature?: number; historyFens?: string[]; batchSize?: number; searchPolicy?: SearchPolicy; avWeight?: number; rankWeight?: number; regretWeight?: number; riskWeight?: number; uncertaintyWeight?: number; }
export interface GumbelRootOptions { candidateCount?: number; seed?: number; gumbelScale?: number; qWeight?: number; priorWeight?: number; visitPenalty?: number; }

export interface SearchStats {
  requestedVisits: number;
  completedVisits: number;
  expansions: number;
  terminalHits: number;
  evalCalls: number;
  batchEvalCalls: number;
  maxEvalBatch: number;
}

export interface Edge {
  move: Move;
  prior: number;
  child: Node | null;
  visits: number;
  valueSum: number; // from the child side-to-move perspective
  virtualVisits: number;
  // Future edge-local model/search signals. These are intentionally optional so
  // classic PUCT stays unchanged while AV/risk/uncertainty policies can plug in.
  actionValuePrior?: number;
  rankScore?: number;
  regret?: number;
  risk?: number;
  uncertainty?: number;
}

export interface Node {
  board: BoardState;
  historyFens: string[];
  expanded: boolean;
  terminalValue: number | null;
  edges: Edge[];
  isRoot?: boolean;
}

export interface SearchPolicyContext {
  cpuct: number;
  fpu: number;
  temperature: number;
  /** Weights for optional edge-local aux signals in experimental search policies. */
  avWeight: number;
  rankWeight: number;
  regretWeight: number;
  riskWeight: number;
  uncertaintyWeight: number;
}

export interface SearchPolicy {
  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number;
  backup(path: Edge[], leafValue: number, context: SearchPolicyContext): void;
  rootPolicy(edges: Edge[], context: SearchPolicyContext, node?: Node): SearchPolicyEntry[];
  chooseFinalMove(entries: SearchPolicyEntry[], context: SearchPolicyContext): SearchPolicyEntry | null;
}

function valueFromWdl(wdl: [number, number, number]): number {
  return wdl[0] - wdl[2];
}

export function edgeQForParent(edge: Edge, fpu = 0): number {
  return edge.visits ? -edge.valueSum / edge.visits : fpu;
}

export function edgeSelectVisits(edge: Edge): number {
  return edge.visits + edge.virtualVisits;
}

export class ClassicPUCTPolicy implements SearchPolicy {
  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number {
    const parentVisits = node.edges.reduce((sum, e) => sum + edgeSelectVisits(e), 0);
    const sqrtParent = Math.sqrt(parentVisits + 1);
    const q = edgeQForParent(edge, context.fpu);
    const sv = edgeSelectVisits(edge);
    const u = context.cpuct * edge.prior * sqrtParent / (1 + sv);
    return q + u;
  }

  backup(path: Edge[], leafValue: number): void {
    let value = leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      const edge = path[i];
      edge.virtualVisits = Math.max(0, edge.virtualVisits - 1);
      edge.visits += 1;
      edge.valueSum += value;
      value = -value;
    }
  }

  rootPolicy(edges: Edge[], context: SearchPolicyContext): SearchPolicyEntry[] {
    if (!edges.length) return [];
    const tau = Math.max(0, context.temperature);
    if (tau === 0) {
      const better = (a: Edge, b: Edge) => {
        if (b.visits !== a.visits) return b.visits > a.visits;
        const aq = edgeQForParent(a, context.fpu), bq = edgeQForParent(b, context.fpu);
        if (bq !== aq) return bq > aq;
        return b.prior > a.prior;
      };
      const best = edges.reduce((a, b) => better(a, b) ? b : a);
      return edges.map((edge) => ({ move: edge.move, visits: edge.visits, prior: edge.prior, q: edgeQForParent(edge, context.fpu), probability: edge === best ? 1 : 0 }));
    }
    const weights = edges.map((edge) => Math.pow(Math.max(edge.visits, 1e-9), 1 / tau));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    return edges.map((edge, i) => ({ move: edge.move, visits: edge.visits, prior: edge.prior, q: edgeQForParent(edge, context.fpu), probability: weights[i] / total }));
  }

  chooseFinalMove(entries: SearchPolicyEntry[]): SearchPolicyEntry | null {
    if (!entries.length) return null;
    return entries.reduce((a, b) => b.probability > a.probability ? b : a);
  }
}

export class ActionValuePUCTPolicy extends ClassicPUCTPolicy {
  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number {
    const base = super.scoreEdge(node, edge, context);
    const av = edge.actionValuePrior ?? 0;
    // AV is a parent-perspective prior. It guides unvisited/low-visit moves and
    // decays as backed-up search evidence accumulates.
    return base + context.avWeight * av / (1 + edgeSelectVisits(edge));
  }
}

export class AuxPUCTPolicy extends ClassicPUCTPolicy {
  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number {
    const base = super.scoreEdge(node, edge, context);
    const sv = 1 + edgeSelectVisits(edge);
    return base
      + context.avWeight * (edge.actionValuePrior ?? 0) / sv
      + context.rankWeight * (edge.rankScore ?? 0) / sv
      - context.regretWeight * (edge.regret ?? 0) / sv
      - context.riskWeight * (edge.risk ?? 0) / sv
      + context.uncertaintyWeight * (edge.uncertainty ?? 0) / sv;
  }
}

function splitmix32(seed: number): () => number {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x9e3779b9) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    z = (z ^ (z >>> 15)) >>> 0;
    return (z + 0.5) / 0x100000000;
  };
}

function gumbel01(rng: () => number): number {
  const u = Math.min(1 - 1e-12, Math.max(1e-12, rng()));
  return -Math.log(-Math.log(u));
}

interface GumbelRootState { candidates: Set<Edge>; noise: Map<Edge, number>; }

export class GumbelRootPolicy extends ClassicPUCTPolicy {
  private options: Required<GumbelRootOptions>;
  private rng: () => number;
  private states = new WeakMap<Node, GumbelRootState>();

  constructor(options: GumbelRootOptions = {}) {
    super();
    this.options = {
      candidateCount: Math.max(1, Math.floor(options.candidateCount ?? 16)),
      seed: Math.floor(options.seed ?? 1),
      gumbelScale: options.gumbelScale ?? 1,
      qWeight: options.qWeight ?? 1,
      priorWeight: options.priorWeight ?? 1,
      visitPenalty: options.visitPenalty ?? 0.15,
    };
    this.rng = splitmix32(this.options.seed);
  }

  private rootState(node: Node): GumbelRootState {
    let state = this.states.get(node);
    if (state) return state;
    const scored = node.edges.map((edge) => {
      const noise = gumbel01(this.rng) * this.options.gumbelScale;
      return { edge, noise, score: Math.log(Math.max(edge.prior, 1e-12)) + noise };
    }).sort((a, b) => b.score - a.score);
    const keep = new Set(scored.slice(0, Math.min(this.options.candidateCount, scored.length)).map((x) => x.edge));
    const noise = new Map(scored.map((x) => [x.edge, x.noise]));
    state = { candidates: keep, noise };
    this.states.set(node, state);
    return state;
  }

  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number {
    if (!node.isRoot) return super.scoreEdge(node, edge, context);
    const state = this.rootState(node);
    if (!state.candidates.has(edge)) return -Infinity;
    const parentVisits = node.edges.reduce((sum, e) => sum + edgeSelectVisits(e), 0);
    const q = edgeQForParent(edge, context.fpu);
    const sv = edgeSelectVisits(edge);
    const puct = context.cpuct * edge.prior * Math.sqrt(parentVisits + 1) / (1 + sv);
    const g = state.noise.get(edge) ?? 0;
    const logPrior = Math.log(Math.max(edge.prior, 1e-12));
    // Root-only Gumbel demo policy: sample a policy-prior candidate set with
    // log(P)+Gumbel, then spend low visits on candidates using a conservative
    // sequential-halving-like score. Non-root selection remains classic PUCT.
    return this.options.qWeight * q
      + this.options.priorWeight * (logPrior + g) / (1 + sv)
      + puct
      - this.options.visitPenalty * sv;
  }

}

export const classicPuctPolicy = new ClassicPUCTPolicy();
export const actionValuePuctPolicy = new ActionValuePUCTPolicy();
export const auxPuctPolicy = new AuxPUCTPolicy();

function makeStats(visits: number): SearchStats {
  return { requestedVisits: visits, completedVisits: 0, expansions: 0, terminalHits: 0, evalCalls: 0, batchEvalCalls: 0, maxEvalBatch: 0 };
}

function makeChild(parent: Node, move: Move): Node {
  return { board: makeMove(parent.board, move), historyFens: [boardToFen(parent.board), ...parent.historyFens], expanded: false, terminalValue: null, edges: [], isRoot: false };
}

function selectBestEdge(node: Node, searchPolicy: SearchPolicy, context: SearchPolicyContext): Edge {
  let best = node.edges[0];
  let bestScore = -Infinity;
  for (const edge of node.edges) {
    const score = searchPolicy.scoreEdge(node, edge, context);
    if (score > bestScore) { best = edge; bestScore = score; }
  }
  return best;
}

async function expand(node: Node, evaluator: Evaluator, stats?: SearchStats): Promise<number> {
  const moves = legalMoves(node.board);
  if (!moves.length) {
    node.expanded = true;
    // No legal moves: checkmate is loss for the side to move, stalemate is draw.
    node.terminalValue = inCheck(node.board) ? -1 : 0;
    node.edges = [];
    if (stats) { stats.expansions += 1; stats.terminalHits += 1; }
    return node.terminalValue;
  }
  if (stats) stats.evalCalls += 1;
  const evaln = await evaluator.evaluate(node.board, { historyFens: node.historyFens, legalMoves: moves });
  const value = finishExpansion(node, moves, evaln);
  if (stats) stats.expansions += 1;
  return value;
}

async function simulate(node: Node, evaluator: Evaluator, searchPolicy: SearchPolicy, context: SearchPolicyContext, stats: SearchStats): Promise<number> {
  if (!node.expanded) return expand(node, evaluator, stats);
  if (node.terminalValue !== null) { stats.terminalHits += 1; return node.terminalValue; }
  const best = selectBestEdge(node, searchPolicy, context);
  if (!best.child) best.child = makeChild(node, best.move);
  const childValue = await simulate(best.child, evaluator, searchPolicy, context, stats);
  searchPolicy.backup([best], childValue, context);
  return -childValue;
}

interface SelectedLeaf {
  node: Node;
  path: Edge[];
}

type PreparedLeaf =
  | { kind: 'terminal'; sel: SelectedLeaf; value: number }
  | { kind: 'eval'; sel: SelectedLeaf; slot: number };

function unwindVirtualVisits(path: Edge[]): void {
  for (const edge of path) edge.virtualVisits = Math.max(0, edge.virtualVisits - 1);
}

function selectLeaf(node: Node, searchPolicy: SearchPolicy, context: SearchPolicyContext, path: Edge[] = []): SelectedLeaf {
  if (!node.expanded || node.terminalValue !== null) return { node, path };
  const best = selectBestEdge(node, searchPolicy, context);
  best.virtualVisits += 1;
  if (!best.child) best.child = makeChild(node, best.move);
  return selectLeaf(best.child, searchPolicy, context, [...path, best]);
}

function prepareExpansion(node: Node, stats: SearchStats): Move[] | number {
  const moves = legalMoves(node.board);
  if (!moves.length) {
    node.expanded = true;
    node.terminalValue = inCheck(node.board) ? -1 : 0;
    node.edges = [];
    stats.expansions += 1;
    stats.terminalHits += 1;
    return node.terminalValue;
  }
  return moves;
}

function finishExpansion(node: Node, moves: Move[], evaln: Evaluation): number {
  const raw = moves.map((move) => Math.max(0, evaln.policy.get(moveToActionId(move)) ?? 0));
  const total = raw.reduce((a, b) => a + b, 0);
  const fallback = 1 / moves.length;
  node.edges = moves.map((move, i) => {
    const actionId = moveToActionId(move);
    return {
      move,
      prior: total > 0 ? raw[i] / total : fallback,
      child: null,
      visits: 0,
      valueSum: 0,
      virtualVisits: 0,
      actionValuePrior: evaln.actionValues?.get(actionId),
      rankScore: evaln.rankScores?.get(actionId),
      regret: evaln.regrets?.get(actionId),
      risk: evaln.risks?.get(actionId),
      uncertainty: evaln.uncertainties?.get(actionId),
    };
  });
  node.expanded = true;
  node.terminalValue = null;
  return valueFromWdl(evaln.wdl);
}

async function runBatchedVisits(root: Node, evaluator: Evaluator, visits: number, searchPolicy: SearchPolicy, context: SearchPolicyContext, batchSize: number, stats: SearchStats) {
  let done = 0;
  while (done < visits) {
    const want = Math.min(batchSize, visits - done);
    const selected: SelectedLeaf[] = [];
    for (let i = 0; i < want; i++) selected.push(selectLeaf(root, searchPolicy, context));

    const evalNodes: Node[] = [];
    const evalMoves: Move[][] = [];
    const evalIndex = new Map<Node, number>();
    const prepared: PreparedLeaf[] = [];
    for (const sel of selected) {
      if (sel.node.terminalValue !== null) { prepared.push({ kind: 'terminal', sel, value: sel.node.terminalValue }); stats.terminalHits += 1; continue; }
      if (sel.node.expanded) {
        unwindVirtualVisits(sel.path);
        throw new Error('selectLeaf returned an expanded non-terminal node');
      }
      const prep = prepareExpansion(sel.node, stats);
      if (typeof prep === 'number') prepared.push({ kind: 'terminal', sel, value: prep });
      else {
        let slot = evalIndex.get(sel.node);
        if (slot === undefined) {
          slot = evalNodes.length;
          evalIndex.set(sel.node, slot);
          evalNodes.push(sel.node);
          evalMoves.push(prep);
        }
        prepared.push({ kind: 'eval', sel, slot });
      }
    }

    let evals: Evaluation[] = [];
    if (evalNodes.length) {
      const contexts = evalNodes.map((node, i) => ({ historyFens: node.historyFens, legalMoves: evalMoves[i] }));
      stats.evalCalls += evalNodes.length;
      stats.maxEvalBatch = Math.max(stats.maxEvalBatch, evalNodes.length);
      if (evaluator.evaluateBatch) {
        stats.batchEvalCalls += 1;
        evals = await evaluator.evaluateBatch(evalNodes.map((node) => node.board), contexts);
      } else {
        evals = await Promise.all(evalNodes.map((node, i) => evaluator.evaluate(node.board, contexts[i])));
      }
    }
    const values = evalNodes.map((node, i) => { stats.expansions += 1; return finishExpansion(node, evalMoves[i], evals[i]); });
    for (const item of prepared) {
      let value: number | undefined;
      if (item.kind === 'terminal') value = item.value;
      else {
        value = values[item.slot];
        if (value === undefined) throw new Error(`missing batched evaluation value for slot ${item.slot}`);
      }
      searchPolicy.backup(item.sel.path, value, context);
      stats.completedVisits += 1;
    }
    done += want;
  }
}

export async function searchRoot(board: BoardState, evaluator: Evaluator, options: SearchOptions = {}): Promise<SearchResult> {
  const visits = Math.max(1, Math.floor(options.visits ?? 8));
  const context: SearchPolicyContext = { cpuct: options.cpuct ?? 1.5, fpu: options.fpu ?? 0, temperature: options.temperature ?? 1, avWeight: options.avWeight ?? 0.25, rankWeight: options.rankWeight ?? 0, regretWeight: options.regretWeight ?? 0, riskWeight: options.riskWeight ?? 0, uncertaintyWeight: options.uncertaintyWeight ?? 0 };
  const searchPolicy = options.searchPolicy ?? classicPuctPolicy;
  const stats = makeStats(visits);
  const root: Node = { board, historyFens: options.historyFens ?? [], expanded: false, terminalValue: null, edges: [], isRoot: true };
  const rootValue = await expand(root, evaluator, stats);
  if (!root.edges.length) return { move: null, visits: 0, value: rootValue, policy: [], stats };
  const batchSize = Math.max(1, Math.floor(options.batchSize ?? 1));
  if (batchSize > 1) await runBatchedVisits(root, evaluator, visits, searchPolicy, context, batchSize, stats);
  else {
    for (let i = 0; i < visits; i++) {
      await simulate(root, evaluator, searchPolicy, context, stats);
      stats.completedVisits += 1;
    }
  }
  const policy = searchPolicy.rootPolicy(root.edges, context, root);
  const bestEntry = searchPolicy.chooseFinalMove(policy, context);
  return { move: bestEntry?.move ?? null, visits: root.edges.reduce((sum, edge) => sum + edge.visits, 0), value: bestEntry?.q ?? rootValue, policy, stats };
}

export async function chooseMove(board: BoardState, evaluator: Evaluator, options: SearchOptions = {}): Promise<SearchResult> {
  return searchRoot(board, evaluator, { ...options, temperature: options.temperature ?? 0 });
}
