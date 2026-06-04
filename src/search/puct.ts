import { boardToFen, type BoardState } from '../chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import { automaticDrawReason } from '../chess/drawRules.ts';
import type { Evaluation, Evaluator } from '../nn/evaluator.ts';

export interface SearchPolicyEntry { move: Move; visits: number; prior: number; q: number; probability: number; }
export interface PrincipalVariationEntry { move: Move; visits: number; prior: number; q: number; depth: number; }
export type PrincipalVariationSelector = 'visits' | 'q' | 'puct';
export interface ProgressiveWideningOptions {
  /** Policy mass admitted before widening. Set to 1 to disable top-p pruning. */
  topP?: number;
  /** Minimum number of policy-sorted legal moves admitted at every node. */
  minActions?: number;
  /** Parent visit interval for admitting more quiet moves in linear mode. */
  visitsPerAction?: number;
  /** Number of additional policy-sorted moves admitted per widening step. */
  actionsPerStep?: number;
  /** Widening schedule: linear is Tiny-Leela's original approximation; exponential is Monty-shaped. */
  wideningSchedule?: 'linear' | 'exponential';
  /** Exponential schedule starts at 2^visitThresholdPower visits, then doubles. */
  visitThresholdPower?: number;
  /** Always admit captures, promotions, and checking moves even outside top-p. */
  includeForcing?: boolean;
  /** Experimental cpuct multiplier from policy Gini impurity; 0 leaves cpuct unchanged. */
  giniExploreScale?: number;
  /** Gini mode: flat explores more for flatter policy; sharp explores more for sharper policy. */
  giniExploreMode?: 'flat' | 'sharp';
  minExploreScale?: number;
  maxExploreScale?: number;
}
export interface SearchResult { move: Move | null; visits: number; value: number; policy: SearchPolicyEntry[]; principalVariation?: PrincipalVariationEntry[]; multiPvLines?: PrincipalVariationEntry[][]; stats?: SearchStats; root?: Node; }
export type CpuctSchedule = 'constant' | 'lc0-log';
export type FpuStrategy = 'constant' | 'lc0-reduction';
export type SearchBudgetMode = 'visits' | 'neural';
export type SearchBatchCollisionMode = 'retry' | 'backup';
export type SearchEarlyStop = 'none' | 'root-dominance' | 'kld-stable' | 'best-stable';
export type ValueWdlBlendMode = 'constant' | 'confidence';
export interface SearchOptions {
  visits?: number;
  cpuct?: number;
  fpu?: number;
  temperature?: number;
  /** Experimental LC0-style cpuct growth. Default constant preserves classic Tiny Leela PUCT. */
  cpuctSchedule?: CpuctSchedule;
  cpuctBase?: number;
  cpuctFactor?: number;
  /** Experimental LC0-style FPU reduction by visited policy mass. Default constant preserves old behavior. */
  fpuStrategy?: FpuStrategy;
  fpuReduction?: number;
  historyFens?: string[];
  batchSize?: number;
  /**
   * Batched MCTS leaf collisions. `retry` keeps temporary virtual visits on
   * already-selected in-flight leaves and tries to fill the batch with distinct
   * leaves; `backup` preserves the older behavior of backing up duplicate leaf
   * hits after one shared evaluation.
   */
  batchCollisionMode?: SearchBatchCollisionMode;
  /** Max retry selections per desired batch slot when avoiding in-flight leaves. */
  batchCollisionRetryLimit?: number;
  searchPolicy?: SearchPolicy;
  avWeight?: number;
  rankWeight?: number;
  regretWeight?: number;
  riskWeight?: number;
  uncertaintyWeight?: number;
  /** Optional auxiliary WDL head to blend into leaf values, e.g. wdl_sf18. */
  valueWdlAuxHead?: string;
  /** Constant alpha, or max alpha in confidence mode. */
  valueWdlAuxWeight?: number;
  valueWdlBlendMode?: ValueWdlBlendMode;
  /** Temperature applied to base/aux WDL probability distributions before value extraction. */
  valueWdlBaseTemp?: number;
  valueWdlAuxTemp?: number;
  includePv?: boolean;
  pvDepth?: number;
  pvSelector?: PrincipalVariationSelector;
  /** Extract this many principal variations (one per top visited root move). >1 enables MultiPV. */
  multiPv?: number;
  rootMoves?: Move[];
  signal?: AbortSignal;
  /** Soft wall-clock budget. When elapsed, search returns best-so-far instead of throwing. */
  movetimeMs?: number;
  yieldEveryMs?: number;
  /** Reusable root from a previous search at the same position. Ignored when rootMoves restricts the root. */
  root?: Node | null;
  /** Experimental per-search/game transposition table. Key includes FEN and history for history-aware model correctness. */
  transpositionTable?: Map<string, Node>;
  /** Budget fixed visits or paid neural/backend eval misses when a cache-aware evaluator is present. */
  budgetMode?: SearchBudgetMode;
  /** Adaptive neural mode cap: max additional root visits = visits * multiplier. */
  maxVisitsMultiplier?: number;
  /** Safe adaptive stop; root-dominance is for neural budget; kld-stable is opt-in for datagen/self-play. */
  earlyStop?: SearchEarlyStop;
  /** KLD early-stop guardrails for datagen/search-label generation only. */
  kldMinVisits?: number;
  kldCheckInterval?: number;
  kldThreshold?: number;
  kldStableChecks?: number;
  /** Best-move stability early-stop guardrails for play/search UX. */
  bestStableMinVisits?: number;
  bestStableCheckInterval?: number;
  bestStableChecks?: number;
  bestStableMinVisitLead?: number;
  bestStableMaxQDelta?: number;
  /** Experimental variance-aware exploration multiplier. 0 disables. */
  cpuctVarianceWeight?: number;
  cpuctVarianceMaxScale?: number;
  /** Experimental Butterfly/history prior. 0 disables. */
  butterflyWeight?: number;
  butterflyDecay?: number;
  butterflyMaxBonus?: number;
}
export interface GumbelRootOptions { candidateCount?: number; seed?: number; gumbelScale?: number; qWeight?: number; priorWeight?: number; visitPenalty?: number; }

export interface SearchStats {
  requestedVisits: number;
  completedVisits: number;
  expansions: number;
  terminalHits: number;
  /** Evaluation requests made by search before cache filtering. */
  evalCalls: number;
  batchEvalCalls: number;
  maxEvalBatch: number;
  /** Histogram of leaf-evaluation batch sizes, keyed by the number of leaves sent to evaluateBatch. */
  evalBatchSizeHistogram?: Record<string, number>;
  /** Duplicate in-flight leaves encountered while collecting a search batch. */
  batchLeafCollisions?: number;
  /** Extra selection attempts spent to avoid duplicate in-flight leaves. */
  batchLeafRetries?: number;
  budgetMode?: SearchBudgetMode;
  requestedNeuralEvals?: number;
  neuralEvalMisses?: number;
  cacheHits?: number;
  maxRootVisits?: number;
  stopReason?: 'visit-budget' | 'movetime' | 'neural-budget' | 'max-visits' | 'root-dominance' | 'kld-stable' | 'best-stable' | 'no-cache-metrics-fixed-visits';
  rootReused?: boolean;
  transpositionHits?: number;
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
  /** Side to move at the parent; used by the opt-in Butterfly prior table. */
  side?: BoardState['turn'];
}

export interface Node {
  board: BoardState;
  historyFens: string[];
  expanded: boolean;
  terminalValue: number | null;
  edges: Edge[];
  /** Non-terminal model evaluation used to expand this node, retained for diagnostics/UI only. */
  evaluation?: Evaluation;
  /** Number of completed simulations that reached this node. For reused roots,
   * this preserves the inbound edge visit count so a target-visit search does
   * not add a full fresh budget on top of already-reached nodes. */
  visits?: number;
  isRoot?: boolean;
}

export interface SearchPolicyContext {
  cpuct: number;
  fpu: number;
  temperature: number;
  cpuctSchedule: CpuctSchedule;
  cpuctBase: number;
  cpuctFactor: number;
  fpuStrategy: FpuStrategy;
  fpuReduction: number;
  /** Weights for optional edge-local aux signals in experimental search policies. */
  avWeight: number;
  rankWeight: number;
  regretWeight: number;
  riskWeight: number;
  uncertaintyWeight: number;
  valueWdlAuxHead?: string;
  valueWdlAuxWeight: number;
  valueWdlBlendMode: ValueWdlBlendMode;
  valueWdlBaseTemp: number;
  valueWdlAuxTemp: number;
  cpuctVarianceWeight: number;
  cpuctVarianceMaxScale: number;
  butterflyWeight: number;
  butterflyDecay: number;
  butterflyMaxBonus: number;
  butterflyTable?: Map<string, number>;
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

function normalizeWdl(wdl: [number, number, number], temperature = 1): [number, number, number] {
  const t = Number.isFinite(temperature) && temperature > 0 ? temperature : 1;
  if (Math.abs(t - 1) < 1e-9) return wdl;
  const pow = 1 / t;
  const vals = wdl.map((p) => Math.pow(Math.max(1e-12, Number(p) || 0), pow));
  const sum = vals.reduce((a, b) => a + b, 0) || 1;
  return [vals[0] / sum, vals[1] / sum, vals[2] / sum];
}

function wdlConfidence(wdl: [number, number, number]): number {
  const h = -wdl.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
  return Math.max(0, Math.min(1, 1 - h / Math.log(3)));
}

function valueFromEvaluation(evaln: Evaluation, context: SearchPolicyContext): number {
  const baseWdl = normalizeWdl(evaln.wdl, context.valueWdlBaseTemp);
  const baseValue = valueFromWdl(baseWdl);
  const auxHead = context.valueWdlAuxHead;
  const aux = auxHead ? evaln.auxiliaryWdls?.[auxHead] : undefined;
  const maxAlpha = Math.max(0, Math.min(1, context.valueWdlAuxWeight));
  if (!aux || maxAlpha <= 0) return baseValue;
  const auxWdl = normalizeWdl(aux, context.valueWdlAuxTemp);
  const auxValue = valueFromWdl(auxWdl);
  const alpha = context.valueWdlBlendMode === 'confidence'
    ? maxAlpha * wdlConfidence(auxWdl) * (1 - wdlConfidence(baseWdl))
    : maxAlpha;
  return (1 - alpha) * baseValue + alpha * auxValue;
}

export function edgeQForParent(edge: Edge, fpu = 0): number {
  return edge.visits ? -edge.valueSum / edge.visits : fpu;
}

export function edgeSelectVisits(edge: Edge): number {
  return edge.visits + edge.virtualVisits;
}

export function computeCpuct(context: SearchPolicyContext, parentVisits: number): number {
  if (context.cpuctSchedule !== 'lc0-log') return context.cpuct;
  const base = Math.max(1e-9, context.cpuctBase);
  return context.cpuct + context.cpuctFactor * Math.log((Math.max(0, parentVisits) + base) / base);
}

function edgeQVariance(edges: Edge[], context: SearchPolicyContext): number {
  if (context.cpuctVarianceWeight <= 0 || edges.length < 2) return 0;
  const weights = edges.map((edge) => Math.max(edge.prior, 1e-9));
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const qForVariance = (edge: Edge) => edge.visits > 0
    ? edgeQForParent(edge, context.fpu)
    : context.fpuStrategy === 'lc0-reduction'
      ? parentQForNode({ board: null as unknown as BoardState, historyFens: [], expanded: true, terminalValue: null, edges }, context.fpu) - context.fpuReduction * Math.sqrt(visitedPolicyMass({ board: null as unknown as BoardState, historyFens: [], expanded: true, terminalValue: null, edges }))
      : context.fpu;
  const qs = edges.map(qForVariance);
  const mean = qs.reduce((sum, q, i) => sum + q * weights[i], 0) / total;
  return qs.reduce((sum, q, i) => sum + weights[i] * (q - mean) * (q - mean), 0) / total;
}

function cpuctVarianceScale(edge: Edge, edges: Edge[], context: SearchPolicyContext): number {
  if (context.cpuctVarianceWeight <= 0) return 1;
  const sigma = Math.sqrt(Math.max(0, edgeQVariance(edges, context)));
  const uncertainty = 1 / Math.sqrt(1 + edgeSelectVisits(edge));
  return clamp(1 + context.cpuctVarianceWeight * sigma * uncertainty, 1, Math.max(1, context.cpuctVarianceMaxScale));
}

function butterflyKey(side: BoardState['turn'] | undefined, move: Move): string {
  return `${side ?? 'u'}:${move.from}:${move.to}`;
}

function butterflyScore(edge: Edge, context: SearchPolicyContext): number {
  if (context.butterflyWeight === 0 || !context.butterflyTable) return 0;
  const raw = context.butterflyTable.get(butterflyKey(edge.side, edge.move)) ?? 0;
  return context.butterflyWeight * Math.tanh(raw) * context.butterflyMaxBonus / (1 + edgeSelectVisits(edge));
}

function updateButterfly(edge: Edge, parentValue: number, context: SearchPolicyContext): void {
  if (context.butterflyWeight === 0 || !context.butterflyTable) return;
  const key = butterflyKey(edge.side, edge.move);
  const old = context.butterflyTable.get(key) ?? 0;
  context.butterflyTable.set(key, old * context.butterflyDecay + parentValue);
}

function visitedPolicyMass(node: Node): number {
  return node.edges.reduce((sum, edge) => sum + (edge.visits > 0 ? edge.prior : 0), 0);
}

function parentQForNode(node: Node, fpu: number): number {
  let weighted = 0;
  let visits = 0;
  for (const edge of node.edges) {
    if (edge.visits <= 0) continue;
    weighted += edge.visits * edgeQForParent(edge, fpu);
    visits += edge.visits;
  }
  return visits > 0 ? weighted / visits : fpu;
}

export function edgeQForParentInNode(edge: Edge, node: Node | undefined, context: SearchPolicyContext): number {
  if (edge.visits > 0) return edgeQForParent(edge, context.fpu);
  if (!node || context.fpuStrategy !== 'lc0-reduction') return context.fpu;
  return parentQForNode(node, context.fpu) - context.fpuReduction * Math.sqrt(visitedPolicyMass(node));
}

export class ClassicPUCTPolicy implements SearchPolicy {
  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number {
    const parentVisits = node.edges.reduce((sum, e) => sum + edgeSelectVisits(e), 0);
    const sqrtParent = Math.sqrt(parentVisits + 1);
    const q = edgeQForParentInNode(edge, node, context);
    const sv = edgeSelectVisits(edge);
    const u = computeCpuct(context, parentVisits) * cpuctVarianceScale(edge, node.edges, context) * edge.prior * sqrtParent / (1 + sv);
    return q + u + butterflyScore(edge, context);
  }

  backup(path: Edge[], leafValue: number, context: SearchPolicyContext): void {
    let value = leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      const edge = path[i];
      edge.virtualVisits = Math.max(0, edge.virtualVisits - 1);
      edge.visits += 1;
      if (edge.child) edge.child.visits = (edge.child.visits ?? 0) + 1;
      edge.valueSum += value;
      updateButterfly(edge, -value, context);
      value = -value;
    }
  }

  rootPolicy(edges: Edge[], context: SearchPolicyContext, node?: Node): SearchPolicyEntry[] {
    if (!edges.length) return [];
    const tau = Math.max(0, context.temperature);
    if (tau === 0) {
      const better = (a: Edge, b: Edge) => {
        if (b.visits !== a.visits) return b.visits > a.visits;
        const aq = edgeQForParentInNode(a, node, context), bq = edgeQForParentInNode(b, node, context);
        if (bq !== aq) return bq > aq;
        return b.prior > a.prior;
      };
      const best = edges.reduce((a, b) => better(a, b) ? b : a);
      return edges.map((edge) => ({ move: edge.move, visits: edge.visits, prior: edge.prior, q: edgeQForParentInNode(edge, node, context), probability: edge === best ? 1 : 0 }));
    }
    const weights = edges.map((edge) => Math.pow(Math.max(edge.visits, 1e-9), 1 / tau));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    return edges.map((edge, i) => ({ move: edge.move, visits: edge.visits, prior: edge.prior, q: edgeQForParentInNode(edge, node, context), probability: weights[i] / total }));
  }

  chooseFinalMove(entries: SearchPolicyEntry[]): SearchPolicyEntry | null {
    if (!entries.length) return null;
    return entries.reduce((a, b) => b.probability > a.probability ? b : a);
  }
}

interface ProgressiveWideningState {
  parentVisits: number;
  active: Set<Edge>;
  policyGini: number;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

function isForcingMove(board: BoardState, move: Move): boolean {
  if (move.promotion) return true;
  const moving = board.squares[move.from];
  const target = board.squares[move.to];
  if (target && target[0] !== board.turn) return true;
  if (moving?.[1] === 'p' && move.to === board.epSquare) return true;
  try {
    // makeMove flips side-to-move, so the default inCheck(child) asks whether
    // the opponent is in check after this move.
    return inCheck(makeMove(board, move));
  } catch {
    return false;
  }
}

export class ProgressiveWideningPUCTPolicy extends ClassicPUCTPolicy {
  private options: Required<ProgressiveWideningOptions>;
  private states = new WeakMap<Node, ProgressiveWideningState>();
  private forcingStates = new WeakMap<Node, Set<Edge>>();

  constructor(options: ProgressiveWideningOptions = {}) {
    super();
    this.options = {
      topP: clamp(options.topP ?? 0.85, 0, 1),
      minActions: Math.max(1, Math.floor(options.minActions ?? 6)),
      visitsPerAction: Math.max(1, Math.floor(options.visitsPerAction ?? 8)),
      actionsPerStep: Math.max(0, Math.floor(options.actionsPerStep ?? 2)),
      wideningSchedule: options.wideningSchedule ?? 'linear',
      visitThresholdPower: Math.max(0, Math.floor(options.visitThresholdPower ?? 2)),
      includeForcing: options.includeForcing ?? true,
      giniExploreScale: options.giniExploreScale ?? 0,
      giniExploreMode: options.giniExploreMode ?? 'flat',
      minExploreScale: options.minExploreScale ?? 0.5,
      maxExploreScale: options.maxExploreScale ?? 1.5,
    };
  }

  private state(node: Node): ProgressiveWideningState {
    const parentVisits = node.edges.reduce((sum, edge) => sum + edgeSelectVisits(edge), 0);
    const cached = this.states.get(node);
    if (cached && cached.parentVisits === parentVisits) return cached;

    const sorted = [...node.edges].sort((a, b) => b.prior - a.prior);
    const active = new Set<Edge>();
    let cumulative = 0;
    const baseLimit = Math.min(sorted.length, this.options.minActions);
    for (const edge of sorted) {
      if (active.size < baseLimit || cumulative < this.options.topP) {
        active.add(edge);
        cumulative += edge.prior;
      }
      if (active.size >= baseLimit && cumulative >= this.options.topP) break;
    }

    let widenedCount = active.size;
    if (this.options.wideningSchedule === 'exponential') {
      let threshold = 2 ** this.options.visitThresholdPower;
      while (parentVisits >= threshold && widenedCount < sorted.length) {
        widenedCount += this.options.actionsPerStep;
        threshold *= 2;
      }
    } else {
      widenedCount += this.options.actionsPerStep * Math.floor(parentVisits / this.options.visitsPerAction);
    }
    widenedCount = Math.min(sorted.length, widenedCount);
    for (const edge of sorted.slice(0, widenedCount)) active.add(edge);

    if (this.options.includeForcing) {
      let forcing = this.forcingStates.get(node);
      if (!forcing) {
        forcing = new Set(node.edges.filter((edge) => isForcingMove(node.board, edge.move)));
        this.forcingStates.set(node, forcing);
      }
      for (const edge of forcing) active.add(edge);
    }

    const policyGini = 1 - node.edges.reduce((sum, edge) => sum + edge.prior * edge.prior, 0);
    const state = { parentVisits, active, policyGini };
    this.states.set(node, state);
    return state;
  }

  scoreEdge(node: Node, edge: Edge, context: SearchPolicyContext): number {
    const state = this.state(node);
    if (!state.active.has(edge)) return -Infinity;
    const parentVisits = node.edges.reduce((sum, e) => sum + edgeSelectVisits(e), 0);
    const sqrtParent = Math.sqrt(parentVisits + 1);
    const q = edgeQForParentInNode(edge, node, context);
    const sv = edgeSelectVisits(edge);
    const exploreScale = this.options.giniExploreScale === 0
      ? 1
      : this.options.giniExploreMode === 'sharp'
        // Monty-shaped: sharp/low-Gini policies get extra exploration so search
        // does not overtrust a very peaked policy prefix. Pivot at Gini ~= 0.5.
        ? clamp(1 + this.options.giniExploreScale * (-Math.log(clamp(state.policyGini + 0.001, 0.001, 1)) - Math.LN2), this.options.minExploreScale, this.options.maxExploreScale)
        : clamp(1 + this.options.giniExploreScale * (state.policyGini - 0.5), this.options.minExploreScale, this.options.maxExploreScale);
    const u = computeCpuct(context, parentVisits) * cpuctVarianceScale(edge, node.edges, context) * exploreScale * edge.prior * sqrtParent / (1 + sv);
    return q + u + butterflyScore(edge, context);
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
    const q = edgeQForParentInNode(edge, node, context);
    const sv = edgeSelectVisits(edge);
    const puct = computeCpuct(context, parentVisits) * edge.prior * Math.sqrt(parentVisits + 1) / (1 + sv);
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
export const progressiveWideningPuctPolicy = new ProgressiveWideningPUCTPolicy();
export const montyLitePuctPolicy = new ProgressiveWideningPUCTPolicy({ topP: 0.85, minActions: 6, visitsPerAction: 8, actionsPerStep: 2, wideningSchedule: 'exponential', visitThresholdPower: 2, includeForcing: true, giniExploreScale: 0.5, giniExploreMode: 'sharp' });
export const actionValuePuctPolicy = new ActionValuePUCTPolicy();
export const auxPuctPolicy = new AuxPUCTPolicy();

function makeStats(visits: number): SearchStats {
  return { requestedVisits: visits, completedVisits: 0, expansions: 0, terminalHits: 0, evalCalls: 0, batchEvalCalls: 0, maxEvalBatch: 0, neuralEvalMisses: 0, cacheHits: 0 };
}

interface EvaluatorMetricsSnapshot { hits: number; misses: number; }

function evaluatorMetrics(evaluator: Evaluator): EvaluatorMetricsSnapshot | null {
  const maybe = evaluator as Evaluator & { metrics?: () => { hits?: number; misses?: number } };
  if (typeof maybe.metrics !== 'function') return null;
  const metrics = maybe.metrics();
  if (!metrics) return null;
  const hits = Number(metrics.hits ?? 0);
  const misses = Number(metrics.misses ?? 0);
  if (!Number.isFinite(hits) || !Number.isFinite(misses)) return null;
  return { hits, misses };
}

function recordEvaluatorMetricsDelta(evaluator: Evaluator, before: EvaluatorMetricsSnapshot | null, fallbackMisses: number, stats: SearchStats): void {
  const after = before ? evaluatorMetrics(evaluator) : null;
  if (before && after) {
    stats.cacheHits = (stats.cacheHits ?? 0) + Math.max(0, after.hits - before.hits);
    stats.neuralEvalMisses = (stats.neuralEvalMisses ?? 0) + Math.max(0, after.misses - before.misses);
  } else {
    stats.neuralEvalMisses = (stats.neuralEvalMisses ?? 0) + Math.max(0, fallbackMisses);
  }
}

function rootVisitCount(root: Node): number {
  return root.edges.reduce((sum, edge) => sum + edge.visits, 0);
}

function rootVisitDominanceStop(root: Node, maxRootVisits: number): boolean {
  const current = rootVisitCount(root);
  if (current >= maxRootVisits) return true;
  if (root.edges.length < 2) return false;
  const visits = root.edges.map((edge) => edge.visits).sort((a, b) => b - a);
  const remaining = Math.max(0, maxRootVisits - current);
  return visits[0] > visits[1] + remaining;
}

interface KldEarlyStopState { lastDistribution?: number[]; lastBest?: number; stableChecks: number; }
interface BestStableEarlyStopState { lastBest?: number; lastBestQ?: number; stableChecks: number; }

function rootVisitDistribution(root: Node): number[] {
  const total = Math.max(1, rootVisitCount(root));
  return root.edges.map((edge) => edge.visits / total);
}

function rootBestEdgeIndex(root: Node): number | undefined {
  if (!root.edges.length) return undefined;
  let best = 0;
  for (let i = 1; i < root.edges.length; i++) {
    if (root.edges[i].visits > root.edges[best].visits) best = i;
  }
  return best;
}

function kld(current: number[], previous: number[]): number {
  const eps = 1e-12;
  let out = 0;
  for (let i = 0; i < current.length; i++) {
    const p = Math.max(eps, current[i] ?? 0);
    const q = Math.max(eps, previous[i] ?? 0);
    out += p * Math.log(p / q);
  }
  return out;
}

function rootKldEarlyStop(root: Node, options: SearchOptions, state: KldEarlyStopState): boolean {
  if (options.earlyStop !== 'kld-stable') return false;
  const visits = rootVisitCount(root);
  const minVisits = Math.max(1, Math.floor(options.kldMinVisits ?? Math.min(options.visits ?? visits, 32)));
  const interval = Math.max(1, Math.floor(options.kldCheckInterval ?? 8));
  if (visits < minVisits || visits % interval !== 0) return false;
  const current = rootVisitDistribution(root);
  const best = rootBestEdgeIndex(root);
  if (!state.lastDistribution || state.lastDistribution.length !== current.length) {
    state.lastDistribution = current;
    state.lastBest = best;
    state.stableChecks = 0;
    return false;
  }
  const threshold = Math.max(0, options.kldThreshold ?? 1e-4);
  const sameBest = best !== undefined && best === state.lastBest;
  const stable = sameBest && kld(current, state.lastDistribution) <= threshold;
  state.lastDistribution = current;
  state.lastBest = best;
  state.stableChecks = stable ? state.stableChecks + 1 : 0;
  return state.stableChecks >= Math.max(1, Math.floor(options.kldStableChecks ?? 2));
}

function rootBestStableEarlyStop(root: Node, context: SearchPolicyContext, options: SearchOptions, state: BestStableEarlyStopState): boolean {
  if (options.earlyStop !== 'best-stable') return false;
  const visits = rootVisitCount(root);
  const minVisits = Math.max(1, Math.floor(options.bestStableMinVisits ?? 32));
  const interval = Math.max(1, Math.floor(options.bestStableCheckInterval ?? 8));
  if (visits < minVisits || visits % interval !== 0) return false;
  const best = rootBestEdgeIndex(root);
  if (best === undefined) return false;
  const sorted = root.edges
    .map((edge, index) => ({ index, visits: edge.visits, q: edgeQForParentInNode(edge, root, context) }))
    .sort((a, b) => b.visits - a.visits || b.q - a.q);
  const bestEntry = sorted[0];
  const secondVisits = sorted[1]?.visits ?? 0;
  const minLead = Math.max(0, Math.floor(options.bestStableMinVisitLead ?? interval));
  const maxQDelta = Math.max(0, options.bestStableMaxQDelta ?? 0.05);
  const sameBest = bestEntry.index === state.lastBest;
  const stableQ = state.lastBestQ === undefined || Math.abs(bestEntry.q - state.lastBestQ) <= maxQDelta;
  const stable = sameBest && bestEntry.visits >= secondVisits + minLead && stableQ;
  state.lastBest = bestEntry.index;
  state.lastBestQ = bestEntry.q;
  state.stableChecks = stable ? state.stableChecks + 1 : 0;
  return state.stableChecks >= Math.max(1, Math.floor(options.bestStableChecks ?? 2));
}

function rootEarlyStopReason(root: Node, context: SearchPolicyContext, options: SearchOptions, visitTarget: number, kldState: KldEarlyStopState, bestState: BestStableEarlyStopState): SearchStats['stopReason'] | undefined {
  // Early-stop telemetry should only report a real early exit, not completion
  // exactly at the scheduled visit cap.
  if (rootVisitCount(root) >= visitTarget) return undefined;
  if (options.earlyStop === 'root-dominance' && rootVisitDominanceStop(root, visitTarget)) return 'root-dominance';
  if (rootBestStableEarlyStop(root, context, options, bestState)) return 'best-stable';
  if (rootKldEarlyStop(root, options, kldState)) return 'kld-stable';
  return undefined;
}

function searchAbortError(): Error {
  if (typeof DOMException !== 'undefined') return new DOMException('Search cancelled', 'AbortError');
  const err = new Error('Search cancelled');
  err.name = 'AbortError';
  return err;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw searchAbortError();
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function browserParam(name: string): string | null {
  try {
    if (typeof location === 'undefined') return null;
    return new URLSearchParams(location.search).get(name);
  } catch {
    return null;
  }
}

function debugTokens(value: string | null | undefined): string[] {
  return String(value ?? '').toLowerCase().split(/[,+\s]+/).map((s) => s.trim()).filter(Boolean);
}

function latencyDebugEnabled(): boolean {
  const direct = browserParam('debugLatency') ?? browserParam('latency');
  if (direct !== null) return !['0', 'false', 'no', 'off'].includes(direct.toLowerCase());
  const tokens = [...debugTokens(browserParam('debug')), ...debugTokens(browserParam('tlDebug'))];
  return tokens.some((token) => ['1', 'true', 'yes', 'on', 'all', 'perf', 'timing', 'latency', 'search', 'puct'].includes(token));
}

function logSearchLatency(label: string, payload: Record<string, unknown>): void {
  if (!latencyDebugEnabled()) return;
  console.info(`Tiny Leela latency: ${label}`, Object.fromEntries(Object.entries(payload).map(([key, value]) => [key, typeof value === 'number' ? Number(value.toFixed(2)) : value])));
}

function yieldToUi(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function deadlineFromMovetime(startMs: number, movetimeMs?: number): number | undefined {
  return movetimeMs && Number.isFinite(movetimeMs) && movetimeMs > 0 ? startMs + Math.floor(movetimeMs) : undefined;
}

function deadlineExpired(deadlineMs?: number): boolean {
  return deadlineMs !== undefined && nowMs() >= deadlineMs;
}

function sameHistory(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((fen, i) => fen === b[i]);
}

export function searchNodeKey(board: BoardState, historyFens: string[] = []): string {
  return `${boardToFen(board)}\nh:${historyFens.join('|')}`;
}

function makeSearchRoot(board: BoardState, historyFens: string[] = []): Node {
  return { board, historyFens, expanded: false, terminalValue: null, edges: [], visits: 0, isRoot: true };
}

function makeChild(parent: Node, move: Move, transpositionTable?: Map<string, Node>, stats?: SearchStats): Node {
  const childBoard = makeMove(parent.board, move);
  const childHistory = [boardToFen(parent.board), ...parent.historyFens];
  const key = searchNodeKey(childBoard, childHistory);
  const cached = transpositionTable?.get(key);
  if (cached) {
    if (stats) stats.transpositionHits = (stats.transpositionHits ?? 0) + 1;
    cached.isRoot = false;
    return cached;
  }
  const child = { board: childBoard, historyFens: childHistory, expanded: false, terminalValue: null, edges: [], visits: 0, isRoot: false };
  transpositionTable?.set(key, child);
  return child;
}

export function advanceSearchRoot(root: Node | null | undefined, move: Move, nextBoard: BoardState, nextHistoryFens: string[] = []): Node | null {
  if (!root?.expanded) return null;
  const actionId = moveToActionId(move);
  const expectedFen = boardToFen(nextBoard);
  for (const edge of root.edges) {
    if (moveToActionId(edge.move) !== actionId || !edge.child) continue;
    if (boardToFen(edge.child.board) !== expectedFen || !sameHistory(edge.child.historyFens, nextHistoryFens)) continue;
    root.isRoot = false;
    edge.child.isRoot = true;
    return edge.child;
  }
  return null;
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

function comparePvEdges(a: Edge, b: Edge, context: SearchPolicyContext, selector: PrincipalVariationSelector, node?: Node, searchPolicy?: SearchPolicy): Edge {
  if (selector === 'q') {
    const aq = edgeQForParentInNode(a, node, context), bq = edgeQForParentInNode(b, node, context);
    if (bq !== aq) return bq > aq ? b : a;
  } else if (selector === 'puct' && node && searchPolicy) {
    const as = searchPolicy.scoreEdge(node, a, context), bs = searchPolicy.scoreEdge(node, b, context);
    if (bs !== as) return bs > as ? b : a;
  } else if (b.visits !== a.visits) {
    return b.visits > a.visits ? b : a;
  }
  const aq = edgeQForParentInNode(a, node, context), bq = edgeQForParentInNode(b, node, context);
  if (bq !== aq) return bq > aq ? b : a;
  return b.prior > a.prior ? b : a;
}

function extractPrincipalVariation(root: Node, context: SearchPolicyContext, searchPolicy: SearchPolicy, maxDepth: number, selector: PrincipalVariationSelector): PrincipalVariationEntry[] {
  const pv: PrincipalVariationEntry[] = [];
  let node: Node | null = root;
  const depthLimit = Math.max(1, Math.floor(maxDepth));
  for (let depth = 1; depth <= depthLimit && node?.expanded && node.terminalValue === null; depth++) {
    const current = node;
    const visited: Edge[] = current.edges.filter((edge) => edge.visits > 0);
    if (!visited.length) break;
    let best: Edge = visited[0];
    for (const candidate of visited.slice(1)) best = comparePvEdges(best, candidate, context, selector, current, searchPolicy);
    pv.push({ move: best.move, visits: best.visits, prior: best.prior, q: edgeQForParentInNode(best, current, context), depth });
    node = best.child;
  }
  return pv;
}

// MultiPV: one principal variation per top visited root move, each line starting
// with its root move and then following the usual best-visited continuation.
function extractMultiPv(root: Node, context: SearchPolicyContext, searchPolicy: SearchPolicy, maxDepth: number, selector: PrincipalVariationSelector, count: number): PrincipalVariationEntry[][] {
  const depthLimit = Math.max(1, Math.floor(maxDepth));
  const rootEdges = root.edges
    .filter((edge) => edge.visits > 0)
    .sort((a, b) => b.visits - a.visits || b.prior - a.prior)
    .slice(0, Math.max(1, Math.floor(count)));
  return rootEdges.map((rootEdge) => {
    const line: PrincipalVariationEntry[] = [{
      move: rootEdge.move,
      visits: rootEdge.visits,
      prior: rootEdge.prior,
      q: edgeQForParentInNode(rootEdge, root, context),
      depth: 1,
    }];
    if (rootEdge.child) {
      for (const entry of extractPrincipalVariation(rootEdge.child, context, searchPolicy, depthLimit - 1, selector)) {
        line.push({ ...entry, depth: entry.depth + 1 });
      }
    }
    return line;
  });
}

async function expand(node: Node, evaluator: Evaluator, context: SearchPolicyContext, stats?: SearchStats): Promise<number> {
  const moves = legalMoves(node.board);
  if (!moves.length) {
    node.expanded = true;
    // No legal moves: checkmate is loss for the side to move, stalemate is draw.
    node.terminalValue = inCheck(node.board) ? -1 : 0;
    node.edges = [];
    if (stats) { stats.expansions += 1; stats.terminalHits += 1; }
    return node.terminalValue;
  }
  if (automaticDrawReason(node.board, node.historyFens)) {
    node.expanded = true;
    node.terminalValue = 0;
    node.edges = [];
    if (stats) { stats.expansions += 1; stats.terminalHits += 1; }
    return node.terminalValue;
  }
  const beforeMetrics = stats ? evaluatorMetrics(evaluator) : null;
  if (stats) stats.evalCalls += 1;
  const evaln = await evaluator.evaluate(node.board, { historyFens: node.historyFens, legalMoves: moves });
  if (stats) recordEvaluatorMetricsDelta(evaluator, beforeMetrics, 1, stats);
  const value = finishExpansion(node, moves, evaln, context);
  if (stats) stats.expansions += 1;
  return value;
}

async function simulate(node: Node, evaluator: Evaluator, searchPolicy: SearchPolicy, context: SearchPolicyContext, stats: SearchStats, transpositionTable?: Map<string, Node>): Promise<number> {
  if (!node.expanded) return expand(node, evaluator, context, stats);
  if (node.terminalValue !== null) { stats.terminalHits += 1; return node.terminalValue; }
  const best = selectBestEdge(node, searchPolicy, context);
  if (!best.child) best.child = makeChild(node, best.move, transpositionTable, stats);
  const childValue = await simulate(best.child, evaluator, searchPolicy, context, stats, transpositionTable);
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

function selectLeaf(node: Node, searchPolicy: SearchPolicy, context: SearchPolicyContext, path: Edge[] = [], transpositionTable?: Map<string, Node>, stats?: SearchStats): SelectedLeaf {
  if (!node.expanded || node.terminalValue !== null) return { node, path };
  const best = selectBestEdge(node, searchPolicy, context);
  best.virtualVisits += 1;
  if (!best.child) best.child = makeChild(node, best.move, transpositionTable, stats);
  return selectLeaf(best.child, searchPolicy, context, [...path, best], transpositionTable, stats);
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
  if (automaticDrawReason(node.board, node.historyFens)) {
    node.expanded = true;
    node.terminalValue = 0;
    node.edges = [];
    stats.expansions += 1;
    stats.terminalHits += 1;
    return node.terminalValue;
  }
  return moves;
}

function finishExpansion(node: Node, moves: Move[], evaln: Evaluation, context: SearchPolicyContext): number {
  const raw = moves.map((move) => Math.max(0, evaln.policy.get(moveToActionId(move)) ?? 0));
  const total = raw.reduce((a, b) => a + b, 0);
  const fallback = 1 / moves.length;
  node.evaluation = evaln;
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
      side: node.board.turn,
    };
  });
  node.expanded = true;
  node.terminalValue = null;
  return valueFromEvaluation(evaln, context);
}

async function runBatchedVisits(root: Node, evaluator: Evaluator, visits: number, searchPolicy: SearchPolicy, context: SearchPolicyContext, batchSize: number, stats: SearchStats, signal?: AbortSignal, yieldEveryMs = 0, transpositionTable?: Map<string, Node>, deadlineMs?: number, collisionMode: SearchBatchCollisionMode = 'retry', collisionRetryLimit = batchSize * 4) {
  let done = 0;
  let lastYield = nowMs();
  while (done < visits && !deadlineExpired(deadlineMs)) {
    throwIfAborted(signal);
    const want = Math.min(batchSize, visits - done);
    const selected: SelectedLeaf[] = [];
    const inFlightEvalLeaves = new Set<Node>();
    const retryVirtualPaths: Edge[][] = [];
    let attempts = 0;
    const maxAttempts = collisionMode === 'retry'
      ? Math.max(want, want + Math.max(0, Math.floor(collisionRetryLimit)))
      : want;
    while (selected.length < want && attempts < maxAttempts) {
      attempts += 1;
      const sel = selectLeaf(root, searchPolicy, context, [], transpositionTable, stats);
      if (collisionMode === 'retry' && !sel.node.expanded && sel.node.terminalValue === null) {
        if (inFlightEvalLeaves.has(sel.node)) {
          stats.batchLeafCollisions = (stats.batchLeafCollisions ?? 0) + 1;
          stats.batchLeafRetries = (stats.batchLeafRetries ?? 0) + 1;
          // Keep this temporary virtual path until batch collection finishes so
          // subsequent selections see the leaf as in flight and can diversify.
          retryVirtualPaths.push(sel.path);
          continue;
        }
        inFlightEvalLeaves.add(sel.node);
      }
      selected.push(sel);
    }
    while (selected.length < want) selected.push(selectLeaf(root, searchPolicy, context, [], transpositionTable, stats));
    for (const path of retryVirtualPaths) unwindVirtualVisits(path);

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
      const batchKey = String(evalNodes.length);
      stats.evalBatchSizeHistogram = { ...(stats.evalBatchSizeHistogram ?? {}), [batchKey]: (stats.evalBatchSizeHistogram?.[batchKey] ?? 0) + 1 };
      const beforeMetrics = evaluatorMetrics(evaluator);
      if (evaluator.evaluateBatch) {
        stats.batchEvalCalls += 1;
        evals = await evaluator.evaluateBatch(evalNodes.map((node) => node.board), contexts);
      } else {
        evals = await Promise.all(evalNodes.map((node, i) => evaluator.evaluate(node.board, contexts[i])));
      }
      recordEvaluatorMetricsDelta(evaluator, beforeMetrics, evalNodes.length, stats);
    }
    const values = evalNodes.map((node, i) => { stats.expansions += 1; return finishExpansion(node, evalMoves[i], evals[i], context); });
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
    throwIfAborted(signal);
    if (deadlineExpired(deadlineMs)) break;
    if (yieldEveryMs > 0 && nowMs() - lastYield >= yieldEveryMs) {
      await yieldToUi();
      lastYield = nowMs();
      throwIfAborted(signal);
    }
  }
}

export async function searchRoot(board: BoardState, evaluator: Evaluator, options: SearchOptions = {}): Promise<SearchResult> {
  const tSearch0 = nowMs();
  const visits = Math.max(1, Math.floor(options.visits ?? (options.movetimeMs && options.movetimeMs > 0 ? Number.MAX_SAFE_INTEGER : 8)));
  const context: SearchPolicyContext = {
    cpuct: options.cpuct ?? 1.5,
    fpu: options.fpu ?? 0,
    temperature: options.temperature ?? 1,
    cpuctSchedule: options.cpuctSchedule ?? 'constant',
    cpuctBase: options.cpuctBase ?? 38739,
    cpuctFactor: options.cpuctFactor ?? 3.894,
    fpuStrategy: options.fpuStrategy ?? 'constant',
    fpuReduction: options.fpuReduction ?? 0.330,
    avWeight: options.avWeight ?? 0.25,
    rankWeight: options.rankWeight ?? 0,
    regretWeight: options.regretWeight ?? 0,
    riskWeight: options.riskWeight ?? 0,
    uncertaintyWeight: options.uncertaintyWeight ?? 0,
    valueWdlAuxHead: options.valueWdlAuxHead,
    valueWdlAuxWeight: options.valueWdlAuxWeight ?? 0,
    valueWdlBlendMode: options.valueWdlBlendMode ?? 'constant',
    valueWdlBaseTemp: options.valueWdlBaseTemp ?? 1,
    valueWdlAuxTemp: options.valueWdlAuxTemp ?? 1,
    cpuctVarianceWeight: Math.max(0, options.cpuctVarianceWeight ?? 0),
    cpuctVarianceMaxScale: Math.max(1, options.cpuctVarianceMaxScale ?? 2),
    butterflyWeight: options.butterflyWeight ?? 0,
    butterflyDecay: Math.max(0, Math.min(1, options.butterflyDecay ?? 0.995)),
    butterflyMaxBonus: Math.max(0, options.butterflyMaxBonus ?? 0.25),
    butterflyTable: (options.butterflyWeight ?? 0) === 0 ? undefined : new Map<string, number>(),
  };
  const searchPolicy = options.searchPolicy ?? classicPuctPolicy;
  const stats = makeStats(visits);
  const requestedHistory = options.historyFens ?? [];
  const reusableRoot = !options.rootMoves && options.root && boardToFen(options.root.board) === boardToFen(board) && sameHistory(options.root.historyFens, requestedHistory) ? options.root : null;
  const root: Node = reusableRoot ?? makeSearchRoot(board, requestedHistory);
  root.isRoot = true;
  stats.rootReused = !!reusableRoot;
  options.transpositionTable?.set(searchNodeKey(root.board, root.historyFens), root);
  let rootValue: number;
  if (root.expanded) {
    rootValue = root.terminalValue ?? 0;
  } else if (options.rootMoves) {
    if (!options.rootMoves.length) {
      root.expanded = true;
      root.terminalValue = inCheck(root.board) ? -1 : 0;
      root.edges = [];
      stats.expansions += 1;
      stats.terminalHits += 1;
      rootValue = root.terminalValue;
    } else if (automaticDrawReason(root.board, root.historyFens)) {
      root.expanded = true;
      root.terminalValue = 0;
      root.edges = [];
      stats.expansions += 1;
      stats.terminalHits += 1;
      rootValue = 0;
    } else {
      const beforeMetrics = evaluatorMetrics(evaluator);
      stats.evalCalls += 1;
      const evaln = await evaluator.evaluate(root.board, { historyFens: root.historyFens, legalMoves: options.rootMoves });
      recordEvaluatorMetricsDelta(evaluator, beforeMetrics, 1, stats);
      rootValue = finishExpansion(root, options.rootMoves, evaln, context);
      stats.expansions += 1;
    }
  } else {
    rootValue = await expand(root, evaluator, context, stats);
  }
  if (!root.edges.length) {
    logSearchLatency('puct.search', { totalMs: nowMs() - tSearch0, requestedVisits: visits, completedVisits: 0, expansions: stats.expansions, terminalHits: stats.terminalHits, evalCalls: stats.evalCalls, batchEvalCalls: stats.batchEvalCalls, maxEvalBatch: stats.maxEvalBatch, stopReason: 'terminal-root' });
    return { move: null, visits: 0, value: rootValue, policy: [], stats };
  }
  const batchSize = Math.max(1, Math.floor(options.batchSize ?? 1));
  const signal = options.signal;
  const deadlineMs = deadlineFromMovetime(tSearch0, options.movetimeMs);
  const yieldEveryMs = Math.max(0, Math.floor(options.yieldEveryMs ?? 0));
  const priorRootVisits = reusableRoot ? (root.visits ?? root.edges.reduce((sum, edge) => sum + edge.visits, 0)) : 0;
  const visitsToRun = reusableRoot ? Math.max(0, visits - priorRootVisits) : visits;
  const budgetMode = options.budgetMode ?? 'visits';
  stats.budgetMode = budgetMode;
  const kldState: KldEarlyStopState = { stableChecks: 0 };
  const bestStableState: BestStableEarlyStopState = { stableChecks: 0 };
  const fixedVisitTarget = Math.max(visits, rootVisitCount(root));
  throwIfAborted(signal);
  if (budgetMode === 'neural') {
    stats.requestedNeuralEvals = visits;
    const cacheAware = evaluatorMetrics(evaluator) !== null;
    const maxRootVisits = priorRootVisits + Math.max(visits, Math.ceil(visits * Math.max(1, options.maxVisitsMultiplier ?? 4)));
    stats.maxRootVisits = maxRootVisits;
    if (!cacheAware) {
      stats.stopReason = 'no-cache-metrics-fixed-visits';
      if (batchSize > 1) await runBatchedVisits(root, evaluator, visitsToRun, searchPolicy, context, batchSize, stats, signal, yieldEveryMs, options.transpositionTable, deadlineMs, options.batchCollisionMode, options.batchCollisionRetryLimit);
      else {
        let lastYield = nowMs();
        for (let i = 0; i < visitsToRun && !deadlineExpired(deadlineMs); i++) {
          throwIfAborted(signal);
          await simulate(root, evaluator, searchPolicy, context, stats, options.transpositionTable);
          stats.completedVisits += 1;
          if (deadlineExpired(deadlineMs)) break;
          if (yieldEveryMs > 0 && nowMs() - lastYield >= yieldEveryMs) {
            await yieldToUi();
            lastYield = nowMs();
            throwIfAborted(signal);
          }
        }
      }
      if (deadlineExpired(deadlineMs) && stats.completedVisits < visitsToRun) stats.stopReason = 'movetime';
    } else {
      const startMisses = stats.neuralEvalMisses ?? 0;
      let lastYield = nowMs();
      while (((stats.neuralEvalMisses ?? 0) - startMisses) < visits && rootVisitCount(root) < maxRootVisits && !deadlineExpired(deadlineMs)) {
        throwIfAborted(signal);
        const room = Math.max(0, maxRootVisits - rootVisitCount(root));
        const chunk = Math.max(1, Math.min(batchSize, room));
        const beforeCompleted = stats.completedVisits;
        if (batchSize > 1) await runBatchedVisits(root, evaluator, chunk, searchPolicy, context, batchSize, stats, signal, yieldEveryMs, options.transpositionTable, deadlineMs, options.batchCollisionMode, options.batchCollisionRetryLimit);
        else {
          await simulate(root, evaluator, searchPolicy, context, stats, options.transpositionTable);
          stats.completedVisits += 1;
        }
        if (deadlineExpired(deadlineMs)) {
          stats.stopReason = 'movetime';
          break;
        }
        if (stats.completedVisits === beforeCompleted) break;
        const earlyStop = rootEarlyStopReason(root, context, options, maxRootVisits, kldState, bestStableState);
        if (earlyStop) {
          stats.stopReason = earlyStop;
          break;
        }
        if (yieldEveryMs > 0 && nowMs() - lastYield >= yieldEveryMs) {
          await yieldToUi();
          lastYield = nowMs();
          throwIfAborted(signal);
        }
      }
      if (!stats.stopReason) stats.stopReason = deadlineExpired(deadlineMs) ? 'movetime' : ((stats.neuralEvalMisses ?? 0) - startMisses) >= visits ? 'neural-budget' : 'max-visits';
    }
  } else if (batchSize > 1) {
    let done = 0;
    while (done < visitsToRun && !deadlineExpired(deadlineMs)) {
      const chunk = Math.min(batchSize, visitsToRun - done);
      await runBatchedVisits(root, evaluator, chunk, searchPolicy, context, batchSize, stats, signal, yieldEveryMs, options.transpositionTable, deadlineMs, options.batchCollisionMode, options.batchCollisionRetryLimit);
      done += chunk;
      if (deadlineExpired(deadlineMs)) break;
      const earlyStop = rootEarlyStopReason(root, context, options, fixedVisitTarget, kldState, bestStableState);
      if (earlyStop) {
        stats.stopReason = earlyStop;
        break;
      }
    }
    if (!stats.stopReason) stats.stopReason = deadlineExpired(deadlineMs) && done < visitsToRun ? 'movetime' : 'visit-budget';
  } else {
    let lastYield = nowMs();
    for (let i = 0; i < visitsToRun && !deadlineExpired(deadlineMs); i++) {
      throwIfAborted(signal);
      await simulate(root, evaluator, searchPolicy, context, stats, options.transpositionTable);
      stats.completedVisits += 1;
      if (deadlineExpired(deadlineMs)) break;
      const earlyStop = rootEarlyStopReason(root, context, options, fixedVisitTarget, kldState, bestStableState);
      if (earlyStop) {
        stats.stopReason = earlyStop;
        break;
      }
      if (yieldEveryMs > 0 && nowMs() - lastYield >= yieldEveryMs) {
        await yieldToUi();
        lastYield = nowMs();
        throwIfAborted(signal);
      }
    }
    if (!stats.stopReason) stats.stopReason = deadlineExpired(deadlineMs) && stats.completedVisits < visitsToRun ? 'movetime' : 'visit-budget';
  }
  root.visits = priorRootVisits + stats.completedVisits;
  const policy = searchPolicy.rootPolicy(root.edges, context, root);
  const bestEntry = searchPolicy.chooseFinalMove(policy, context);
  const pvDepth = options.pvDepth ?? 12;
  const pvSelector = options.pvSelector ?? 'visits';
  const principalVariation = options.includePv ? extractPrincipalVariation(root, context, searchPolicy, pvDepth, pvSelector) : undefined;
  const multiPvCount = Math.max(0, Math.floor(options.multiPv ?? 0));
  const multiPvLines = options.includePv && multiPvCount > 1
    ? extractMultiPv(root, context, searchPolicy, pvDepth, pvSelector, multiPvCount)
    : undefined;
  const realizedVisits = Math.max(root.visits, root.edges.reduce((sum, edge) => sum + edge.visits, 0));
  logSearchLatency('puct.search', {
    totalMs: nowMs() - tSearch0,
    requestedVisits: visits,
    realizedVisits,
    completedVisits: stats.completedVisits,
    batchSize,
    budgetMode: stats.budgetMode,
    stopReason: stats.stopReason,
    rootReused: stats.rootReused,
    evalCalls: stats.evalCalls,
    batchEvalCalls: stats.batchEvalCalls,
    maxEvalBatch: stats.maxEvalBatch,
    evalBatchSizeHistogram: stats.evalBatchSizeHistogram ?? {},
    expansions: stats.expansions,
    terminalHits: stats.terminalHits,
    cacheHits: stats.cacheHits,
    neuralEvalMisses: stats.neuralEvalMisses,
    transpositionHits: stats.transpositionHits ?? 0,
    batchLeafCollisions: stats.batchLeafCollisions ?? 0,
    batchLeafRetries: stats.batchLeafRetries ?? 0,
  });
  return { move: bestEntry?.move ?? null, visits: realizedVisits, value: bestEntry?.q ?? rootValue, policy, ...(principalVariation ? { principalVariation } : {}), ...(multiPvLines ? { multiPvLines } : {}), stats, root };
}

export async function chooseMove(board: BoardState, evaluator: Evaluator, options: SearchOptions = {}): Promise<SearchResult> {
  return searchRoot(board, evaluator, { ...options, temperature: options.temperature ?? 0 });
}
